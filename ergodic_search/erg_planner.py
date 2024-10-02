#!/usr/bin/env python
# class for performing ergodic trajectory optimization
# given a spatial distribution and starting location

import os
import argparse
import copy
import torch
import matplotlib.pyplot as plt

from ergodic_search import erg_metric
from ergodic_search.dynamics import DiffDrive


# parameters that can be changed
def ErgArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_pixels', type=int, default=500, help='Number of pixels along one side of the map')
    parser.add_argument('--gpu', action='store_true', help='Flag for using the GPU instead of CPU')
    parser.add_argument('--traj_steps', type=int, default=100, help='Number of steps in trajectory')
    parser.add_argument('--iters', type=int, default=1000, help='Maximum number of iterations for trajectory optimization')
    parser.add_argument('--epsilon', type=float, default=0.005, help='Threshold for ergodic metric (if lower than this, optimization stops)')
    parser.add_argument('--start_pose', type=float, nargs=3, default=[0.,0.,0.], help='Starting position in x, y, theta')
    parser.add_argument('--end_pose', type=float, nargs=3, default=[0.,0.,0.], help='Ending position in x, y, theta')
    parser.add_argument('--num_freqs', type=int, default=0, help='Number of frequencies to use. If 0, expects fourier_freqs provided.')
    parser.add_argument('--erg_wt', type=float, default=1, help='Weight on ergodic metric in loss function')
    parser.add_argument('--transl_vel_wt', type=float, default=0.1, help='Weight on translational velocity control size in loss function')
    parser.add_argument('--ang_vel_wt', type=float, default=0.05, help='Weight on angular velocity control size in loss function')
    parser.add_argument('--bound_wt', type=float, default=1000, help='Weight on boundary condition in loss function')
    parser.add_argument('--end_pose_wt', type=float, default=0.5, help='Weight on end position in loss function')
    parser.add_argument('--debug', action='store_true', help='Whether to print loss components for debugging')
    parser.add_argument('--outpath', type=str, help='File path to save images to, None displays them in a window', default=None)
    args = parser.parse_args()
    print(args)

    # check if outpath exists and make it if not
    if args.outpath is not None and os.path.exists(args.outpath) == False:
        os.mkdir(args.outpath)

    return args


# ergodic planner
class ErgPlanner():

    # initialize planner
    def __init__(self, args, pdf=None, init_controls=None, dyn_model=None, fourier_freqs=None, freq_wts=None):
        
        # store information
        self.args = args
        self.pdf = pdf
        self.fourier_freqs = fourier_freqs
        self.freq_wts = freq_wts

        # get device
        dev = torch.device("cuda") if args.gpu else torch.device("cpu")

        # convert starting and ending positions to tensors
        self.start_pose = torch.tensor(self.args.start_pose, requires_grad=True, device=dev)
        self.end_pose = torch.tensor(self.args.end_pose, requires_grad=True, device=dev)

        # flatten pdf if needed
        if pdf is not None and len(pdf.shape) > 1:
            self.pdf = pdf.flatten()

        # dynamics module
        if not isinstance(init_controls, torch.Tensor):
            init_controls = torch.tensor(init_controls)
        init_controls.requires_grad = True

        if dyn_model is not None:
            if init_controls is not None:
                # update initial controls in the model if not none
                for n,p in dyn_model.state_dict().items():
                    if n == 'controls':
                        p.copy_(init_controls)
            self.dyn_model = dyn_model

        else:
            print("Using DiffDrive dynamics model")
            self.dyn_model = DiffDrive(self.start_pose, self.args.traj_steps, init_controls, device=dev)

        # loss module
        self.loss = erg_metric.ErgLoss(self.args, self.dyn_model, self.pdf, fourier_freqs, freq_wts)

        # optimizer
        self.optimizer = torch.optim.Adam(self.dyn_model.parameters(), lr=self.args.learn_rate)

        # initialize empty trajectory and step counter
        self.prev_traj = torch.empty((1, 2))
        self.step_counter = 0


    # update the spatial distribution and store it in the loss computation module
    def update_pdf(self, pdf, fourier_freqs=None, freq_wts=None):
        if len(pdf.shape) > 1:
            self.pdf = pdf.flatten()
        self.fourier_freqs = fourier_freqs
        self.freq_wts = freq_wts
        self.loss.update_pdf(self.pdf, self.fourier_freqs, self.freq_wts)


    # update the controls
    def update_controls(self, new_controls):

        # check that the new controls have the correct type
        if not isinstance(new_controls, torch.Tensor):
            new_controls = torch.tensor(new_controls, requires_grad = True)

        # update the parameter
        for n,p in self.dyn_model.state_dict().items():
            if n == 'controls':
                p.copy_(new_controls)

    
    # take a step along the trajectory
    def take_step(self):

        # save previous trajectory
        with torch.no_grad():
            prev_traj = self.dyn_model.forward().detach()

        # update the starting point
        self.dyn_model.start_pose = prev_traj[0,:]

        # update the stored trajectory
        if self.step_counter > 0:
            self.prev_traj = torch.cat((self.prev_traj, prev_traj[0,:].unsqueeze(0)))
        else:
            self.prev_traj = prev_traj[0,:].unsqueeze(0)

        # update the controls
        # last set will be the same as the final set of controls from previous iteration
        prev_controls = self.dyn_model.controls.detach()
        new_controls = torch.roll(prev_controls, -1, 0)
        new_controls[-1,:] = prev_controls[-1,:]
        self.update_controls(new_controls)

        # increment step counter
        self.step_counter += 1


    # compute ergodic trajectory over spatial distribution
    def compute_traj(self, debug=False):
        
        # iterate
        for i in range(self.args.iters):
            self.optimizer.zero_grad()
            erg = self.loss(print_flag=debug)

            # print progress every 100th iter
            if i % 100 == 0 and not debug:
                print("[INFO] Iteration {:d} of {:d}, ergodic metric is {:4.4f}".format(i, self.args.iters, erg))
            
            # if ergodic metric is low enough, quit
            if erg < self.args.epsilon:
                break
            
            erg.backward()
            self.optimizer.step()

        # final controls and trajectory
        with torch.no_grad():
            controls = self.dyn_model.controls.detach()
            traj = self.dyn_model.forward()

        print("[INFO] Final ergodic metric is {:4.4f}".format(erg))

        # return controls, trajectory, and final ergodic metric
        return controls, traj, erg


    # visualize the output
    def visualize(self, img_name='results', cmap='viridis'):

        plt.rcParams['figure.figsize'] = [10,15]

        with torch.no_grad():
            traj = self.dyn_model.forward().detach()

        traj_np = traj.numpy()
        traj_recon = self.loss.traj_recon(traj).reshape((self.args.num_pixels, self.args.num_pixels))
        map_recon = self.loss.map_recon.detach().reshape((self.args.num_pixels, self.args.num_pixels))

        _, ax = plt.subplots(2,2)

        # original map with trajectory
        ax[0,0].imshow(self.pdf.reshape((self.args.num_pixels, self.args.num_pixels)), extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[0,0].set_title('Original Map and Trajectory')
        ax[0,0].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        # reconstructed map from map stats
        ax[1,0].imshow(map_recon, extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[1,0].set_title('Reconstructed Map from Map Stats')
        ax[1,0].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        # error between traj stats and map stats
        ax[0,1].imshow(map_recon - traj_recon, extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[0,1].set_title('Reconstruction Difference (Map - Traj)')
        ax[0,1].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        # reconstructed map from trajectory stats
        ax[1,1].imshow(traj_recon, extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[1,1].set_title('Reconstructed Map from Traj Stats')
        ax[1,1].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        if self.step_counter > 0:
            prev_traj_np = self.prev_traj.numpy()
            ax[0,0].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)
            ax[1,0].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)
            ax[0,1].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)
            ax[1,1].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)

        if self.args.outpath is not None:
            plt.savefig(self.args.outpath+'/'+img_name+'.png', dpi=100, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
