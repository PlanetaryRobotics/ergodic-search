#!/usr/bin/env python
# class for performing ergodic trajectory optimization
# given a spatial distribution and starting location

import os
import argparse
import copy
import torch
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler

from ergodic_search import erg_metric
from ergodic_search.dynamics import DiffDrive

from tqdm import tqdm


# parameters that can be changed
def ErgArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', type=float, nargs='+', default=[0.001], help='Learning rate for optimizer')
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
    parser.add_argument('--replan_type', type=str, default='full', help='Type of replanning to perform (accepts partial or full)')
    parser.add_argument('--sensor_variance', type=float, default=0.01, help='Variance of the Gaussian sensor footprint')

    args = parser.parse_args()
    print(args)

    # check if outpath exists and make it if not
    if args.outpath is not None and os.path.exists(args.outpath) == False:
        os.mkdir(args.outpath)

    # check that replan type is one of options
    if args.replan_type != 'full' and args.replan_type != 'partial':
        print('Incorrect replan_type, must be one of full or partial. Planner will not perform as expected.')

    return args


# ergodic planner
class ErgPlanner():

    # initialize planner
    def __init__(self, args, pdf=None, init_controls=None, dyn_model=None, fourier_freqs=None, freq_wts=None):
        
        # check learning rate args
        if len(args.learn_rate) > 3:
            print("Too many values provided to args.learn_rate, using first 3 with linear LR scheduler")
            args.learn_rate = args.learn_rate[0:3]

        # store information
        self.args = args
        self.pdf = pdf

        # get device
        self.device = torch.device("cuda") if args.gpu else torch.device("cpu")

        # convert starting and ending positions to tensors
        self.start_pose = torch.tensor(self.args.start_pose, device=self.device)
        self.end_pose = torch.tensor(self.args.end_pose, device=self.device)

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
            self.dyn_model.to(self.device)

        else:
            print("Using DiffDrive dynamics model")
            self.dyn_model = DiffDrive(self.start_pose, self.args.traj_steps, init_controls, device=self.device)

        # loss module
        self.loss = erg_metric.ErgLoss(self.args, self.dyn_model, self.pdf, fourier_freqs, freq_wts)
        self.loss.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.dyn_model.parameters(), lr=self.args.learn_rate[0])

        # set up learning rate scheduler if more than one value provided for args.learn_rate
        self.scheduler = None
        if len(self.args.learn_rate) == 2:
            print("Using exponential learning rate scheduler")
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.learn_rate[1])

        elif len(self.args.learn_rate) == 3:
            print("Using linear learning rate scheduler")
            endf = self.args.learn_rate[1] / self.args.learn_rate[0]
            self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1, end_factor=endf, total_iters=self.args.learn_rate[2])

        # initialize empty trajectory and step counter
        self.prev_traj = torch.empty((1, 2))
        self.step_counter = 0

        # register hook for partial replanning gradient edit
        self.loss.dyn_model.controls.register_hook(lambda grad: self.grad_hook(grad, self.step_counter))

    # update the spatial distribution and store it in the loss computation module
    def update_pdf(self, pdf, fourier_freqs=None, freq_wts=None):
        if len(pdf.shape) > 1:
            self.pdf = pdf.flatten()
        self.loss.update_pdf(self.pdf, fourier_freqs, freq_wts)


    # update the controls
    def update_controls(self, new_controls):

        # check that the new controls have the correct type
        if not isinstance(new_controls, torch.Tensor):
            new_controls = torch.tensor(new_controls, requires_grad=True, device=new_controls.device)

        # update the parameter
        for n,p in self.dyn_model.state_dict().items():
            if n == 'controls':
                p.copy_(new_controls)

    
    # take a step along the trajectory
    def take_step(self):

        # save previous trajectory
        with torch.no_grad():
            prev_traj = self.dyn_model.forward().detach()

        if self.args.replan_type == 'partial':

            # update stored previous trajectory
            self.prev_traj = prev_traj[0:self.step_counter+1,:]

        elif self.args.replan_type == 'full':

            # append to stored previous trajectory
            if self.step_counter > 0:
                self.prev_traj = torch.cat((self.prev_traj, prev_traj[0,:].unsqueeze(0)))
            else:
                self.prev_traj = prev_traj[0,:].unsqueeze(0)

            # update the starting point
            self.dyn_model.start_pose = prev_traj[0,:]

            # update the controls
            # last set will be the same as the final set of controls from previous iteration
            prev_controls = self.dyn_model.controls.detach()
            new_controls = torch.roll(prev_controls, -1, 0)
            new_controls[-1,:] = prev_controls[-1,:]
            self.update_controls(new_controls)

        else:
            print('Incorrect replanning type used, not storing previous trajectory or incrementing steps')
            return

        # increment step counter
        self.step_counter += 1


    def compute_traj(self, debug=False):
        # iterate with a progress bar that updates with the current ergodic metric
        with tqdm(range(self.args.iters), desc="Trajectory optimization in progress...") as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                erg = self.loss(print_flag=debug)
                
                # update the progress bar with the current ergodic metric
                pbar.set_postfix({'erg': erg.item()})
                
                # if ergodic metric is low enough, quit early
                if erg < self.args.epsilon:
                    break

                # optimize the controls
                erg.backward(inputs=self.loss.dyn_model.controls)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        # final controls and trajectory after optimization
        with torch.no_grad():
            controls = self.dyn_model.controls.detach().cpu()
            traj = self.dyn_model.forward().detach().cpu()
            erg = erg.cpu()

        print("[INFO] Final ergodic metric is {:4.4f}".format(erg))
        return controls, traj, erg



    # visualize the output
    def visualize(self, img_name='results', cmap='viridis'):

        with torch.no_grad():
            traj = self.dyn_model.forward().detach()
            traj_recon = self.loss.traj_recon(traj).cpu()
            map_recon = self.loss.map_recon.detach().cpu()

        traj_np = traj.cpu().numpy()
        traj_recon = traj_recon.reshape((self.args.num_pixels, self.args.num_pixels))
        map_recon = map_recon.reshape((self.args.num_pixels, self.args.num_pixels))

        fig, ax = plt.subplots(2,2)
        fig.set_size_inches(10, 10)

        s = self.step_counter if self.args.replan_type == 'partial' else 0

        # original map with trajectory
        ax[0,0].imshow(self.pdf.reshape((self.args.num_pixels, self.args.num_pixels)), extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[0,0].set_title('Original Map and Trajectory')
        ax[0,0].scatter(traj_np[s:,0], traj_np[s:,1], c='r', s=2)

        # reconstructed map from map stats
        ax[1,0].imshow(map_recon, extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[1,0].set_title('Reconstructed Map from Map Stats')
        ax[1,0].scatter(traj_np[s:,0], traj_np[s:,1], c='r', s=2)

        # error between traj stats and map stats
        ax[0,1].imshow(map_recon - traj_recon, extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[0,1].set_title('Reconstruction Difference (Map - Traj)')
        ax[0,1].scatter(traj_np[s:,0], traj_np[s:,1], c='r', s=2)

        # reconstructed map from trajectory stats
        ax[1,1].imshow(traj_recon, extent=[0,1,0,1], origin='lower', cmap=cmap)
        ax[1,1].set_title('Reconstructed Map from Traj Stats')
        ax[1,1].scatter(traj_np[s:,0], traj_np[s:,1], c='r', s=2)

        if self.step_counter > 0:
            prev_traj_np = self.prev_traj.cpu().numpy()
            ax[0,0].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)
            ax[1,0].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)
            ax[0,1].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)
            ax[1,1].scatter(prev_traj_np[:,0], prev_traj_np[:,1], c='k', s=2)

        if self.args.outpath is not None:
            plt.savefig(self.args.outpath+'/'+img_name+'.png', dpi=100, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # hook for setting gradient for certain controls equal to 0
    # used for partial replanning
    def grad_hook(self, grad, s):
        if self.args.replan_type == 'partial':
            new_grad = torch.cat((torch.zeros((s, 2)).to(self.device), grad[s:,:]))
            return new_grad
