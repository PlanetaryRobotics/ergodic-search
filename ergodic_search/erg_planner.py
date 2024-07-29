#!/usr/bin/env python
# class for performing ergodic trajectory optimization
# given a spatial distribution and starting location

import argparse
import copy
import torch
import numpy as np
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
    parser.add_argument('--start_pose', type=float, nargs=3, default=[0,0,0], help='Starting position in x, y, theta')
    parser.add_argument('--end_pose', type=float, nargs=3, default=[0,0,0], help='Ending position in x, y, theta')
    parser.add_argument('--num_freqs', type=int, default=0, help='Number of frequencies to use. If 0, expects fourier_freqs provided.')
    parser.add_argument('--erg_wt', type=float, default=1, help='Weight on ergodic metric in loss function')
    parser.add_argument('--transl_vel_wt', type=float, default=0.1, help='Weight on translational velocity control size in loss function')
    parser.add_argument('--ang_vel_wt', type=float, default=0.05, help='Weight on angular velocity control size in loss function')
    parser.add_argument('--bound_wt', type=float, default=1000, help='Weight on boundary condition in loss function')
    parser.add_argument('--end_pose_wt', type=float, default=0.5, help='Weight on end position in loss function')
    parser.add_argument('--debug', action='store_true', help='Whether to print loss components for debugging')
    args = parser.parse_args()
    print(args)
    return args

# ergodic planner
class ErgPlanner():

    # initialize planner
    def __init__(self, args, pdf=None, init_controls=None, dyn_model=None, fourier_freqs=None, freq_wts=None, ):
        
        # store information
        self.args = args
        self.pdf = pdf
        self.fourier_freqs = fourier_freqs
        self.freq_wts = freq_wts

        # convert starting and ending positions to tensors
        self.start_pose = torch.tensor(self.args.start_pose, requires_grad=True)

        # set up pdf, dynamics model, and loss module
        if pdf is not None and len(pdf.shape) > 1:
            self.pdf = pdf.flatten()

        if dyn_model is None:
            self.dyn_model = DiffDrive(self.start_pose, self.args.traj_steps)
        else:
            self.dyn_model = dyn_model

        # initialize parameters (controls) for module
        if init_controls is None:
            self.controls = torch.zeros((args.traj_steps, 2), requires_grad=True)

        elif (init_controls.shape[0] != args.traj_steps):
            print("[INFO] Initial controls do not have correct length, initializing to zero")
            self.controls = torch.zeros((args.traj_steps, 2), requires_grad=True)

        else:
            if not isinstance(init_controls, torch.Tensor):
                init_controls = torch.tensor(init_controls, requires_grad=True)
            self.controls = init_controls

        self.optimizer = torch.optim.Adam([self.controls], lr=self.args.learn_rate)
        self.loss = erg_metric.ErgLoss(self.args, dyn_model, self.pdf, fourier_freqs, freq_wts)


    # update the spatial distribution and store it in the loss computation module
    def update_pdf(self, pdf, fourier_freqs, freq_wts):
        if len(pdf.shape) > 1:
            self.pdf = pdf.flatten()
        self.fourier_freqs = fourier_freqs
        self.freq_wts = freq_wts
        self.loss.update_pdf(self.pdf, self.fourier_freqs, self.freq_wts)

    # compute ergodic trajectory over spatial distribution
    def compute_traj(self, debug=False):
        
        # iterate
        for i in range(self.args.iters):
            self.optimizer.zero_grad()
            erg = self.loss(self.controls, print_flag=debug)

            # print progress every 100th iter
            if i % 100 == 0 and not debug:
                print("[INFO] Iteration {:d} of {:d}, ergodic metric is {:4.4f}".format(i, self.args.iters, erg))
            
            # if ergodic metric is low enough, quit
            if erg < self.args.epsilon:
                break
            
            erg.backward()
            
            # print(erg.grad)
            self.optimizer.step()

        # final controls and trajectory
        with torch.no_grad():
            self.controls = self.controls.detach()
            self.traj = self.loss.dyn_model.forward(self.controls)

        print("[INFO] Final ergodic metric is {:4.4f}".format(erg))

        # return controls, trajectory, and final ergodic metric
        return self.controls, self.traj, erg

    # visualize the output
    def visualize(self):

        plt.rcParams['figure.figsize'] = [10,15]

        traj_np = self.traj.detach().numpy()
        traj_recon = self.loss.traj_recon(self.traj.detach()).reshape((self.args.num_pixels, self.args.num_pixels))
        map_recon = self.loss.map_recon.detach().reshape((self.args.num_pixels, self.args.num_pixels))

        _, ax = plt.subplots(2,2)

        # original map with trajectory
        ax[0,0].imshow(self.pdf.reshape((self.args.num_pixels, self.args.num_pixels)), extent=[0,1,0,1], origin='lower')
        ax[0,0].set_title('Original Map and Trajectory')
        ax[0,0].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        # reconstructed map from map stats
        ax[1,0].imshow(map_recon, extent=[0,1,0,1], origin='lower')
        ax[1,0].set_title('Reconstructed Map from Map Stats')
        ax[1,0].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        # reconstructed map from trajectory stats
        ax[1,1].imshow(traj_recon.T, extent=[0,1,0,1], origin='lower')
        ax[1,1].set_title('Reconstructed Map from Traj Stats')
        ax[1,1].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        # error between traj stats and map stats
        ax[0,1].imshow(map_recon - traj_recon.T, extent=[0,1,0,1], origin='lower')
        ax[0,1].set_title('Reconstruction Difference (Map - Traj)')
        ax[0,1].scatter(traj_np[:,0], traj_np[:,1], c='r', s=2)

        plt.show()
        plt.close()
