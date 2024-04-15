#!/usr/bin/env python
# class for performing ergodic trajectory optimization
# given a spatial distribution and starting location

import argparse
import copy
import torch
import numpy as np
from ergodic_search import erg_metric

# parameters that can be changed
def ErgArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_pixels', type=int, default=500, help='Number of pixels along one side of the map')
    parser.add_argument('--gpu', action='store_true', help='Flag for using the GPU instead of CPU')
    parser.add_argument('--traj_steps', type=int, default=100, help='Number of steps in trajectory')
    parser.add_argument('--iters', type=int, default=1000, help='Maximum number of iterations for trajectory optimization')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Threshold for ergodic metric (if lower than this, optimization stops)')
    parser.add_argument('--start_pose', type=float, nargs=3, default=[0,0,0], help='Starting position in x, y, theta')
    parser.add_argument('--num_freqs', type=int, default=10, help='Number of frequencies to use, if frequencies not provided')
    parser.add_argument('--erg_wt', type=float, default=1, help='Weight on ergodic metric in loss function')
    parser.add_argument('--transl_vel_wt', type=float, default=0.02, help='Weight on translational velocity control size in loss function')
    parser.add_argument('--ang_vel_wt', type=float, default=0.02, help='Weight on angular velocity control size in loss function')
    parser.add_argument('--bound_wt', type=float, default=1000, help='Weight on boundary condition in loss function')
    args = parser.parse_args()
    return args

# ergodic planner
class ErgPlanner(object):

    # initialize planner
    def __init__(self, args, pdf=None, init_controls=None, dyn_model=None, fourier_freqs=None, freq_wts=None, ):
        
        # store information
        self.args = args
        self.pdf = pdf
        self.fourier_freqs = fourier_freqs
        self.freq_wts = freq_wts

        # set up pdf, dynamics model, and loss module
        if pdf is not None and len(pdf.shape) > 1:
            self.pdf = pdf.flatten()

        if dyn_model is None:
            self.dyn_model = erg_metric.DynModel(self.args.start_pose, self.args.traj_steps, init_controls)
        else:
            self.dyn_model = dyn_model(self.args.start_pose, self.args.traj_steps, init_controls)

        self.optimizer = torch.optim.Adam(self.dyn_model.parameters(), lr=self.lr)
        self.loss = erg_metric.ErgLoss(self.args, pdf, fourier_freqs, freq_wts)

    # update the spatial distribution and store it in the loss computation module
    def update_pdf(self, pdf, fourier_freqs, freq_wts):
        if len(pdf.shape) > 1:
            self.pdf = pdf.flatten()
        self.fourier_freqs = fourier_freqs
        self.freq_wts = freq_wts
        self.loss.update_pdf(self.pdf, self.fourier_freqs, self.freq_wts)

    # compute ergodic trajectory over spatial distribution
    def compute_traj(self):
        
        # iterate
        for i in range(self.args.iters):
            self.optimizer.zero_grad()
            controls, traj = self.dyn_model()
            erg = self.loss(controls, traj)

            # print progress every 100th iter
            if i % 100 == 0:
                print("[INFO] Iteration {:d} of {:d}, ergodic metric is {:4.4f}".format(i, self.args.iters, erg))
            
            # if ergodic metric is low enough, quit
            if erg < self.args.epsilon:
                break
            
            erg.backward()
            self.optimizer.step()

        # return controls, trajectory, and final ergodic metric
        return controls, traj, erg

