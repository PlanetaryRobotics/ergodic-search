#!/usr/bin/env python
# basic dynamics module for differential drive

import torch

# Dynamics model for computing trajectory given controls
class DiffDrive():

    # Initialize the module
    def __init__(self, start_pose, traj_steps):
        
        if not isinstance(start_pose, torch.Tensor):
            start_pose = torch.tensor(start_pose, requires_grad=True)

        self.start_pose = start_pose
        self.traj_steps = traj_steps


    # Compute the trajectory given the controls
    def forward(self, controls):
        if not isinstance(controls, torch.Tensor):
            controls = torch.tensor(controls)

        if controls.shape[0] != self.traj_steps:
            print("[ERROR] Controls with incorrect length provided to forward dynamics module. Returning with None.")
            return None

        # compute theta based on propagating forward the angular velocities
        theta = self.start_pose[2] + torch.cumsum(controls[:,1], axis=0)

        # compute x and y based on thetas and controls
        x = self.start_pose[0] + torch.cumsum(torch.cos(theta) * torch.abs(controls[:,0]), axis=0)
        y = self.start_pose[1] + torch.cumsum(torch.sin(theta) * torch.abs(controls[:,0]), axis=0)

        traj = torch.stack((x, y, theta), dim=1)
        
        return traj
    

    # Compute the inverse (given trajectory, compute controls)
    def inverse(self, traj):

        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj)

        # add start point to trajectory
        traj_with_start = torch.cat((self.start_pose.unsqueeze(0), traj), axis=0)

        # translational velocity = difference between (x,y) points along trajectory
        traj_diff = torch.diff(traj_with_start, axis=0)
        trans_vel = torch.sqrt(torch.sum(traj_diff[:,:2]**2, axis=1))

        # angular velocity = difference between angles, with first computed from starting point
        ang_vel = traj_diff[:,2]

        controls = torch.cat((trans_vel.unsqueeze(1), ang_vel.unsqueeze(1)), axis=1)
        return controls