#!/usr/bin/env python
# basic dynamics and simple differential drive modules

import torch

# base class for dynamics modules
# sets up parameters and converts inputs as needed
class DynModule(torch.nn.Module):

    def __init__(self, start_pose, traj_steps, init_controls=None, device="cpu"):

        super(DynModule, self).__init__()

        if not isinstance(start_pose, torch.Tensor):
            start_pose = torch.tensor(start_pose, requires_grad=True)

        if init_controls is None:
            init_controls = torch.zeros((traj_steps, 2), requires_grad=True)
        
        elif init_controls.shape[0] != traj_steps:
            print("[INFO] Initial controls do not have correct length, initializing to zero")
            init_controls = torch.zeros((traj_steps, 2), requires_grad=True)

        if not isinstance(init_controls, torch.Tensor):
            init_controls = torch.tensor(init_controls, requires_grad=True)

        self.device = device
        self.traj_steps = traj_steps
        self.controls = torch.nn.Parameter(init_controls)
        self.register_buffer("start_pose", start_pose)


# Dynamics model for computing trajectory given controls
class DiffDrive(DynModule):

    # Compute the trajectory given the controls
    def forward(self):

        # compute theta based on propagating forward the angular velocities
        theta = self.start_pose[2] + torch.cumsum(self.controls[:,1], axis=0)

        # compute x and y based on thetas and controls
        x = self.start_pose[0] + torch.cumsum(torch.cos(theta) * torch.abs(self.controls[:,0]), axis=0)
        y = self.start_pose[1] + torch.cumsum(torch.sin(theta) * torch.abs(self.controls[:,0]), axis=0)

        traj = torch.stack((x, y, theta), dim=1)
        
        return traj
    
    # Compute the inverse (given trajectory, compute controls)
    def inverse(self, traj):

        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj, device=self.device)

        # add start point to trajectory
        traj_with_start = torch.cat((self.start_pose.unsqueeze(0), traj), axis=0)

        # translational velocity = difference between (x,y) points along trajectory
        traj_diff = torch.diff(traj_with_start, axis=0)
        trans_vel = torch.sqrt(torch.sum(traj_diff[:,:2]**2, axis=1))

        # angular velocity = difference between angles, with first computed from starting point
        ang_vel = traj_diff[:,2]

        controls = torch.cat((trans_vel.unsqueeze(1), ang_vel.unsqueeze(1)), axis=1)
        return controls
