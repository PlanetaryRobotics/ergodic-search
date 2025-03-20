#!/usr/bin/env python
# basic dynamics and simple differential drive modules

import torch

# base class for dynamics modules
# sets up parameters and converts inputs as needed
class DynModule(torch.nn.Module):

    def __init__(self, start_pose, traj_steps, init_controls=None, device="cpu", v_max=0.1, omega_max=0.05):

        self.v_max = v_max
        self.omega_max = omega_max

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

        # Apply a tanh activation to enforce velocity limits
        clamped_controls = torch.clone(self.controls)
        clamped_controls[:, 0] = self.v_max * torch.tanh(clamped_controls[:, 0])  # Translational velocity
        clamped_controls[:, 1] = self.omega_max * torch.tanh(clamped_controls[:, 1])  # Angular velocity

        # Compute theta based on propagating forward the angular velocities
        theta = self.start_pose[2] + torch.cumsum(clamped_controls[:, 1], axis=0)

        # Compute x and y based on thetas and controls
        x = self.start_pose[0] + torch.cumsum(torch.cos(theta) * clamped_controls[:, 0], axis=0)
        y = self.start_pose[1] + torch.cumsum(torch.sin(theta) * clamped_controls[:, 0], axis=0)

        traj = torch.stack((x, y, theta), dim=1)
        
        return traj
    
    # Compute the inverse (given trajectory, compute controls)
    def inverse(self, traj):

        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj, device=self.device)

        # Add start point to trajectory
        traj_with_start = torch.cat((self.start_pose.unsqueeze(0), traj), axis=0)

        # Translational velocity = difference between (x,y) points along trajectory
        traj_diff = torch.diff(traj_with_start, axis=0)
        trans_vel = torch.sqrt(torch.sum(traj_diff[:, :2]**2, axis=1))

        # Angular velocity = difference between angles
        ang_vel = traj_diff[:, 2]

        # Directly clamp velocities to enforce limits
        trans_vel = torch.clamp(trans_vel, -self.v_max, self.v_max)
        ang_vel = torch.clamp(ang_vel, -self.omega_max, self.omega_max)

        controls = torch.cat((trans_vel.unsqueeze(1), ang_vel.unsqueeze(1)), axis=1)
        return controls
