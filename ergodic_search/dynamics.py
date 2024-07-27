#!/usr/bin/env python
# basic dynamics module for differential drive

import torch

# Dynamics model for computing trajectory given controls
class DiffDrive(torch.nn.Module):

    # Initialize the module
    def __init__(self, start_pose, traj_steps, init_controls=None):
        super(DiffDrive, self).__init__()
        
        if not isinstance(start_pose, torch.Tensor):
            start_pose = torch.tensor(start_pose)

        self.start_pose = start_pose
        self.traj_steps = traj_steps

        # initialize parameters (controls) for module
        if init_controls == None:
            self.controls = torch.nn.parameter.Parameter(torch.zeros((traj_steps, 2)))

        elif (init_controls.shape[0] != traj_steps):
            print("[INFO] Initial controls do not have correct length, initializing to zero")
            self.controls = torch.nn.parameter.Parameter(torch.zeros((traj_steps, 2)))

        else:
            if not isinstance(init_controls, torch.Tensor):
                init_controls = torch.tensor(init_controls)
            self.controls = torch.nn.parameter.Parameter(init_controls)

    # Compute the trajectory given the controls
    def forward(self, controls):
        if not isinstance(controls, torch.Tensor):
            controls = torch.tensor(controls)

        if controls.shape[0] != self.traj_steps:
            print("[ERROR] Controls with incorrect length provided to forward dynamics module. Returning with None.")
            return None

        tr = []
        xnew = self.start_pose
        for i in range(self.traj_steps):
            dx = torch.cos(xnew[2]) * torch.abs(controls[i,0])
            dy = torch.sin(xnew[2]) * torch.abs(controls[i,0])
            dth = controls[i,1]
            xnew = xnew + torch.tensor([dx, dy, dth])
            tr.append(xnew)
        
        traj = torch.stack(tr)
        return traj
    
    # Compute the inverse (given trajectory, compute controls)
    # assumes starting point is in the trajectory as the first point
    def inverse(self, traj):

        # translational velocity = difference between (x,y) points along trajectory
        traj_diff = torch.diff(traj, axis=0)
        trans_vel = torch.sqrt(torch.sum(traj_diff[:,:2]**2, axis=1))

        # angular velocity = difference between angles, with first computed from starting point
        ang_vel = traj_diff[:,2]

        controls = torch.cat((trans_vel, ang_vel), axis=1)
        return controls