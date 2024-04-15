#!/usr/bin/env python
# class for computing the ergodic metric over a trajectory 
# given a spatial distribution

import copy
from xml.sax.handler import DTDHandler
import torch

# Dynamics model for computing trajectory given controls
class DynModel(torch.nn.Module):

    # Initialize the module
    def __init__(self, start_pose, traj_steps, init_controls=None):
        super(DynModel, self).__init__()
        
        self.start_pose = start_pose
        self.traj_steps = traj_steps

        # initialize parameters (controls) for module
        control_cond = (init_controls.shape[0] != traj_steps)
        if control_cond: print("[INFO] Initial controls does not have correct length, initializing to zero")
        if init_controls == None or control_cond:
            self.controls = torch.nn.parameter.Parameter(torch.zeros((traj_steps, 2)))
        else:
            self.controls = torch.nn.parameter.Parameter(torch.tensor(init_controls))

    # Compute the trajectory given the controls
    def forward(self):
        traj = torch.zeros((self.traj_steps+1,3))
        traj[0,:] = copy.copy(self.start_pose)
        for i in self.traj_steps:
            prev_pos = copy.copy(traj[i,:])
            dx = torch.abs(self.controls[i,0]) * torch.cos(prev_pos[2])
            dy = torch.abs(self.controls[i,0]) * torch.sin(prev_pos[2])
            dth = 10*self.controls[1] # TODO: why multiplied by 10?
            traj[i+1,:] = traj[i,:] + torch.tensor([dx, dy, dth])

        return self.controls, traj


# Module for computing ergodic loss over a PDF
class ErgLoss(torch.nn.Module):

    def __init__(self, args, pdf=None, fourier_freqs=None, freq_wts=None):
        super(ErgLoss, self).__init__()
        self.args = args
        self.init_flag=False

        if fourier_freqs is not None: self.fourier_freqs = fourier_freqs
        if freq_wts is not None: self.freq_wts = freq_wts

        if pdf is not None:
            self.pdf = pdf
            self.set_up_calcs()

    # compute the ergodic metric
    def forward(self, controls, traj, print_flag=False):

        # confirm we can do this
        if ~self.init_flag:
            print("[ERROR] Ergodic loss module not initialized properly, need to provide map before attempting to calculate. Returning with None.")
            return None

        # trajectory statistics
        fk = torch.prod(torch.cos(traj[:,0:2] * self.k), dim=1) # TODO: need to check
        ck = torch.mean(fk, dim=1)
        ck = ck / self.hk

        # ergodic metric
        erg_metric = torch.sum(self.lamk * torch.square(self.phik - ck))

        # controls regularizer
        control_metric = torch.mean(controls**2, dim=0)

        # boundary condition counts number of points out of bounds
        bound_metric = torch.sum(torch.ceil(torch.maximum(0, traj[:,0:2]-1)) + torch.ceil(torch.maximum(0, -traj[:,0:2])))

        # print info if desired
        if print_flag:
            print("LOSS: erg = {:4.4f}, control = ({:4.4f}, {:4.4f}), boundary = {:4.4f}".format(erg_metric, control_metric[0], control_metric[1], bound_metric))

        return (self.args.erg_wt * erg_metric) + (self.args.transl_vel_wt * control_metric[0]) + (self.args.ang_vel_wt * control_metric[1]) + (self.args.bound_wt * bound_metric)

    # Update the stored map
    def update_pdf(self, pdf, fourier_freqs=None, freq_wts=None):
        self.pdf = pdf
        if fourier_freqs is not None: self.fourier_freqs = fourier_freqs
        if freq_wts is not None: self.freq_wts = freq_wts
        self.set_up_calcs()

    # set up calculations related to pdf
    def set_up_calcs(self):

        # frequencies to use
        if self.fourier_freqs is not None:
            pass

        # weights to use

        # compute map stats

        # map stats for reconstruction

        # compute hk
        self.hk = torch.sqrt(torch.prod((2.0*self.k + torch.sin(2.0*self.k)) / (4.0*self.k)), dim=1) # TODO: check

        # set flag to true so we know we can compute the metric
        self.init_flag = True
