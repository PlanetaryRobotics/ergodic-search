#!/usr/bin/env python
# class for computing the ergodic metric over a trajectory 
# given a spatial distribution

import copy
import torch

from functools import partial


# Module for computing ergodic loss over a PDF
class ErgLoss(torch.nn.Module):

    def __init__(self, args, pdf=None, fourier_freqs=None, freq_wts=None):
        super(ErgLoss, self).__init__()
        self.args = args
        self.init_flag=False

        if args.num_freqs == 0 and fourier_freqs is None:
            print("[ERROR] args.num_freqs needs to be positive or fourier_freqs must be provided. Returning with None.")
            return None

        if fourier_freqs is not None:
            if not isinstance(fourier_freqs, torch.Tensor):
                fourier_freqs = torch.tensor(fourier_freqs)
        self.fourier_freqs = fourier_freqs
        
        if freq_wts is not None:
            if not isinstance(freq_wts, torch.Tensor):
                freq_wts = torch.tensor(freq_wts)
        self.freq_wts = freq_wts

        if pdf is not None:
            if not isinstance(pdf, torch.Tensor):
                pdf = torch.tensor(pdf)
            if len(pdf.shape) > 1:
                pdf = pdf.flatten()
            self.pdf = pdf
            self.set_up_calcs()

    # compute the ergodic metric
    def forward(self, controls, traj, print_flag=False):

        # confirm we can do this
        if not self.init_flag:
            print("[ERROR] Ergodic loss module not initialized properly, need to provide map before attempting to calculate. Returning with None.")
            return None

        # ergodic metric
        erg_metric = torch.sum(self.lambdak * torch.square(self.phik - self.ck(traj)))

        # controls regularizer
        control_metric = torch.mean(controls**2, dim=0)

        # boundary condition counts number of points out of bounds
        zt = torch.tensor([0])
        bound_metric = torch.sum(torch.ceil(torch.maximum(zt, traj[:,:2]-1)) + torch.ceil(torch.maximum(zt, -traj[:,:2])))

        # print info if desired
        if print_flag:
            print("LOSS: erg = {:4.4f}, control = ({:4.4f}, {:4.4f}), boundary = {:4.4f}".format(erg_metric, control_metric[0], control_metric[1], bound_metric))

        return (self.args.erg_wt * erg_metric) + (self.args.transl_vel_wt * control_metric[0]) + (self.args.ang_vel_wt * control_metric[1]) + (self.args.bound_wt * bound_metric)

    # Update the stored map
    def update_pdf(self, pdf, fourier_freqs=None, freq_wts=None):
        if len(pdf.shape) > 1:
            pdf = pdf.flatten()
        self.pdf = pdf
        if fourier_freqs is not None: self.fourier_freqs = fourier_freqs
        if freq_wts is not None: self.freq_wts = freq_wts
        self.set_up_calcs()

    # set up calculations related to pdf
    # TODO: adjust so we can also use a 3d state space
    # for this will need 3d frequencies, X, and Y, and d = 4 instead of 3 for lambda exponent
    def set_up_calcs(self):

        # define frequencies to use if none have been provided
        if self.fourier_freqs is None:
            k1, k2 = torch.meshgrid(*[torch.arange(0, self.args.num_freqs, dtype=torch.float64)]*2)
            k = torch.stack([k1.ravel(), k2.ravel()], dim=1)
            self.k = torch.pi * k
        else:
            self.k = self.fourier_freqs

        # weights to use
        if self.freq_wts is None:
            self.lambdak = (1. + torch.linalg.norm(self.k / torch.pi, dim=1)**2)**(-3./2.)
        else:
            self.lambdak = self.freq_wts

        # state variables corresponding to pdf grid
        grid = torch.linspace(0, 1, steps = self.args.num_pixels, dtype=torch.float64)
        Y, X = torch.meshgrid(grid, grid) # torch creates these opposite to how numpy does it
        self.s = torch.stack([X.ravel(), Y.ravel()], dim=1)

        # vmap function for computing fourier coefficients efficiently (hopefully)
        self.fk = lambda x, k : torch.prod(torch.cos(x*k))
        self.fk_vmap = lambda x, k : torch.vmap(self.fk, in_dims=(0,None))(x, k)

        # compute hk normalizing factor
        _hk = (2.*self.k + torch.sin(2.*self.k)) / (4.*self.k)
        _hk[torch.isnan(_hk)] = 1.
        self.hk = torch.sqrt(torch.prod(_hk, dim=1))

        # compute map stats
        fk_map = torch.vmap(self.fk_vmap, in_dims=(None, 0))(self.s, self.k)
        phik = fk_map @ self.pdf
        phik = phik / phik[0]
        self.phik = phik / self.hk

        # map stats for reconstruction
        self.map_recon = self.phik @ fk_map

        # set flag to true so we know we can compute the metric
        self.init_flag = True

    # compute the trajectory statistics for a trajectory
    def ck(self, traj):
        fk_traj = torch.vmap(partial(self.fk_vmap, traj[:,:2]))(self.k)
        ck = torch.mean(fk_traj, dim=1)
        ck = ck / self.hk
        return ck

    # compute trajectory reconstruction of map
    def traj_recon(self, traj):
        return self.ck(traj) @ torch.vmap(self.fk_vmap, in_dims=(None, 0))(self.s, self.k)
