#!/usr/bin/env python
# class for computing the ergodic metric over a trajectory 
# given a spatial distribution

import torch

from functools import partial


# Module for computing ergodic loss over a PDF
class ErgLoss(torch.nn.Module):

    def __init__(self, args, dyn_model, pdf=None, fourier_freqs=None, freq_wts=None):
        
        super(ErgLoss, self).__init__()
        
        self.args = args
        self.device = 'cuda' if self.args.gpu else 'cpu'
        self.init_flag = False
        self.has_pdf = False if pdf is None else True

        end_pose = torch.tensor(self.args.end_pose, device=self.device)

        if args.num_freqs == 0 and fourier_freqs is None:
            print("[ERROR] args.num_freqs needs to be positive or fourier_freqs must be provided. Returning with None.")
            return None

        if fourier_freqs is not None:
            if not isinstance(fourier_freqs, torch.Tensor):
                fourier_freqs = torch.tensor(fourier_freqs)
        
        if freq_wts is not None:
            if not isinstance(freq_wts, torch.Tensor):
                freq_wts = torch.tensor(freq_wts)

        self.register_buffer("end_pose", end_pose)
        self.register_buffer("fourier_freqs", fourier_freqs)
        self.register_buffer("freq_wts", freq_wts)

        if pdf is not None:
            if not isinstance(pdf, torch.Tensor):
                pdf = torch.tensor(pdf)
            if len(pdf.shape) > 1:
                pdf = pdf.flatten()
            self.register_buffer("pdf", pdf)
            self.set_up_calcs()            

        self.dyn_model = dyn_model


    # compute the ergodic metric
    def forward(self, print_flag=False):

        # confirm we can do this
        if not self.init_flag:
            print("[ERROR] Ergodic loss module not initialized properly, need to provide map before attempting to calculate. Returning with None.")
            return None

        # get the trajectory from the dynamics model
        traj = self.dyn_model.forward()
        
        # ergodic metric
        erg_metric = torch.sum(self.lambdak * torch.square(self.phik - self.ck(traj)))
        
        # controls regularizer
        control_metric = torch.mean(self.dyn_model.controls**2, dim=0)
        
        # boundary condition counts number of points out of bounds
        zt = torch.tensor([0], device=self.device)
        bound_metric = torch.sum(torch.maximum(zt, traj[:,:2]-1) + torch.maximum(zt, -traj[:,:2]))
        
        # end point loss
        end_metric = torch.sum((self.end_pose - traj[-1,:])**2)
        
        # print info if desired
        if print_flag:
            print("LOSS: erg = {:4.4f}, control = ({:4.4f}, {:4.4f}), boundary = {:4.4f}, end = {:4.4f}".format(erg_metric, control_metric[0], control_metric[1], bound_metric, end_metric))

        loss = (self.args.erg_wt * erg_metric) \
            + (self.args.transl_vel_wt * control_metric[0]) \
            + (self.args.ang_vel_wt * control_metric[1]) \
            + (self.args.bound_wt * bound_metric) \
            + (self.args.end_pose_wt * end_metric)
        return loss


    # just compute the ergodic metric
    def calc_erg_metric(self, freqs=None, weights=None):
        with torch.no_grad():
            traj = self.dyn_model.forward().cpu()
            
            # get components using new frequencies if provided
            if freqs is not None:
                # define frequencies to use if none have been provided
                k = self.get_k(freqs).cpu()

                # weights to use
                lambdak = self.get_lambdak(k, weights).cpu()

                fk = lambda x, k : torch.prod(torch.cos(x*k))
                fk_vmap = lambda x, k : torch.vmap(fk, in_dims=(0,None))(x, k)

                _hk = (2.*k + torch.sin(2.*k)) / (4.*k)
                _hk[torch.isnan(_hk)] = 1.
                hk = torch.sqrt(torch.prod(_hk, dim=1))

                s = self.s.cpu()
                fk_map = torch.vmap(fk_vmap, in_dims=(None, 0))(s, k)
                phik = fk_map @ self.pdf.cpu()
                phik = phik / phik[0]
                phik = phik / hk

                fk_traj = torch.vmap(partial(fk_vmap, traj[:,:2]))(k)
                ck = torch.mean(fk_traj, dim=1)
                ck = ck / hk

            else:
                lambdak = self.lambdak
                phik = self.phik
                ck = self.ck(traj)

            erg = torch.sum(lambdak * torch.square(phik - ck))
            return erg.detach()


    # Update the stored map
    def update_pdf(self, pdf, fourier_freqs=None, freq_wts=None):

        if not isinstance(pdf, torch.Tensor):
            pdf = torch.tensor(pdf)

        if len(pdf.shape) > 1:
            pdf = pdf.flatten()
        
        if self.has_pdf == False:

            if fourier_freqs is not None:
                if not isinstance(fourier_freqs, torch.Tensor):
                    fourier_freqs = torch.tensor(fourier_freqs)
                self.register_buffer("fourier_freqs", fourier_freqs)

            if freq_wts is not None:
                if not isinstance(freq_wts, torch.Tensor):
                    freq_wts = torch.tensor(freq_wts)
                self.register_buffer("freq_wts", freq_wts)

            self.register_buffer("pdf", pdf)
            self.has_pdf = True
        
        else:

            if fourier_freqs is not None:
                self.fourier_freqs = fourier_freqs
            
            if freq_wts is not None:
                self.freq_wts = freq_wts
            
            self.pdf = pdf

        self.set_up_calcs()
        self.pdf = self.pdf.to(self.device)


    # set up calculations related to pdf
    # TODO: adjust so we can also use a 3d state space
    # for this will need 3d frequencies, X, and Y, and d = 4 instead of 3 for lambda exponent
    def set_up_calcs(self):

        # define frequencies to use if none have been provided
        k = self.get_k(self.fourier_freqs)

        # weights to use
        lambdak = self.get_lambdak(k, self.freq_wts)

        # state variables corresponding to pdf grid
        X, Y = torch.meshgrid(*[torch.linspace(0, 1, self.args.num_pixels, dtype=torch.float64)]*2, indexing='xy')
        s = torch.stack([X.ravel(), Y.ravel()], dim=1)

        # vmap function for computing fourier coefficients efficiently (hopefully)
        self.fk = lambda x, k : torch.prod(torch.cos(x*k))
        self.fk_vmap = lambda x, k : torch.vmap(self.fk, in_dims=(0,None))(x, k)

        # compute hk normalizing factor
        _hk = (2.*k + torch.sin(2.*k)) / (4.*k)
        _hk[torch.isnan(_hk)] = 1.
        hk = torch.sqrt(torch.prod(_hk, dim=1))

        # compute map stats
        fk_map = torch.vmap(self.fk_vmap, in_dims=(None, 0))(s, k)
        phik = fk_map @ self.pdf
        phik = phik / phik[0]
        phik = phik / hk

        # map stats for reconstruction
        self.map_recon = phik @ fk_map

        if self.init_flag == False:

            self.register_buffer("k", k)
            self.register_buffer("lambdak", lambdak)
            self.register_buffer("s", s)
            self.register_buffer("hk", hk)
            self.register_buffer("phik", phik)

            # set flag to true so we know we can compute the metric
            self.init_flag = True

        else:

            # update previously stored values
            self.k = k.to(self.device)
            self.lambdak = lambdak.to(self.device)
            self.s = s.to(self.device)
            self.hk = hk.to(self.device)
            self.phik = phik.to(self.device)


    # compute the trajectory statistics for a trajectory
    def ck(self, traj):
        fk_traj = torch.vmap(partial(self.fk_vmap, traj[:,:2]))(self.k)
        ck = torch.mean(fk_traj, dim=1)
        ck = ck / self.hk
        return ck


    # compute trajectory reconstruction of map
    def traj_recon(self, traj):
        return self.ck(traj) @ torch.vmap(self.fk_vmap, in_dims=(None, 0))(self.s, self.k)


    # compute k based on frequencies
    def get_k(self, freqs=None):
        if freqs is None:
            k1, k2 = torch.meshgrid(*[torch.arange(0, self.args.num_freqs, dtype=torch.float64)]*2, indexing='ij')
            k = torch.stack([k1.ravel(), k2.ravel()], dim=1)
            k = torch.pi * k
        else:
            if not isinstance(freqs, torch.Tensor):
                freqs = torch.tensor(freqs)
            k = freqs
        return k


    # compute lambda k based on frequency weights
    def get_lambdak(self, k, freq_wts=None):
        if freq_wts is None:
            # MAH NOTE: my reading of the literature suggests the coefficient below should be (-3/2) instead of (-4/2)
            # however MOES uses (-4/2) and that seems to produce better results, at least for this implementation
            lambdak = (1. + torch.linalg.norm(k / torch.pi, dim=1)**2)**(-4./2.)
        else:
            lambdak = freq_wts
        return lambdak

