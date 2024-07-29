#!/usr/bin/env python
# example use of ergodic trajectory planner

import torch
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

from ergodic_search import erg_planner
from ergodic_search.dynamics import DiffDrive

# map settings
LOCS = [[0.2, 0.8], [0.8, 0.2]]
STDS = [0.01, 0.01]
WTS = [1, 1]

# create example map with a few gaussian densities in it
def create_map(dim):
    # set up map and underlying grid
    map = np.zeros((dim, dim))
    res = 1 / dim
    xgrid, ygrid = np.mgrid[0:1:res, 0:1:res]
    map_grid = np.dstack((xgrid, ygrid))

    # add a few gaussians
    for i in range(len(LOCS)):
        dist = norm(LOCS[i], STDS[i])
        vals = dist.pdf(map_grid)
        map += WTS[i] * vals

    # normalize the map
    map = (map - np.min(map)) / (np.max(map) - np.min(map))

    return map


# call main function
if __name__ == "__main__":

    # parse arguments
    args = erg_planner.ErgArgs()

    # set a more interesting starting position and initial controls
    args.start_pose = [0.2, 0.2, 0]
    args.end_pose = [0.8, 0.8, 0]
    args.num_freqs = 10

    # create dynamics module
    diff_drive = DiffDrive(args.start_pose, args.traj_steps)
    
    # create initial trajectory
    ref_tr_init = np.zeros((args.traj_steps, 3))
    x_dist = args.end_pose[0] - args.start_pose[0]
    y_dist = args.end_pose[1] - args.start_pose[1]
    for i in range(args.traj_steps):
        ref_tr_init[i,0] = (args.start_pose[0] + (x_dist * (i+1))/(args.traj_steps-1))
        ref_tr_init[i,1] = (args.start_pose[1] + (y_dist * (i+1))/(args.traj_steps-1))
    
    # have first and last poses be the only ones with changes in angle
    angle = np.pi / 4
    ref_tr_init[:-1, 2] = angle
    ref_tr_init[-1, 2] = args.end_pose[2]
    # print(ref_tr_init)

    with torch.no_grad():
        # this is to trick pytorch into ignoring the computation here
        # otherwise it'll complain about controls not being a leaf tensor
        init_controls = diff_drive.inverse(ref_tr_init)
    # print(init_controls)

    # create example map
    map = create_map(args.num_pixels)

    # initialize planner
    init_controls.requires_grad = True
    planner = erg_planner.ErgPlanner(args, map, init_controls=init_controls, dyn_model=diff_drive)

    # generate a trajectory
    traj = planner.compute_traj(debug=args.debug)

    # visualize map and trajectory
    planner.visualize()
