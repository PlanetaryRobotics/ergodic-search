#!/usr/bin/env python
# example use of ergodic trajectory planner

import time
import torch
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from torchviz import make_dot

from ergodic_search import erg_planner
from ergodic_search.dynamics import DiffDrive

# map settings
LOCS = [[0.2, 0.8], [0.8, 0.2]]
STDS = [0.01, 0.01]
WTS = [1, 1]

# create example map with a few gaussian densities in it
def create_map(dim):
    # set up map and underlying grid
    map_ex = np.zeros((dim, dim))
    res = 1 / dim
    ygrid, xgrid = np.mgrid[0:1:res, 0:1:res]

    map_grid = np.dstack((xgrid, ygrid))

    # add a few gaussians
    for i in range(len(LOCS)):
        dist = norm(LOCS[i], STDS[i])
        vals = dist.pdf(map_grid)
        map_ex += WTS[i] * vals

    # normalize the map
    map_ex = (map_ex - np.min(map_ex)) / (np.max(map_ex) - np.min(map_ex))

    return map_ex


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

    # create example map
    map_ex = create_map(args.num_pixels)

    # initialize planner
    planner = erg_planner.ErgPlanner(args, map_ex, init_controls=init_controls, dyn_model=diff_drive)

    # print model graph
    # dot = make_dot(planner.loss(), params=dict(planner.loss.named_parameters()))
    # dot.render("model_visualization", format="pdf")

    # generate a trajectory
    start = time.time()
    controls, traj, erg = planner.compute_traj(debug=args.debug)
    end = time.time()
    print("Execution time in secs: ", (end-start))

    # visualize map and trajectory
    planner.visualize()
