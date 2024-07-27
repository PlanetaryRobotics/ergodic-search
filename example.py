#!/usr/bin/env python
# example use of ergodic trajectory planner

import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

from ergodic_search import erg_planner
from ergodic_search.dynamics import DiffDrive

# map settings
LOCS = [[0.25, 0.25], [0.85, 0.5], [0.6, 0.85]]
STDS = [0.1, 0.02, 0.01]
WTS = [8, 1, 0.9]

# create example map with a few gaussian densities in it
def create_map(dim):
    # set up map and underlying grid
    map = np.zeros((dim, dim))
    res = 1 / dim
    xgrid, ygrid = np.mgrid[0:1:res, 0:1:res]
    map_grid = np.dstack((xgrid, ygrid))

    # add a few gaussians
    for i in range(3):
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
    args.start_pose = [0.2, 0.4, 0]
    args.num_freqs = 10

    init_controls = np.zeros((args.traj_steps,2))
    dx = (0.8 - 0.2) / args.traj_steps
    init_controls[:,0] = dx
    print(init_controls)

    # create dynamics module so we can test it
    diff_drive = DiffDrive(args.start_pose, args.traj_steps)
    traj = diff_drive.forward(init_controls)
    print(traj)

    controls = diff_drive.inverse(traj)
    print(controls)

    # create example map
    map = create_map(args.num_pixels)

    # initialize planner
    planner = erg_planner.ErgPlanner(args, map, init_controls=init_controls, dyn_model=diff_drive)

    # generate a trajectory
    traj = planner.compute_traj(debug=args.debug)

    # visualize map and trajectory
    planner.visualize()
