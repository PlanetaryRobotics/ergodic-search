#!/usr/bin/env python
# example use of ergodic trajectory planner

import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt

from ergodic_search import erg_planner

# map settings
LOCS = [[0.25, 0.25], [0.85, 0.5], [0.6, 0.85]]
STDS = [0.1, 0.02, 0.01]
WTS = [8, 1, 0.9]

# create example map with a few gaussian densities in it
def create_map():
    # set up map and underlying grid
    map = np.zeros((500, 500))
    res = 1 / 500
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

# plan ergodic trajectory over map
def plan(map, args):
    planner = erg_planner.ErgPlanner(args, map)
    return planner.compute_traj()


# call main function
if __name__ == "__main__":

    # parse arguments
    args = erg_planner.ErgArgs()

    # create example map
    map = create_map(args)

    # call main function
    traj = plan(map, args)

    # visualize map and trajectory
    plt.imshow(map)
    # plt.scatter(traj[:,0:2])
    # plt.plot(traj[:,0:2])
    plt.show()
