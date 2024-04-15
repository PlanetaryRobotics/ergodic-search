# Ergodic Search

This repository implements basic ergodic search as outlined in [PAPER] and modified from [MO-ES].

[ADD DETAILS ON WHAT IT DOES HERE]

The package allows for external provision of dynamics models for use in the 
optimization step; more information is provided below on how to include a 
custom dynamics model.


## Dependencies

This implementation is built in Python and relies on PyTorch for optimization. The versions listed
below were the latest versions on which the code has been tested. The code is capable of running on 
a CPU or GPU.
- Python >= 3.8.10
- PyTorch >= 2.0.1
- [GPU ONLY] CUDA >= 11.8

## How to Install

Once the dependencies have been installed, clone this repository and navigate into it:
```
git clone git@github.com:PlanetaryRobotics/ergodic-search
cd ergodic-search
```

Then the package can be locally installed with ```pip install -e .```

## How to Use

This section provides more details on how this repository can be used to perform ergodic search on 
custom maps with various parameter settings.

An example script is included in ```example.py``` for reference. This script enables the user to 
adjust parameters but uses the same underlying map. Reviewing the script can help see how the ```ErgPlanner```
class can be used in practice.

### Parameters

The table below lists the hyperparameters available for tuning:



### Using Custom Maps

Note that this implementation assumes the map 


### Incorporating Dynamics Models

The dynamics of the robot are implemented via a PyTorch module, which requires an initialization and implementation
of the forward function. This function should compute a trajectory based on the controls stored in this module, which
are the parameters being optimized by PyTorch.

The example relies on a simple differential drive dynamics model. This model can be replaced with any PyTorch module


## References


