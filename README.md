###  Next-Activity Prediction for Non-stationary Processes with Unseen Data Variability

This repository contains the code and simulation results presented in the
experiments section of the paper. The commands described bellow must be
run from the root directory of the project.

Python >= 3.8 and PyTorch >= 1.10 (via conda) are required.

```
# Install additional libraries 
pip install numpy toolz sklearn tensorboard pandas 

# Extract the data sets and simulation results
sh extract_tar.sh

# To view the simulation results presented in the paper run
python -m kronos.evaluate_runs

# The simulations can be run via 
python -m kronos.simulation
```

