###  Next-Activity Prediction for Non-stationary Processes with Unseen Data Variability

This repository contains the code and simulation results presented in the
experimental section of the paper. All the commands described bellow require
that we are in the root directory of the project.

Python >= 3.8 and conda are required. 

```
# Extract the data sets and simulation results
sh extract_tar.sh

# To view the simulation results presented in the paper run
python -m kronos.evaluate_runs

# The simulations can be run 
python -m kronos.simulation
```

