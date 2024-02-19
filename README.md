# plt_returns

Investigating machine learning-guided issuing policies to reduce platelet wastage in a hospital blood bank when issued units may be returned and reissued.

## Introduction

This repository provides the code to support the paper <i>"Many happy returns: machine learning to support platelet issuing and waste reduction in a hospital blood bank"</i> by Farrington et al.

In this paper, we modeled the workflow of a hospital blood bank managing platelets, where not all the of the platelets issued to patients are ultimately transfused. Using a simulation-first approach, we evaluated the potential benefits of an machine learning model that could predict whether issed platelets would be transfused or not at different levels of predictive model performance. 

Based on the findings from the simulation-first experiments, we trained an ML model using real requests for platelets from our partner hospital. 

## Simulation-first experiments

The code for the simulation experiments is located in the directory ```plt_returns/simulation```. There are [hydra](https://hydra.cc/) configuration files in ```plt_returns/simulation/conf``` corresponding to each experiment in the paper. 

Additional scenario analysis was performed using the code in the directory ```plt_returns/simulation/sim_compare_input```. There are hydra configuration files in ```plt_returns/simulation/sim_compare_input/conf``` corresponding to each experiment in the paper. 

## Machine learning experiments

The code used for predictive model training and hyperparameter tuning is contained in the directory ```plt_returns/simulation/ml```. 
