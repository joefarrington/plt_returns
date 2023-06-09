import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import logging
from datetime import datetime
import pandas as pd
import optuna
from typing import Dict, Tuple, List
from optuna.study import Study
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
import chex
import matplotlib.pyplot as plt
from plt_returns.evaluation.evaluate_policy import create_evaluation_output_summary
from plt_returns.utils.rollout import RolloutWrapper
from plt_returns.utils.plotting import plot_heatmap
from viso_jax.policies.heuristic_policy import HeuristicPolicy
from viso_jax.utils.yaml import to_yaml, from_yaml
from viso_jax.simopt.run_optuna_simopt import (
    param_search_bounds_from_config,
    grid_search_space_from_config,
    simopt_grid_sampler,
    simopt_other_sampler,
)
from plt_returns.simulation.run_replenishment_simopt import (
    evaluation_run_with_simulated_model,
    run_simopt,
    process_params_for_log,
)


# Enable logging
log = logging.getLogger(__name__)

# TODO: Flip sens and spec in env and here, so we're trying to predict if all units transfused


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """For different combinations of input params, run simulation optimization using Optuna
    to find the best parameters for a policy and evaluate the policy using the best parameters
    on a separate set of rollouts"""
    output_info = {}
    res = pd.DataFrame()

    for combination in cfg.sweep_over:
        start_time = datetime.now()
        cfg = OmegaConf.merge(cfg, combination.updates)

        # Scale the demand based on return rate, which may be changed in combination
        cfg.rollout_wrapper.env_params.poisson_mean_demand_pre_return = [
            float(x)
            for x in np.array(cfg.base_demand.poisson_mean_demand_pre_return)
            / (1 - cfg.rollout_wrapper.env_params.return_probability)
        ]
        cfg.rollout_wrapper.env_params.poisson_mean_demand_post_return = [
            float(x)
            for x in np.array(cfg.base_demand.poisson_mean_demand_post_return)
            / (1 - cfg.rollout_wrapper.env_params.return_probability)
        ]

        policy = hydra.utils.instantiate(cfg.policy)
        # Set sens to 0, equaivalent to OUFO
        cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity = 0.0
        print(cfg)
        study = run_simopt(cfg, policy)
        simopt_complete_time = datetime.now()
        simopt_run_time = simopt_complete_time - start_time
        log.info(
            f"Simulation optimization for case {combination.name} complete. Duration: {(simopt_run_time).total_seconds():.2f}s.  Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        output_info[combination.name] = {}
        output_info[combination.name]["running_times"] = {}
        output_info[combination.name]["running_times"][
            "simopt_run_time"
        ] = simopt_run_time.total_seconds()

        best_params = np.array([v for v in study.best_params.values()]).reshape(
            policy.params_shape
        )
        output_info[combination.name]["policy_params"] = process_params_for_log(
            policy, best_params
        )

        policy_params = jnp.array(best_params)

        # Evaluate best replenishment policy under OUFO allocation
        sens = 0.0
        spec = 1.0
        results_oufo = evaluation_run_with_simulated_model(
            sens, spec, cfg, policy, policy_params
        )
        results_oufo["name"] = combination.name
        results_oufo["allocation"] = "OUFO"

        # Evaluate best replenishment policy under perfect predictive model
        sens = 1.0
        spec = 1.0
        results_perfect_model = evaluation_run_with_simulated_model(
            sens, spec, cfg, policy, policy_params
        )

        results_perfect_model["name"] = combination.name
        results_perfect_model["allocation"] = "Perfect predictive model"

        res = pd.concat([res, results_oufo, results_perfect_model], ignore_index=True)

    # Any final logging that combines results across different combinations of input params
    res = res.set_index(["name", "allocation"], drop=True)
    res.to_csv("results.csv")
    log.info("Final results saved")

    to_yaml(output_info, "output_info.yaml")
    log.info("Evaluation output saved")


if __name__ == "__main__":
    main()
