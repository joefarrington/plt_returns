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


def run_simopt_and_evaluate(
    rep_policy, cfg, combination_name, alloc_name, sensitivity, specificity
):
    cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity = sensitivity
    cfg.rollout_wrapper.env_params.return_prediction_model_specificity = specificity
    print(cfg)
    study = run_simopt(cfg, rep_policy)
    params_output = {}
    best_params = np.array([v for v in study.best_params.values()]).reshape(
        rep_policy.params_shape
    )
    params_output[f"policy_params_alloc_{alloc_name}"] = process_params_for_log(
        rep_policy, best_params
    )

    policy_params = jnp.array(best_params)

    # Evaluate best replenishment policy under OUFO allocation
    results = evaluation_run_with_simulated_model(
        sensitivity, specificity, cfg, rep_policy, policy_params
    )
    results["name"] = combination_name
    results["allocation"] = alloc_name
    return params_output, results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """For different combinations of input params, and each specified allocation policy
    run simulation optimization using Optuna to find the best parameters for  the replenishment
    policy and evaluate the policy using the best parameters on a separate set of rollouts
    """
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

        output_info[combination.name] = {}
        results_list = []
        rep_policy = hydra.utils.instantiate(cfg.policy)

        for alloc_policy_name, alloc_policy_params in OmegaConf.to_container(
            cfg.alloc_policies
        ).items():
            sensitivity = alloc_policy_params["sensitivity"]
            specificity = alloc_policy_params["specificity"]

            # Run simulation optimization and evaluate
            params_output, results = run_simopt_and_evaluate(
                rep_policy,
                cfg,
                combination.name,
                alloc_policy_name,
                sensitivity,
                specificity,
            )
            output_info[combination.name][alloc_policy_name] = params_output
            results_list.append(results)

        res = pd.concat([res, *results_list], ignore_index=True)

    # Any final logging that combines results across different combinations of input params
    res = res.set_index(["name", "allocation"], drop=True)
    res.to_csv("results.csv")
    log.info("Final results saved")

    to_yaml(output_info, "output_info.yaml")
    log.info("Evaluation output saved")


if __name__ == "__main__":
    main()
