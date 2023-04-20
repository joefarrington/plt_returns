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

# Enable logging
log = logging.getLogger(__name__)

# TODO: Flip sens and spec in env and here, so we're trying to predict if all units transfused
# TODO: Note on rounding sens and spec
# TODO: Recall that weekend standard delivery arrives at about the time of the returns, so
# we might want to do something a little different there; or just acknowledge as a limitation/simplification.


def evaluation_run_with_simulated_model(
    sensitivity, specificity, cfg, policy, policy_params
):
    cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity = float(
        sensitivity
    )
    cfg.rollout_wrapper.env_params.return_prediction_model_specificity = float(
        specificity
    )
    rollout_wrapper = hydra.utils.instantiate(
        cfg.rollout_wrapper, model_forward=policy.forward, return_info=True
    )
    rng_eval = jax.random.split(
        jax.random.PRNGKey(cfg.evaluation.seed), cfg.evaluation.num_rollouts
    )
    rollout_results = rollout_wrapper.batch_rollout(rng_eval, policy_params)
    evaluation_output = create_evaluation_output_summary(cfg, rollout_results)
    evaluation_output["sensitivity"] = sensitivity
    evaluation_output["specificity"] = specificity
    evaluation_output["1-specificity"] = round(
        1 - specificity, 2
    )  # Assume we're never interested in more than 2dp
    return pd.DataFrame(evaluation_output, index=[0])


def run_simopt(cfg, policy):
    rollout_wrapper = hydra.utils.instantiate(
        cfg.rollout_wrapper, model_forward=policy.forward
    )
    rng_eval = jax.random.split(
        jax.random.PRNGKey(cfg.param_search.seed), cfg.param_search.num_rollouts
    )

    if cfg.param_search.sampler._target_ == "optuna.samplers.GridSampler":
        study = simopt_grid_sampler(cfg, policy, rollout_wrapper, rng_eval)
    else:
        study = simopt_other_sampler(cfg, policy, rollout_wrapper, rng_eval)
    return study


def process_params_for_log(policy, params):
    # If no row labels, we don't want a multi-level dict
    # so handle separately
    if policy.param_row_names == []:
        processed_params = {
            str(param_name): int(param_value)
            for param_name, param_value in zip(policy.param_names.flat, params.flat)
        }
    # If there are row labels, easiest to convert to a dataframe and then into nested dict
    else:
        processed_params = pd.DataFrame(
            params,
            index=policy.param_row_names,
            columns=policy.param_col_names,
        ).to_dict()
    return processed_params


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run simulation optimization using Optuna to find the best parameters for a policy,
    and evaluate the policy using the best parameters on a separate set of rollouts"""
    start_time = datetime.now()

    output_info = {}
    policy = hydra.utils.instantiate(cfg.policy)
    # Set sens to 0, equaivalent to OUFO
    cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity = 0.0

    study = run_simopt(cfg, policy)

    best_trial_idx = study.best_trial.number
    trials_df = study.trials_dataframe()
    trials_df.to_csv("trials.csv")

    best_trial_df = trials_df.loc[[best_trial_idx]]
    best_trial_df.to_csv("best_trial.csv")

    simopt_complete_time = datetime.now()
    simopt_run_time = simopt_complete_time - start_time
    log.info(
        f"Simulation optimization complete. Duration: {(simopt_run_time).total_seconds():.2f}s.  Best params: {study.best_params}, mean return: {study.best_value:.4f}"
    )
    output_info["running_times"] = {}
    output_info["running_times"]["simopt_run_time"] = simopt_run_time.total_seconds()

    # Extract best params and add to output_info
    # We assume here that all parameters are integers
    # which they should be for the kinds of heuristic
    # policies we're using

    log.info(
        "Running evaluation rollouts for the best params, for different assumed levels of model sensitivity and specificity"
    )

    best_params = np.array([v for v in study.best_params.values()]).reshape(
        policy.params_shape
    )

    output_info["policy_params"] = process_params_for_log(policy, best_params)

    # For each assume level of sensitivity and specificity, run evaluation

    policy_params = jnp.array(best_params)
    sensitivities = np.arange(0, 1.1, 0.1).round(2)
    specificities = np.arange(0, 1.1, 0.1).round(2)

    pred_return_res = pd.DataFrame()
    for sens in sensitivities:
        for spec in specificities:
            results = evaluation_run_with_simulated_model(
                sens, spec, cfg, policy, policy_params
            )
            # Add to res dataframe
            pred_return_res = pd.concat([pred_return_res, results], ignore_index=True)

    pred_return_res["1-specificity"] = (1 - pred_return_res["specificity"]).round(1)

    eval_complete_time = datetime.now()
    eval_run_time = eval_complete_time - simopt_complete_time
    log.info(f"Evaluation duration: {(eval_run_time).total_seconds():.2f}s")
    output_info["running_times"]["eval_run_time"] = eval_run_time.total_seconds()

    # Expiry heatmap
    # TODO: Could do one that combines expiries and slippage as total wastage
    expiry_pc_heatmap = plot_heatmap(
        pred_return_res,
        "1-specificity",
        "sensitivity",
        "expiries_%_mean",
        "Expiry % with an imagined model of different levels of quality",
        heatmap_kwargs={"cmap": "RdYlGn_r"},
    )
    plt.savefig("expiry_pc_heatmap.png")

    # Total wastage heatmap
    pred_return_res["total_wastage_%_mean"] = pred_return_res['slippage_%_mean'] + pred_return_res['expiries_%_mean']
    wastage_pc_heatmap = plot_heatmap(
        pred_return_res,
        "1-specificity",
        "sensitivity",
        "total_wastage_%_mean",
        "Wastage (expiries plus slippage) % with an imagined model of different levels of quality",
        heatmap_kwargs={"cmap": "RdYlGn_r"},
    )
    plt.savefig("wastage_pc_heatmap.png")

    # Service level heatmap
    service_level_pc_heatmap = plot_heatmap(
        pred_return_res,
        "1-specificity",
        "sensitivity",
        "service_level_%_mean",
        "Service level % with an imagined model of different levels of quality",
        heatmap_kwargs={"cmap": "RdYlGn"},
    )
    plt.savefig("service_level_pc_heatmap.png")

    pred_return_res = pred_return_res.set_index(
        ["sensitivity", "specificity"], drop=True
    )
    pred_return_res.to_csv("performance_by_sens_spec.csv")

    # output_info["evaluation_output"] = evaluation_output
    to_yaml(output_info, "output_info.yaml")
    log.info("Evaluation output saved")


if __name__ == "__main__":
    main()
