import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import logging
from datetime import datetime
import pandas as pd
import optuna
from typing import Dict, Tuple, List, Optional
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

# TODO: Flip sens and spec in env and here, so we're trying to predict if all units
# transfused
# TODO: Note on rounding sens and spec
# TODO: Recall that weekend standard delivery arrives at about the time of the returns, so
# we might want to do something a little different there; or just acknowledge as a limitation/simplification.


def simopt_grid_sampler(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    rollout_wrapper: RolloutWrapper,
    rng_eval: chex.PRNGKey,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using Optuna's GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, policy)
    search_space = grid_search_space_from_config(search_bounds, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, search_space=search_space, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    i = 1
    while (
        len(sampler._get_unvisited_grid_ids(study)) > 0
        and i <= cfg.param_search.max_iterations
    ):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        num_parallel_trials = min(
            len(sampler._get_unvisited_grid_ids(study)),
            cfg.param_search.max_parallel_trials,
        )
        for j in range(num_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            policy_params.append(
                np.array(
                    [
                        trial.suggest_int(
                            f"{p}",
                            search_bounds[p]["low"],
                            search_bounds[p]["high"],
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts")
        rollout_results = rollout_wrapper.population_rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = rollout_results["cum_return"].mean(axis=(-2, -1))

        for idx in range(num_parallel_trials):
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_mean",
                rollout_results["reward"][idx].mean(axis=(-2, -1)),
            )
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_std",
                rollout_results["reward"][idx].mean(axis=-1).std(),
            )
            trials[idx].set_user_attr(
                "cumulative_discounted_return_std",
                rollout_results["cum_return"][idx].std(),
            )
            try:
                study.tell(trials[idx], objectives[idx])
            except RuntimeError:
                break
        # Override rollout_results; helps to avoid GPU OOM error on larger problems
        rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        i += 1
    return study


def simopt_other_sampler(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    rollout_wrapper: RolloutWrapper,
    rng_eval: chex.PRNGKey,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using an Optuna sampler other than GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    # Counter for early stopping
    es_counter = 0

    for i in range(1, cfg.param_search.max_iterations + 1):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        for j in range(cfg.param_search.max_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            policy_params.append(
                np.array(
                    [
                        trial.suggest_int(
                            f"{p}",
                            search_bounds[p]["low"],
                            search_bounds[p]["high"],
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts")
        rollout_results = rollout_wrapper.population_rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = rollout_results["cum_return"].mean(axis=(-2, -1))

        for idx in range(cfg.param_search.max_parallel_trials):
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_mean",
                rollout_results["reward"][idx].mean(axis=(-2, -1)),
            )
            trials[idx].set_user_attr(
                "daily_undiscounted_reward_std",
                rollout_results["reward"][idx].mean(axis=-1).std(),
            )
            trials[idx].set_user_attr(
                "cumulative_discounted_return_std",
                rollout_results["cum_return"][idx].std(),
            )
            study.tell(trials[idx], objectives[idx])

        # Override rollout_results; helps to avoid GPU OOM error on larger problems
        rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        # Perform early stopping starting on the second round
        if i > 1:
            if study.best_params == best_params_last_round:
                es_counter += 1
            else:
                es_counter = 0
        if es_counter >= cfg.param_search.early_stopping_rounds:
            log.info(
                f"No change in best parameters for {cfg.param_search.early_stopping_rounds} rounds. Stopping search."
            )
            break
        best_params_last_round = study.best_params
    return study


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
    return evaluation_output


def run_simopt(
    cfg,
    policy,
    initial_params: Optional[Dict[str, int]] = None,
    sensitivity: Optional[float] = None,
    specificity: Optional[float] = None,
):
    if sensitivity is not None:
        cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity = float(
            sensitivity
        )
    if specificity is not None:
        cfg.rollout_wrapper.env_params.return_prediction_model_specificity = float(
            specificity
        )

    log.info(
        f"Starting simulation optimization for replenishment policy, model sensitivity: {cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity}, model specificity: {cfg.rollout_wrapper.env_params.return_prediction_model_specificity}"
    )

    rollout_wrapper = hydra.utils.instantiate(
        cfg.rollout_wrapper, model_forward=policy.forward
    )
    rng_eval = jax.random.split(
        jax.random.PRNGKey(cfg.param_search.seed), cfg.param_search.num_rollouts
    )

    if cfg.param_search.sampler._target_ == "optuna.samplers.GridSampler":
        study = simopt_grid_sampler(
            cfg, policy, rollout_wrapper, rng_eval, initial_params
        )
    else:
        study = simopt_other_sampler(
            cfg, policy, rollout_wrapper, rng_eval, initial_params
        )

    log.info(
        f"Simulation optimization for replenishment policy complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
    )
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


def process_params_for_df(policy, params):
    return {
        str(param_name): int(param_value)
        for param_name, param_value in zip(policy.param_names.flat, params.flat)
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run simulation optimization using Optuna to find the best parameters for a policy,
    and evaluate the policy using the best parameters on a separate set of rollouts"""
    start_time = datetime.now()

    output_info = {}
    policy = hydra.utils.instantiate(cfg.policy)

    oufo_study = run_simopt(
        cfg, policy, sensitivity=0.0, specificity=1.0
    )  # Equivalent to OUFO

    # Extract best params and add to output_info
    # We assume here that all parameters are integers
    # which they should be for the kinds of heuristic
    # policies we're using

    oufo_params_dict = oufo_study.best_params
    oufo_params = np.array([v for v in oufo_study.best_params.values()]).reshape(
        policy.params_shape
    )

    output_info["oufo_params"] = process_params_for_log(policy, oufo_params)

    # For each assume level of sensitivity and specificity, run evaluation

    sensitivities = np.arange(0, 1.1, 0.1).round(2)
    specificities = np.arange(0, 1.1, 0.1).round(2)

    pred_return_res = pd.DataFrame()
    for sens in sensitivities:
        for spec in specificities:
            if sens == 0.0 and spec == 1.0:  # Equivalent to OUFO
                policy_params = oufo_params
            else:
                rep_study = run_simopt(cfg, policy, oufo_params_dict, sens, spec)
                trials_df = rep_study.trials_dataframe()
                trials_df.to_csv(f"sens_{sens}_spec_{spec}_trials.csv")
                policy_params = np.array(
                    [v for v in rep_study.best_params.values()]
                ).reshape(policy.params_shape)
            labelled_params_for_df = process_params_for_df(policy, policy_params)
            policy_params = jnp.array(policy_params)

            evaluation_output = evaluation_run_with_simulated_model(
                sens, spec, cfg, policy, policy_params
            )
            new_row = pd.DataFrame(
                labelled_params_for_df | evaluation_output, index=[0]
            )
            # Add to res dataframe
            pred_return_res = pd.concat([pred_return_res, new_row], ignore_index=True)

    pred_return_res["1-specificity"] = (1 - pred_return_res["specificity"]).round(1)

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
    pred_return_res["total_wastage_%_mean"] = (
        pred_return_res["slippage_%_mean"] + pred_return_res["expiries_%_mean"]
    )
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
