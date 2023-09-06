import hydra
from omegaconf.dictconfig import DictConfig
import optuna
from typing import Dict, Tuple, List, Optional
from optuna.study import Study
import jax.numpy as jnp
import numpy as np
import chex
from plt_returns.evaluation.evaluate_policy import (
    create_evaluation_output_summary,
    create_evaluation_output_df,
)
from plt_returns.utils.rollout import RolloutWrapper
from plt_returns.replenishment_policies.heuristic_policy import HeuristicPolicy
import logging
from math import inf
import jax
import pandas as pd

# Enable logging
log = logging.getLogger(__name__)


def param_search_bounds_from_config(
    cfg: DictConfig, policy: HeuristicPolicy
) -> Dict[str, int]:
    """Create a dict of search bounds for each parameter from the config file"""
    # Specify search bounds for each parameter
    if cfg.param_search.search_bounds.all_params is None:
        try:
            search_bounds = {
                p: {
                    "low": cfg.param_search.search_bounds[p]["low"],
                    "high": cfg.param_search.search_bounds[p]["high"],
                }
                for p in policy.param_names.flat
            }
        except:
            raise ValueError(
                "Ranges for each parameter must be specified if not using same range for all parameters"
            )
    # Otherwise, use the same range for all parameters
    else:
        search_bounds = {
            p: {
                "low": cfg.param_search.search_bounds.all_params.low,
                "high": cfg.param_search.search_bounds.all_params.high,
            }
            for p in policy.param_names.flat
        }
    return search_bounds


def grid_search_space_from_config(
    search_bounds: Dict[str, int], policy: HeuristicPolicy
) -> Dict[str, List[int]]:
    """Create a grid search space from the search bounds"""
    search_space = {
        p: list(
            range(
                search_bounds[p]["low"],
                search_bounds[p]["high"] + 1,
            )
        )
        for p in policy.param_names.flat
    }
    return search_space


def run_simopt(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    initial_params: Optional[Dict[str, int]] = None,
    sensitivity: Optional[float] = None,
    specificity: Optional[float] = None,
) -> Study:
    """Run a simulation optimization study to find the best parameters for a policy"""
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


def evaluation_run_with_simulated_model(
    sensitivity: float,
    specificity: float,
    cfg: DictConfig,
    policy: HeuristicPolicy,
    policy_params: chex.Array,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a policy and return a dataframe with results for all rollouts and a summary row"""
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
    evaluation_output = create_evaluation_output_df(cfg, rollout_results)
    evaluation_output_summary_row = create_evaluation_output_summary(
        cfg, rollout_results
    )
    evaluation_output_summary_row["sensitivity"] = sensitivity
    evaluation_output_summary_row["specificity"] = specificity
    evaluation_output_summary_row["1-specificity"] = round(
        1 - specificity, 2
    )  # Assume we're never interested in more than 2dp
    return evaluation_output, evaluation_output_summary_row


def process_params_for_log(
    policy: HeuristicPolicy, params: chex.Array
) -> Dict[str, int]:
    """Process policy parameters for logging"""
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


def process_params_for_df(
    policy: HeuristicPolicy, params: chex.Array
) -> Dict[str, int]:
    """Process policy parameters for adding to a dataframe"""
    return {
        str(param_name): int(param_value)
        for param_name, param_value in zip(policy.param_names.flat, params.flat)
    }


# Grid sampler is not straightforwardly compatible with the ask/tell
# interface so we need to treat it a bit differently to avoid
# to avoid duplication and handle RuntimeError
# https://github.com/optuna/optuna/issues/4121
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
        while len(policy_params) < num_parallel_trials:
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
        while len(policy_params) < cfg.param_search.max_parallel_trials:
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
