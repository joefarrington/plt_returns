from omegaconf.dictconfig import DictConfig
import chex
import logging
import pandas as pd
from plt_returns.utils.kpis import get_kpi_function
from viso_jax.utils.yaml import to_yaml
from typing import Dict


def create_evaluation_output_summary(
    cfg: DictConfig, rollout_results: Dict[str, chex.Array]
) -> pd.DataFrame:
    """Create a summary of the evaluation output, including KPIs"""
    eval_output = {}
    eval_output["daily_undiscounted_reward_mean"] = float(
        rollout_results["reward"].mean()
    )  # Equivalent to calculation mean for each rollout, then mean of those
    eval_output["daily_undiscounted_reward_std"] = float(
        rollout_results["reward"].mean(axis=-1).std()
    )  # Calc mean for each rollout, then std of those
    eval_output["cumulative_discounted_return_mean"] = float(
        rollout_results["cum_return"].mean()
    )  # One per rollout
    eval_output["cumulative_discounted_return_std"] = float(
        rollout_results["cum_return"].std()
    )  # One per rollout

    kpi_function = get_kpi_function(
        cfg.rollout_wrapper.env_id, **cfg.rollout_wrapper.env_kwargs
    )
    kpis_per_rollout = kpi_function(rollout_results=rollout_results)
    for k, v in kpis_per_rollout.items():
        eval_output[f"{k}_mean"] = float(v.mean())
        eval_output[f"{k}_std"] = float(v.std())
    return eval_output


def create_evaluation_output_df(
    cfg: DictConfig, rollout_results: Dict[str, chex.Array]
) -> pd.DataFrame:
    """Create a dataframe with one row per evaluation rollout, once column per metric including KPIs"""
    kpi_function = get_kpi_function(
        cfg.rollout_wrapper.env_id, **cfg.rollout_wrapper.env_kwargs
    )
    eval_output = kpi_function(rollout_results=rollout_results)
    eval_output["daily_undiscounted_reward"] = rollout_results["reward"].mean(axis=-1)
    eval_output["cumulative_discounted_return"] = rollout_results[
        "cum_return"
    ].squeeze()
    return pd.DataFrame(eval_output)
