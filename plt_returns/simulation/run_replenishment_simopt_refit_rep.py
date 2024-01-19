import hydra
from omegaconf.dictconfig import DictConfig
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from optuna.study import Study
import jax
import jax.numpy as jnp
import numpy as np
import chex
import matplotlib.pyplot as plt
from plt_returns.evaluation.evaluate_policy import (
    create_evaluation_output_summary,
    create_evaluation_output_df,
)
from plt_returns.replenishment_policies.heuristic_policy import HeuristicPolicy
from plt_returns.utils.plotting import plot_heatmap
from plt_returns.utils.simopt import (
    simopt_grid_sampler,
    simopt_other_sampler,
    run_simopt,
    evaluation_run_with_simulated_model,
    process_params_for_log,
    process_params_for_df,
)
from viso_jax.utils.yaml import to_yaml

# Enable logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run simulation optimization using Optuna to find the best parameters for a policy,
    and evaluate the policy using the best parameters on a separate set of rollouts"""
    start_time = datetime.now()

    # Create a directory to store the simopt outputs for each sens/spec combination
    simopt_trials_path = Path("./simopt_trials")
    simopt_trials_path.mkdir(parents=True, exist_ok=True)

    # Create a directory to store the evaluation outputs for each sens/spec combination
    evaluation_output_path = Path("./evaluation_output")
    evaluation_output_path.mkdir(parents=True, exist_ok=True)

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

    sensitivities = hydra.utils.instantiate(cfg.metric_ranges.sensitivity).round(2)
    specificities = hydra.utils.instantiate(cfg.metric_ranges.specificity).round(2)

    pred_return_res = pd.DataFrame()
    for sens in sensitivities:
        for spec in specificities:
            if sens == 0.0 and spec == 1.0:  # Equivalent to OUFO
                policy_params = oufo_params
            else:
                rep_study = run_simopt(cfg, policy, oufo_params_dict, sens, spec)
                trials_df = rep_study.trials_dataframe()
                cur_trials_path = (
                    simopt_trials_path / f"sens_{sens}_spec_{spec}_trials.csv"
                )
                trials_df.to_csv(cur_trials_path)
                policy_params = np.array(
                    [v for v in rep_study.best_params.values()]
                ).reshape(policy.params_shape)
            labelled_params_for_df = process_params_for_df(policy, policy_params)
            policy_params = jnp.array(policy_params)

            (
                evaluation_output,
                evaluation_output_summary_row,
            ) = evaluation_run_with_simulated_model(
                sens, spec, cfg, policy, policy_params
            )
            # Save down evaluation output to csv
            cur_eval_path = (
                evaluation_output_path
                / f"sens_{sens}_spec_{spec}_evaluation_output.csv"
            )
            evaluation_output.to_csv(cur_eval_path)

            new_row = pd.DataFrame(
                labelled_params_for_df | evaluation_output_summary_row, index=[0]
            )
            # Add to res dataframe
            pred_return_res = pd.concat([pred_return_res, new_row], ignore_index=True)

    pred_return_res["1-specificity"] = (1 - pred_return_res["specificity"]).round(2)

    performance_by_sens_spec = pred_return_res.set_index(
        ["sensitivity", "specificity"], drop=True
    )
    performance_by_sens_spec.to_csv("performance_by_sens_spec.csv")

    # Expiry heatmap
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
