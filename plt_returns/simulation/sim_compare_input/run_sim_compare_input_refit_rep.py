import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from viso_jax.policies.heuristic_policy import HeuristicPolicy
from viso_jax.utils.yaml import to_yaml
from plt_returns.utils.simopt import (
    run_simopt,
    evaluation_run_with_simulated_model,
    process_params_for_log,
    process_params_for_df,
)


# Enable logging
log = logging.getLogger(__name__)


def run_simopt_and_evaluate(
    rep_policy: HeuristicPolicy,
    cfg: DictConfig,
    combination_name: str,
    issue_name: str,
    sensitivity: float,
    specificity: float,
    simopt_trials_path: Path,
    initial_params: Optional[Dict[str, int]] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    cfg.rollout_wrapper.env_params.return_prediction_model_sensitivity = sensitivity
    cfg.rollout_wrapper.env_params.return_prediction_model_specificity = specificity

    study = run_simopt(cfg, rep_policy, initial_params)

    trials_df = study.trials_dataframe()
    cur_trials_path = simopt_trials_path / f"{combination_name}_{issue_name}_trials.csv"
    trials_df.to_csv(cur_trials_path)

    params_output = {}
    best_params = np.array([v for v in study.best_params.values()]).reshape(
        rep_policy.params_shape
    )
    params_output[f"policy_params_issue_{issue_name}"] = process_params_for_log(
        rep_policy, best_params
    )
    params_dict = study.best_params
    labelled_params_for_df = process_params_for_df(rep_policy, best_params)
    policy_params = jnp.array(best_params)

    # Evaluate best replenishment policy under OUFO issuing
    (
        evaluation_output,
        evaluation_output_summary_row,
    ) = evaluation_run_with_simulated_model(
        sensitivity, specificity, cfg, rep_policy, policy_params
    )
    evaluation_output_summary_row["name"] = combination_name
    evaluation_output_summary_row["issuing"] = issue_name
    evaluation_output_summary_row = pd.DataFrame(
        labelled_params_for_df | evaluation_output_summary_row, index=[0]
    )
    return params_output, params_dict, evaluation_output, evaluation_output_summary_row


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """For different combinations of input params, and each specified issuing policy
    run simulation optimization using Optuna to find the best parameters for  the replenishment
    policy and evaluate the policy using the best parameters on a separate set of rollouts
    """

    # Create a directory to store the evaluation outputs for each sens/spec combination
    evaluation_output_path = Path("./evaluation_output")
    evaluation_output_path.mkdir(parents=True, exist_ok=True)

    # Create a directory to store the simopt outputs for each sens/spec combination
    simopt_trials_path = Path("./simopt_trials")
    simopt_trials_path.mkdir(parents=True, exist_ok=True)

    output_info = {}
    rep_policy = hydra.utils.instantiate(cfg.policy)
    res = pd.DataFrame()
    # Initial params are None until we fit first replenishment policy
    # with OUFO-equivalent issuing policy
    initial_params = None

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

        for issue_policy_name, issue_policy_params in OmegaConf.to_container(
            cfg.issue_policies
        ).items():
            sensitivity = issue_policy_params["sensitivity"]
            specificity = issue_policy_params["specificity"]

            # Run simulation optimization and evaluate
            (
                rep_params_output,
                rep_params_dict,
                evaluation_output,
                evaluation_output_summary_row,
            ) = run_simopt_and_evaluate(
                rep_policy,
                cfg,
                combination.name,
                issue_policy_name,
                sensitivity,
                specificity,
                simopt_trials_path,
                initial_params,
            )

            # If trial was an OUFO trial, update initial params
            # They will be used as first trial in runs until next time
            # we do an OUFO run.
            # Use OUFO from one run as starting point for different sesn/spec with
            # same settings, and for OUFO of the next setting
            if sensitivity == 0 and specificity == 1:
                initial_params = rep_params_dict

            # Save down evaluation output to csv
            cur_eval_path = (
                evaluation_output_path
                / f"{combination.name}_{issue_policy_name}_evaluation_output.csv"
            )
            evaluation_output.to_csv(cur_eval_path)

            output_info[combination.name][issue_policy_name] = rep_params_output
            results_list.append(evaluation_output_summary_row)

        res = pd.concat([res, *results_list], ignore_index=True)

    # Any final logging that combines results across different combinations of input params
    res = res.set_index(["name", "issuing"], drop=True)
    res.to_csv("results.csv")
    log.info("Final results saved")

    to_yaml(output_info, "output_info.yaml")
    log.info("Evaluation output saved")


if __name__ == "__main__":
    main()
