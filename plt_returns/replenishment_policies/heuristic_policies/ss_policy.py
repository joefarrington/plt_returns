import jax.numpy as jnp
import numpy as np
from typing import Optional, List
from functools import partial
import chex
import pandas as pd
from viso_jax.utils.yaml import from_yaml
from plt_returns.replenishment_policies.heuristic_policy import HeuristicPolicy
from gymnax.environments.environment import Environment, EnvParams


class sSPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["s", "S"]

    def _get_param_row_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if env_id == "PlateletBank":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            return []

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> callable:
        if env_id == "PlateletBank":
            return platelet_bank_sS_policy
        else:
            raise ValueError(f"No (s,S) policy defined for Environment ID {env_id}")


def base_sS_policy(
    s: int, S: int, total_stock: int, policy_params: chex.Array
) -> chex.Array:
    """Basic (s, S) policy for all environments"""
    # s should be less than S
    # Enforce that constraint here, order only made when constraint met
    constraint_met = jnp.all(policy_params[:, 0] < policy_params[:, 1])
    return jnp.where((total_stock <= s) & (constraint_met), S - total_stock, 0)


def platelet_bank_sS_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """(s,S) policy for PlateletBankGymnax environment"""
    # policy_params = [[s_Mon, S_Mon], ..., [s_Sun, S_Sun]]
    weekday = obs[0]
    s = policy_params[weekday][0]
    S = policy_params[weekday][1]
    total_stock = jnp.sum(obs[1:])  # First element of obs is weekday
    order = base_sS_policy(s, S, total_stock, policy_params)
    return jnp.array(order)