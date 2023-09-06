import jax.numpy as jnp
from functools import partial
from typing import List
import chex
from viso_jax.utils.yaml import from_yaml
from plt_returns.replenishment_policies.heuristic_policy import HeuristicPolicy
from gymnax.environments.environment import Environment, EnvParams


class StandingOrderPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the column names for the policy parameters, in this case the standing order quantity"""
        return ["Q"]

    def _get_param_row_names(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        return []

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        # Same policy valid for the env with simulated and real demand/returns
        if env_id == "PlateletBank" or "PlateletBankRealInput":
            return platelet_bank_standing_order_policy
        else:
            raise NotImplementedError(
                f"No standing order policy defined for Environment ID {env_id}"
            )

    @classmethod
    def valid_params(cls, params: chex.Array) -> bool:
        """Order quantity must be greater than or equal to 0"""
        return jnp.all(jnp.array(params) >= 0)


def base_standing_order_policy(
    Q: int, total_stock: int, policy_params: chex.Array
) -> chex.Array:
    """Basic standing order policy for all environments"""
    return jnp.clip(Q, 0)


def platelet_bank_standing_order_policy(
    policy_params: chex.Array, obs: chex.Array, rng: chex.PRNGKey
) -> chex.Array:
    """Standing order policy for PlateletBankGymnax environment"""
    # policy_params = [[Q]], nest for consistency with other policies
    return jnp.array(policy_params[0, 0])
