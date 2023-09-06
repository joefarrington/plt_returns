import jax
import jax.numpy as jnp
from typing import List
import chex
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
        if env_id == "PlateletBank" or "PlateletBankRealInput":
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        else:
            return []

    def _get_forward_method(
        self, env_id: str, env: Environment, env_params: EnvParams
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        # Same policy valid for the env with simulated and real demand/returns
        if env_id == "PlateletBank" or "PlateletBankRealInput":
            return platelet_bank_sS_policy
        else:
            raise ValueError(f"No (s,S) policy defined for Environment ID {env_id}")

    @classmethod
    def valid_params(cls, params: chex.Array) -> bool:
        """s must be less than S; all must be greater than or equal to zero"""
        return jax.lax.bitwise_and(
            jnp.all(params[:, 0] < params[:, 1]), jnp.all(params >= 0)
        )


def base_sS_policy(
    s: int, S: int, total_stock: int, policy_params: chex.Array
) -> chex.Array:
    """Basic (s, S) policy for all environments"""
    # s should be less than S
    # Enforce that constraint here, order only made when constraint met
    constraint_met = sSPolicy.valid_params(policy_params)
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
