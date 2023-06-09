import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, Any, List
import chex
import pandas as pd
import viso_jax
import plt_returns
from viso_jax.utils.yaml import from_yaml
from gymnax.environments.environment import Environment
from viso_jax.policies.heuristic_policy import (
    HeuristicPolicy as VJHeuristicPolicy,
)

# This class is essentially the same as viso_jax.policies.heuristic_policy, but
# uses the make() function from plt_returns to create the environment so that we can make
# environments registered in this repository.


class HeuristicPolicy(VJHeuristicPolicy):
    def __init__(
        self,
        env_id: str,
        env_kwargs: Optional[Dict[str, Any]] = {},
        env_params: Optional[Dict[str, Any]] = {},
        policy_params_filepath: Optional[str] = None,
    ):
        # As in utils/rollout.py env_kwargs and env_params arguments are dicts to
        # override the defaults for an environment.

        # Instantiate an internal envinronment we'll use to access env kwargs/params
        # These are not stored, just used to set up param_col_names, param_row_names and forward
        self.env_id = env_id
        env, default_env_params = plt_returns.make(self.env_id, **env_kwargs)
        all_env_params = default_env_params.create_env_params(**env_params)

        self.param_col_names = self._get_param_col_names(env_id, env, all_env_params)
        self.param_row_names = self._get_param_row_names(env_id, env, all_env_params)
        self.forward = self._get_forward_method(env_id, env, all_env_params)

        if self.param_row_names != []:
            self.param_names = np.array(
                [
                    [f"{p}_{r}" for p in self.param_col_names]
                    for r in self.param_row_names
                ]
            )
        else:
            self.param_names = np.array([self.param_col_names])

        self.params_shape = self.param_names.shape

        if policy_params_filepath:
            self.policy_params = self.load_policy_params(policy_params_filepath)

    @classmethod
    def valid_params(cls, params: chex.Array) -> bool:
        """Return True is proposed params are valid for this policy, False otherwise"""
        raise NotImplementedError
