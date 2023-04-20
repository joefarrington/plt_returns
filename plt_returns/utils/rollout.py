from typing import Optional, Callable, Dict, Any, Tuple
from viso_jax.utils.rollout import RolloutWrapper as VJRolloutWrapper
import plt_returns

# This class is essentially the same as viso_jax.utils.rollout.RolloutWrapper, but
# uses the make() function from plt_returns to create the environment so that we can make
# environments registered in this repository.


class RolloutWrapper(VJRolloutWrapper):
    def __init__(
        self,
        model_forward: Callable = None,
        env_id: str = "PlateletBank",
        num_env_steps: Optional[int] = None,
        env_kwargs: Dict[str, Any] = {},
        env_params: Dict[str, Any] = {},
        num_burnin_steps: int = 0,
        return_info: bool = False,
    ):
        """Wrapper to define batch evaluation for policy parameters."""
        self.env_id = env_id
        # Define the RL environment & network forward function
        self.env, default_env_params = plt_returns.make(self.env_id, **env_kwargs)

        if num_env_steps is None:
            self.num_env_steps = default_env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

        # Run a total of num_burnin_steps + num_env_steps
        # The burn-in steps are run first, and not included
        # in the reported outputs
        self.num_burnin_steps = num_burnin_steps

        # None of our environments have a fixed number of steps
        # so set to match desired number of steps
        env_params["max_steps_in_episode"] = self.num_env_steps + self.num_burnin_steps
        self.env_params = default_env_params.create_env_params(**env_params)
        self.model_forward = model_forward

        # If True, include info from each step in output
        self.return_info = return_info
