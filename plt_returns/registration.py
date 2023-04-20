# This is based on gymnax.registration by Robert T. Lange
# https://github.com/RobertTLange/gymnax/blob/main/gymnax/registration.py
# Modified from commit b9f4795
from typing import Tuple
from gymnax.environments.environment import Environment, EnvParams
from plt_returns.environments import (
    PlateletBankGymnax,
)


def make(env_id: str, **env_kwargs) -> Tuple[Environment, EnvParams]:
    """Version of gymnax.make/OpenAI gym.make for our envs"""

    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered gymnax environments.")
    if env_id == "PlateletBank":
        env = PlateletBankGymnax(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params


registered_envs = [
    "PlateletBank",
]
