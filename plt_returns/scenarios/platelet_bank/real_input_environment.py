from typing import Tuple, Optional, Union, Dict, List
import chex

import jax
import jax.numpy as jnp
from jax import lax

import gymnax
from gymnax.environments import environment, spaces

from flax import struct
import numpyro
import distrax
import pandas as pd

from plt_returns.scenarios.platelet_bank.environment import (
    EnvState,
    DemandInfo,
    PlateletBankGymnax,
)
from plt_returns.utils.real_input import real_input_df_to_array
from omegaconf import OmegaConf


@struct.dataclass
class EnvParams:
    slippage: float
    initial_stock: chex.Array
    age_on_arrival_distributions: chex.Array
    variable_order_cost: int
    fixed_order_cost: int
    shortage_cost: int
    expiry_cost: int
    holding_cost: int
    slippage_cost: int
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        slippage=0.0,
        initial_stock: List[int] = [0, 0, 0],
        age_on_arrival_distributions: List[Union[List[int], int]] = [
            [0, 0, 1] for i in range(7)
        ],
        variable_order_cost: int = -650,
        fixed_order_cost: int = -225,
        shortage_cost: int = -3250,
        expiry_cost: int = -650,
        holding_cost: int = -130,
        slippage_cost: int = -650,
        max_steps_in_episode: int = 365,
        gamma: float = 1.0,
    ):
        # TODO: Check that all inputs are correct types/shapes/in ranges

        if OmegaConf.is_list(age_on_arrival_distributions):
            age_on_arrival_distributions = OmegaConf.to_container(
                age_on_arrival_distributions, resolve=True
            )

        age_on_arrival_distributions_jax = EnvParams.age_on_arrival_to_jax_array(
            age_on_arrival_distributions
        )
        return EnvParams(
            slippage,
            jnp.array(initial_stock),
            age_on_arrival_distributions_jax,
            variable_order_cost,
            fixed_order_cost,
            shortage_cost,
            expiry_cost,
            holding_cost,
            slippage_cost,
            max_steps_in_episode,
            gamma,
        )

    @classmethod
    def age_on_arrival_to_jax_array(
        cls, age_on_arrival_distributions: List
    ) -> chex.Array:
        """Convert age_on_arrival_distributions argument to jax array with one row for each
        day of the week. Accept a single list (which is repeated for each weekday), or a list of 7 lists)
        """
        if isinstance(age_on_arrival_distributions, list):
            # if the first element is also a list, then we assume it's a list of lists
            if isinstance(age_on_arrival_distributions[0], list):
                if len(age_on_arrival_distributions) == 7:
                    return jnp.array(age_on_arrival_distributions)
                else:
                    raise ValueError(
                        "Expected a list of 7 lists. Got a list of {} lists.".format(
                            len(age_on_arrival_distributions)
                        )
                    )
            else:
                # if it's a single list, repeat it seven times
                return jnp.array([age_on_arrival_distributions] * 7)
        else:
            raise TypeError(
                "Expected a list or a list of lists. Got {}.".format(
                    type(age_on_arrival_distributions)
                )
            )


class PlateletBankGymnaxRealInput(PlateletBankGymnax):
    # We need to pass in max_useful_life because it affects array shapes
    # We need to pass in max_order_quantity because self.num_actions depends on it
    # We need to pass in max_demand because it affects array shapes
    def __init__(
        self,
        max_useful_life: int = 3,
        max_order_quantity: int = 100,
        max_demand: int = 100,
        return_detailed_info: bool = False,
        real_input_filepath: str = None,
        start_date: str = "2021-01-01",  # TODO update this
        period_split_hour: int = 12,
    ):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity
        self.max_demand = max_demand  # May demand applies to both am and pm, should be set quite high; sampled demand clipped using this value
        if return_detailed_info:
            self._get_detailed_info_fn = self._get_detailed_info
        else:
            self._get_detailed_info_fn = lambda _, *args, **kwargs: {}

        if real_input_filepath is None:
            raise ValueError("Must specify real_data_filepath")
        real_input_df = pd.read_csv(real_input_filepath, header=0)
        self.real_input_array = real_input_df_to_array(
            real_input_df, max_demand, start_date, period_split_hour
        )  # Add function to utils and import
        self.real_input_days = pd.to_datetime(
            real_input_df["prediction_point_timestamp"]
        ).dt.date.nunique()
        self.start_date = start_date

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @property
    def name(self) -> str:
        """Environment name."""
        return "PlateletBankGymnaxRealInput"

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        # Split key, one for each random process
        (
            key,
            arrival_key,
            demand_key_am,
            returned_key_am,
            pred_return_key_am,
            issue_key_am,
            demand_key_pm,
            returned_key_pm,
            slippage_key,
            pred_return_key_pm,
            issue_key_pm,
        ) = jax.random.split(key, 11)

        # Get the weekday
        weekday = state.weekday

        # Get the age on arrival distribution for the weekday
        age_on_arrival_distribution = params.age_on_arrival_distributions[weekday]

        # Receive order, with random distributed remaining useful lives
        received_order = distrax.Multinomial(
            action, probs=age_on_arrival_distribution
        ).sample(seed=arrival_key)
        opening_stock_am = state.stock + received_order

        ## Pre-return activity

        # Generate pre_return demand, plus arrays of whether they will be returned and simulated prediction of return
        # Sampled arrays are all of length max_demand so we can JIT (effectively padding)

        real_input = self.real_input_array[state.step]
        real_input_am = real_input[0]

        demand_am = jnp.sum(real_input_am[0].sum())
        return_samples_am = real_input_am[1]
        return_pred_samples_am = real_input_am[2]

        demand_info_am = DemandInfo(
            demand_am,
            demand_am,
            opening_stock_am,
            jnp.zeros_like(opening_stock_am),
            return_samples_am,
            return_pred_samples_am,
            age_on_arrival_distribution,  # For the specific weekday
            issue_key_am,
        )

        # Fill pre-return demand if possible
        (
            closing_stock_am,
            to_be_returned_am,
            backorders_am,
            to_be_returned_from_stock_am,
            to_be_returned_from_emergency_orders_am,
        ) = self._fill_demand(demand_info_am)

        ## Returns come back into stock
        # NEW at this point we find out how many units expired while out on the wards
        returned_units = state.to_be_returned
        expiries_from_returned = returned_units[0]
        slippage_units = numpyro.distributions.Binomial(
            total_count=returned_units[1 : self.max_useful_life], probs=params.slippage
        ).sample(key=slippage_key)
        back_in_stock_from_returned = jnp.hstack(
            [returned_units[1 : self.max_useful_life] - slippage_units, 0]
        )  # Aging the units being returned
        opening_stock_pm = closing_stock_am + back_in_stock_from_returned

        ## Post-return activity

        # Generate post_return demand, plus arrays of whether they will be returned and simulated prediction of return
        # Sampled arrays are all of length max_demand so we can JIT (effectively padding)
        real_input_pm = real_input[1]

        demand_pm = jnp.sum(real_input_pm[0].sum())
        return_samples_pm = real_input_pm[1]
        return_pred_samples_pm = real_input_pm[2]

        demand_info_pm = DemandInfo(
            demand_pm,
            demand_pm,
            opening_stock_pm,
            jnp.zeros_like(opening_stock_pm),
            return_samples_pm,
            return_pred_samples_pm,
            age_on_arrival_distribution,
            issue_key_pm,
        )

        # Fill post-return demand if possible
        (
            closing_stock,
            to_be_returned_pm,
            backorders_pm,
            to_be_returned_from_stock_pm,
            to_be_returned_from_emergency_orders_pm,
        ) = self._fill_demand(demand_info_pm)

        # Age stock and calculate expiries
        to_be_returned = to_be_returned_am + to_be_returned_pm
        stock_expiries = closing_stock[0]
        expiries = stock_expiries + expiries_from_returned
        slippage = slippage_units.sum()
        closing_stock = jnp.hstack([closing_stock[1 : self.max_useful_life], 0])

        # Note now we don;t age stock in to be returned, because expiries for those units
        # accounted for on the day returned
        # Calculate reward
        backorders = backorders_am + backorders_pm
        reward = self._calculate_reward(
            action,
            closing_stock,
            backorders,
            stock_expiries,
            expiries_from_returned,
            slippage,
            params,
        )

        # Update the state

        state = EnvState(
            closing_stock, to_be_returned, (weekday + 1) % 7, state.step + 1
        )
        done = self.is_terminal(state, params)

        # More expensive info to compute, so only do it if needed
        detailed_info = self._get_detailed_info_fn(
            demand_am,
            return_samples_am,
            return_pred_samples_am,
            demand_pm,
            return_samples_pm,
            return_pred_samples_pm,
        )
        info = {
            "weekday": weekday,
            "discount": self.discount(state, params),
            "cumulative_gamma": cumulative_gamma,
            "demand_am": demand_am,
            "demand_pm": demand_pm,
            "demand": demand_am + demand_pm,
            "expiries": expiries,
            "stock_expiries": stock_expiries,
            "expiries_from_returned": expiries_from_returned,
            "shortage": backorders,
            "slippage": slippage,
            "holding": jnp.sum(closing_stock),
            "units_in_stock": jnp.sum(closing_stock),
            "opening_stock": jnp.sum(opening_stock_am),
            "received_order": received_order,
            "closing_stock_am": jnp.sum(closing_stock_am),
            "returned_into_stock": jnp.sum(back_in_stock_from_returned),
            "opening_stock_pm": jnp.sum(opening_stock_pm),
            "backorders_am": backorders_am,
            "backorder_pm": backorders_pm,
            "stock_expiries": stock_expiries,
            "expiries_from_returned": expiries_from_returned,
            "to_be_returned": to_be_returned,
            "to_be_returned_from_stock_am": jnp.sum(to_be_returned_from_stock_am),
            "to_be_returned_from_emergency_orders_am": jnp.sum(
                to_be_returned_from_emergency_orders_am
            ),
            "to_be_returned_from_stock_pm": jnp.sum(to_be_returned_from_stock_pm),
            "to_be_returned_from_emergency_orders_pm": jnp.sum(
                to_be_returned_from_emergency_orders_pm
            ),
        }

        info = info | detailed_info

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    # NOTE: Starting with zero inventory here
    # This is what we did before,
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # The day of the week of the start date
        weekday = pd.to_datetime(self.start_date).day_of_week

        state = EnvState(
            stock=params.initial_stock,
            to_be_returned=jnp.zeros_like(params.initial_stock),
            weekday=weekday,
            step=0,
        )
        return self.get_obs(state), state

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = jax.lax.bitwise_or(
            state.step >= params.max_steps_in_episode,
            state.step >= self.real_input_days,
        )
        return done_steps
