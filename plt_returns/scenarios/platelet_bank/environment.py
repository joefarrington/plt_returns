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


# TODO:
# PyTree functions at the end like in viso_jax
# Lead times?
# Weekday in obs one-hot encoded or not?
# Truncation information?
# Separate demand provider-type thing?
# Use jnp_int to deal with single/double precision flexibly
# Allow initial weekday to be changed like in Mirjalili env
# Verification of vars in create_env_params
# Consistent prob package? Numpyro, distrax, tfp


@struct.dataclass
class EnvState:
    stock: chex.Array
    to_be_returned: chex.Array
    weekday: int
    step: int


@struct.dataclass
class DemandInfo:
    total_demand: int
    remaining_demand: int
    remaining_stock: chex.Array
    to_be_returned: chex.Array
    return_samples: chex.Array
    return_pred_samples: chex.Array
    age_on_arrival_distribution: chex.Array
    key: jax.random.PRNGKey


@struct.dataclass
class EnvParams:
    poisson_mean_demand_pre_return: chex.Array
    poisson_mean_demand_post_return: chex.Array
    slippage: float
    initial_weekday: int
    initial_stock: chex.Array
    age_on_arrival_distributions: chex.Array
    variable_order_cost: int
    fixed_order_cost: int
    shortage_cost: int
    expiry_cost: int
    holding_cost: int
    slippage_cost: int
    max_steps_in_episode: int
    return_probability: float
    return_prediction_model_sensitivity: float
    return_prediction_model_specificity: float
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        poisson_mean_demand_pre_return: chex.Array = [
            37.5,
            37.3,
            39.2,
            37.8,
            40.5,
            27.2,
            28.4,
        ],
        poisson_mean_demand_post_return: chex.Array = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        slippage=0.0,
        initial_weekday: int = 6,  # At the first observation, it is Sunday evening
        initial_stock: chex.Array = [0, 0, 0],
        age_on_arrival_distributions: chex.Array = [[0, 0, 1] for i in range(7)],
        variable_order_cost: int = -650,
        fixed_order_cost: int = -225,
        shortage_cost: int = -3250,
        expiry_cost: int = -650,
        holding_cost: int = -130,
        slippage_cost: int = -650,
        max_steps_in_episode: int = 365,
        return_probability: float = 0.0,
        return_prediction_model_sensitivity: float = 1,
        return_prediction_model_specificity: float = 1,
        gamma: float = 1.0,
    ):

        # TODO: Check that all inputs are correct types/shapes/in ranges

        return EnvParams(
            jnp.array(poisson_mean_demand_pre_return),
            jnp.array(poisson_mean_demand_post_return),
            slippage,
            initial_weekday,
            jnp.array(initial_stock),
            jnp.array(age_on_arrival_distributions),
            variable_order_cost,
            fixed_order_cost,
            shortage_cost,
            expiry_cost,
            holding_cost,
            slippage_cost,
            max_steps_in_episode,
            return_probability,
            return_prediction_model_sensitivity,
            return_prediction_model_specificity,
            gamma,
        )


class PlateletBankGymnax(environment.Environment):
    # We need to pass in max_useful_life because it affects array shapes
    # We need to pass in max_order_quantity because self.num_actions depends on it
    # We need to pass in max_demand because it affects array shapes
    def __init__(
        self,
        max_useful_life: int = 6,
        max_order_quantity: int = 100,
        max_demand: int = 100,
    ):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity
        self.max_demand = max_demand  # May demand applies to both am and pm, should be set quite high; sampled demand clipped using this value

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

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

        # Update weekday
        weekday = (state.weekday + 1) % 7

        # Get the age on arrival distribution for the weekday
        age_on_arrival_distribution = params.age_on_arrival_distributions[weekday]

        # Receive order, with random distributed remaining useful lives
        opening_stock_am = state.stock + distrax.Multinomial(
            action, age_on_arrival_distribution
        ).sample(seed=arrival_key)

        ## Pre-return activity

        # Generate pre_return demand, plus arrays of whether they will be returned and simulated prediction of return
        # Sampled arrays are all of length max_demand so we can JIT (effectively padding)
        demand_am = jax.random.poisson(
            demand_key_am, params.poisson_mean_demand_pre_return[weekday]
        ).clip(0, self.max_demand)
        return_samples_am = distrax.Bernoulli(probs=params.return_probability).sample(
            seed=returned_key_am, sample_shape=self.max_demand + 1
        )
        pred_sample_am = distrax.Uniform().sample(
            seed=pred_return_key_am, sample_shape=self.max_demand + 1
        )
        return_pred_samples_am = jnp.where(
            return_samples_am,
            pred_sample_am < params.return_prediction_model_sensitivity,
            pred_sample_am > params.return_prediction_model_specificity,
        ).astype(jnp.int32)

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
        demand_pm = jax.random.poisson(
            demand_key_pm, params.poisson_mean_demand_post_return[weekday]
        ).clip(0, self.max_demand)
        return_samples_pm = distrax.Bernoulli(probs=params.return_probability).sample(
            seed=returned_key_pm, sample_shape=self.max_demand + 1
        )
        pred_sample_pm = distrax.Uniform().sample(
            seed=pred_return_key_pm, sample_shape=self.max_demand + 1
        )
        return_pred_samples_pm = jnp.where(
            return_samples_pm,
            pred_sample_pm < params.return_prediction_model_sensitivity,
            pred_sample_pm > params.return_prediction_model_specificity,
        ).astype(jnp.int32)

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
            action, closing_stock, backorders, expiries, slippage, params
        )

        # Update the state

        state = EnvState(closing_stock, to_be_returned, weekday, state.step + 1)
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {
                "weekday": weekday,
                "discount": self.discount(state, params),
                "cumulative_gamma": cumulative_gamma,
                "demand_am": demand_am,
                "demand_pm": demand_pm,
                "demand": demand_am + demand_pm,
                "expiries": expiries,
                "shortage": backorders,
                "slippage": slippage,
                "holding": jnp.sum(closing_stock),
                "units_in_stock": jnp.sum(closing_stock),
                "opening_stock": jnp.sum(opening_stock_am),
                "closing_stock_am": jnp.sum(closing_stock_am),
                "returned_into_stock": jnp.sum(back_in_stock_from_returned),
                "opening_stock_pm": jnp.sum(opening_stock_pm),
                "backorders_am": backorders_am,
                "backorder_pm": backorders_pm,
                "stock_expiries": stock_expiries,
                "expiries_from_returned": expiries_from_returned,
                "to_be_returned_from_stock_am": jnp.sum(to_be_returned_from_stock_am),
                "to_be_returned_from_emergency_orders_am": jnp.sum(
                    to_be_returned_from_emergency_orders_am
                ),
                "to_be_returned_from_stock_pm": jnp.sum(to_be_returned_from_stock_pm),
                "to_be_returned_from_emergency_orders_pm": jnp.sum(
                    to_be_returned_from_emergency_orders_pm
                ),
            },
        )

    # NOTE: Starting with zero inventory here
    # This is what we did before,
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        state = EnvState(
            stock=params.initial_stock,
            to_be_returned=jnp.zeros_like(params.initial_stock),
            weekday=params.initial_weekday,
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return jnp.array([state.weekday, *state.stock])

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.step >= params.max_steps_in_episode
        return done_steps

    def cumulative_gamma(self, state: EnvState, params: EnvParams) -> float:
        """Return cumulative discount factor"""
        return params.gamma**state.step

    def _calculate_reward(
        self,
        action: int,
        closing_stock: chex.Array,
        backorders: int,
        expiries: int,
        slippage: int,
        params: EnvParams,
    ) -> int:
        costs = jnp.array(
            [
                params.fixed_order_cost,
                params.variable_order_cost,
                params.holding_cost,
                params.shortage_cost,
                params.expiry_cost,
                params.slippage_cost,
            ]
        )
        values = jnp.array(
            [
                jnp.where(action > 0, 1, 0),
                action,
                jnp.sum(closing_stock),
                backorders,
                expiries,
                slippage,
            ]
        )
        return jnp.dot(costs, values)

    def _fill_demand(self, initial_demand_info):
        total_stock = jnp.sum(initial_demand_info.remaining_stock)
        backorders = jnp.where(
            initial_demand_info.total_demand > total_stock,
            initial_demand_info.total_demand - total_stock,
            0,
        )
        # Issue units from stock
        demand_info = jax.lax.while_loop(
            self._remaining_demand_and_stock, self._issue_one_unit, initial_demand_info
        )
        to_be_returned_from_stock = demand_info.to_be_returned
        # Emergency orders if remaining demand after stock used
        demand_info = jax.lax.while_loop(
            self._remaining_demand,
            self._emergency_order_and_issue_one_unit,
            demand_info,
        )
        to_be_returned_from_emergency_orders = (
            demand_info.to_be_returned - to_be_returned_from_stock
        )
        return (
            demand_info.remaining_stock,
            demand_info.to_be_returned,
            backorders,
            to_be_returned_from_stock,
            to_be_returned_from_emergency_orders,
        )

    def _remaining_demand_and_stock(self, demand_info):
        # Only continue if there is both remaining demand to fill and stock to fill it
        return (demand_info.remaining_demand > 0) & (
            demand_info.remaining_stock.sum() > 0
        )

    def _issue_one_unit(self, demand_info):
        idx = demand_info.total_demand - demand_info.remaining_demand
        remaining_demand = demand_info.remaining_demand - 1
        # Identify age of unit to be issued
        remaining_useful_life_of_issued = jax.lax.cond(
            demand_info.return_pred_samples[idx],
            self._yufo,
            self._oufo,
            demand_info.remaining_stock,
        )
        issued = (
            jnp.zeros(self.max_useful_life, dtype=jnp.int32)
            .at[remaining_useful_life_of_issued]
            .add(1)
        )
        remaining_stock = demand_info.remaining_stock - issued
        to_be_returned = demand_info.to_be_returned + jnp.where(
            demand_info.return_samples[idx] == 1,
            issued,
            jnp.zeros(self.max_useful_life, dtype=jnp.int32),
        )
        return DemandInfo(
            demand_info.total_demand,
            remaining_demand,
            remaining_stock,
            to_be_returned,
            demand_info.return_samples,
            demand_info.return_pred_samples,
            demand_info.age_on_arrival_distribution,
            demand_info.key,
        )

    def _remaining_demand(self, demand_info):
        return demand_info.remaining_demand > 0

    def _emergency_order_and_issue_one_unit(self, demand_info):
        idx = demand_info.total_demand - demand_info.remaining_demand
        remaining_demand = demand_info.remaining_demand - 1
        carry_key, order_key = jax.random.split(demand_info.key, 2)
        # Same age of unit from age at arrival distribution
        issued = numpyro.distributions.Multinomial(
            total_count=1, probs=demand_info.age_on_arrival_distribution
        ).sample(key=order_key)
        remaining_stock = demand_info.remaining_stock  # No change, should remain 0
        to_be_returned = demand_info.to_be_returned + jnp.where(
            demand_info.return_samples[idx] == 1,
            issued,
            jnp.zeros(self.max_useful_life, dtype=jnp.int32),
        )
        return DemandInfo(
            demand_info.total_demand,
            remaining_demand,
            remaining_stock,
            to_be_returned,
            demand_info.return_samples,
            demand_info.return_pred_samples,
            demand_info.age_on_arrival_distribution,
            carry_key,
        )

    def _oufo(self, remaining_stock):
        return jnp.where(remaining_stock > 0, True, False).argmax()

    def _yufo(self, remaining_stock):
        return (self.max_useful_life - 1) - jnp.where(remaining_stock > 0, True, False)[
            ::-1
        ].argmax()

    @property
    def name(self) -> str:
        """Environment name."""
        return "PlateletBankGymnax"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.max_order_quantity + 1

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Discrete(params.max_order_quantity + 1)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # [weekday, oldest_stock, ..., freshest_stock]
        max_stock_by_age = self.max_useful_life * params.max_order_quantity
        low = jnp.array([0] * self.max_useful_life)
        high = jnp.array([6] + [max_stock_by_age] * (self.max_useful_life))
        return spaces.Box(low, high, (self.max_useful_life + 1,), dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        max_stock_by_age = self.max_useful_life * params.max_order_quantity
        return spaces.Dict(
            {
                "stock": spaces.Box(
                    0, max_stock_by_age, (self.max_useful_life,), jnp.int32
                ),
                "to_be_returned": spaces.Box(
                    0, max_stock_by_age, (self.max_useful_life,), jnp.int32
                ),
                "step": spaces.Discrete(params.max_steps_in_episode),
                "weekday": spaces.Discrete(7),
            }
        )

    @classmethod
    def calculate_kpis(cls, rollout_results: Dict) -> Dict[str, float]:
        """Calculate KPIs for each rollout, using the output of a rollout from RolloutWrapper"""
        service_level = (
            rollout_results["info"]["demand"] - rollout_results["info"]["shortage"]
        ).sum(axis=-1) / rollout_results["info"]["demand"].sum(axis=-1)

        expiries = rollout_results["info"]["expiries"].sum(axis=-1) / rollout_results[
            "action"
        ].sum(axis=(-1))
        slippage = rollout_results["info"]["slippage"].sum(axis=-1) / rollout_results[
            "action"
        ].sum(axis=(-1))

        holding_units = rollout_results["info"]["holding"].mean(axis=-1)
        demand = rollout_results["info"]["demand"].mean(axis=-1)
        order_q = rollout_results["action"].mean(axis=-1)
        order_made = (rollout_results["action"] > 0).mean(axis=-1)

        return {
            "service_level_%": service_level * 100,
            "expiries_%": expiries * 100,
            "slippage_%": slippage * 100,
            "holding_units": holding_units,
            "demand": demand,
            "order_quantity": order_q,
            "shortage_units": rollout_results["info"]["shortage"].mean(axis=-1),
            "expiries_units": rollout_results["info"]["expiries"].mean(axis=-1),
            "slippage_units": rollout_results["info"]["slippage"].mean(axis=-1),
            "order_made_%": order_made,
        }
