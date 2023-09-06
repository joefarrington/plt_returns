from typing import Tuple, Optional, Union, Dict, List
import chex

import jax
import jax.numpy as jnp
from jax import lax

from gymnax.environments import environment, spaces

from flax import struct
import numpyro
import distrax

from omegaconf import OmegaConf


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
        initial_weekday: int = -1,  # Randomly select initial weekday each episode
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
        return_probability: float = 0.0,
        return_prediction_model_sensitivity: float = 1,
        return_prediction_model_specificity: float = 1,
        gamma: float = 1.0,
    ):
        if OmegaConf.is_list(age_on_arrival_distributions):
            age_on_arrival_distributions = OmegaConf.to_container(
                age_on_arrival_distributions, resolve=True
            )
        age_on_arrival_distributions_jax = EnvParams.age_on_arrival_to_jax_array(
            age_on_arrival_distributions
        )

        return EnvParams(
            jnp.array(poisson_mean_demand_pre_return),
            jnp.array(poisson_mean_demand_post_return),
            slippage,
            initial_weekday,
            jnp.array(initial_stock),
            age_on_arrival_distributions_jax,
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

    @classmethod
    def age_on_arrival_to_jax_array(
        cls, age_on_arrival_distributions: List[float]
    ) -> chex.Array:
        """Convert age_on_arrival_distributions argument to jax array with one row for each
        day of the week. Accept a single list (which is repeated for each weekday), or a list of 7 lists)
        """
        if isinstance(age_on_arrival_distributions, list):
            # if the first element is also a list, then we assume it's a list of lists
            if isinstance(age_on_arrival_distributions[0], list):
                if len(age_on_arrival_distributions) == 7:
                    return jnp.array(
                        [x for x in age_on_arrival_distributions]
                    )  # TODO: Change when configs changed
                else:
                    raise ValueError(
                        "Expected a list of 7 lists. Got a list of {} lists.".format(
                            len(age_on_arrival_distributions)
                        )
                    )
            else:
                # if it's a single list, repeat it seven times
                return jnp.array(
                    [age_on_arrival_distributions] * 7
                )  # TODO Change when configs changed
        else:
            raise TypeError(
                "Expected a list or a list of lists. Got {}.".format(
                    type(age_on_arrival_distributions)
                )
            )


class PlateletBankGymnax(environment.Environment):
    # We need to pass in max_useful_life because it affects array shapes
    # We need to pass in max_order_quantity because self.num_actions depends on it
    # We need to pass in max_demand because it affects array shapes
    def __init__(
        self,
        max_useful_life: int = 3,
        max_order_quantity: int = 100,
        max_demand: int = 100,
        return_detailed_info: bool = False,
    ):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.max_order_quantity = max_order_quantity
        self.max_demand = max_demand  # Max demand applies separately to both am and pm, should be set quite high; sampled demand clipped using this value
        if return_detailed_info:
            self._get_detailed_info_fn = self._get_detailed_info
        else:
            self._get_detailed_info_fn = lambda _, *args, **kwargs: {}

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
        # At this point we find out how many units expired while out on the wards
        returned_units = state.to_be_returned
        expiries_from_returned = returned_units[self.max_useful_life - 1]
        slippage_units = numpyro.distributions.Binomial(
            total_count=returned_units[0 : self.max_useful_life - 1],
            probs=params.slippage,
        ).sample(key=slippage_key)
        back_in_stock_from_returned = jnp.hstack(
            [returned_units[0 : self.max_useful_life - 1] - slippage_units, 0]
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
        stock_expiries = closing_stock[self.max_useful_life - 1]
        expiries = stock_expiries + expiries_from_returned
        slippage = slippage_units.sum()
        closing_stock = jnp.hstack([0, closing_stock[0 : self.max_useful_life - 1]])

        # We don't age stock in to be returned, because expiries for those units
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

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # By default, we start on a random weekday
        # Otherwise, with fixed burn-in, would always
        # count return from same weekday
        weekday = jax.lax.cond(
            params.initial_weekday == -1,
            lambda _: jax.random.randint(key, (), 0, 7, dtype=jnp.int32),
            lambda _: params.initial_weekday.astype(jnp.int32),
            None,
        )

        state = EnvState(
            stock=params.initial_stock,
            to_be_returned=jnp.zeros_like(params.initial_stock),
            weekday=weekday,
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
        stock_expiries: int,
        expiries_from_returned: int,
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
                # Expiries from returned included one timestep
                # later than they would be if in stock, so
                # adjust for discount factor
                (1 / params.gamma) * params.expiry_cost,
                params.slippage_cost,
            ]
        )
        values = jnp.array(
            [
                jnp.where(action > 0, 1, 0),
                action,
                jnp.sum(closing_stock),
                backorders,
                stock_expiries,
                expiries_from_returned,
                slippage,
            ]
        )
        return jnp.dot(costs, values)

    def _fill_demand(
        self, initial_demand_info: DemandInfo
    ) -> Tuple[chex.Array, int, chex.Array, chex.Array, chex.Array]:
        """Fill the demand, and determine the backorders due to shortage and the units to be returned"""
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

    def _remaining_demand_and_stock(self, demand_info: DemandInfo) -> bool:
        """Determine whether to continue filled demand.
        Only continue if there is both remaining demand to fill and stock to fill it"""
        return (demand_info.remaining_demand > 0) & (
            demand_info.remaining_stock.sum() > 0
        )

    def _issue_one_unit(self, demand_info: DemandInfo) -> DemandInfo:
        """Issue a single unit following the issuing policy"""
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

    def _remaining_demand(self, demand_info: DemandInfo) -> bool:
        """Check if there is demand remaining to be filled"""
        return demand_info.remaining_demand > 0

    def _emergency_order_and_issue_one_unit(
        self, demand_info: DemandInfo
    ) -> DemandInfo:
        """Place an emergency order if there is a shortage"""
        idx = demand_info.total_demand - demand_info.remaining_demand
        remaining_demand = demand_info.remaining_demand - 1
        carry_key, order_key = jax.random.split(demand_info.key, 2)
        # Same age of unit from age at arrival distribution
        issued = distrax.Multinomial(
            total_count=1, probs=demand_info.age_on_arrival_distribution
        ).sample(seed=order_key)
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

    def _yufo(self, remaining_stock: chex.Array) -> int:
        """Determine the index of the youngest unit in the remaining stock"""
        return jnp.where(remaining_stock > 0, True, False).argmax()

    def _oufo(self, remaining_stock: chex.Array) -> int:
        """Determine the index of the oldest unit in the remaining stock"""
        return (self.max_useful_life - 1) - jnp.where(remaining_stock > 0, True, False)[
            ::-1
        ].argmax()

    @property
    def name(self) -> str:
        """Environment name."""
        return "PlateletBank"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.max_order_quantity + 1

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Discrete(self.max_order_quantity + 1)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # [weekday, oldest_stock, ..., freshest_stock]
        max_stock_by_age = self.max_useful_life * self.max_order_quantity
        low = jnp.array([0] * self.max_useful_life)
        high = jnp.array([6] + [max_stock_by_age] * (self.max_useful_life))
        return spaces.Box(low, high, (self.max_useful_life + 1,), dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        max_stock_by_age = self.max_useful_life * self.max_order_quantity
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

    def _build_confusion_matrix(
        self, demand: int, return_samples: chex.Array, return_pred_samples: chex.Array
    ) -> chex.Array:
        """Build confusion matrix for return prediction model, return in info for testing purposes"""
        cm = jnp.zeros((2, 2))

        def update_cm(i, cm):
            indices = (return_samples[i], return_pred_samples[i])
            updated_cm = jax.lax.dynamic_update_slice(
                cm, (cm[indices] + 1).reshape(1, 1), indices
            )
            return updated_cm

        cm = jax.lax.fori_loop(0, demand, update_cm, cm)
        return cm

    def _get_detailed_info(
        self,
        demand_am: int,
        return_samples_am: chex.Array,
        return_pred_samples_am: chex.Array,
        demand_pm: int,
        return_samples_pm: chex.Array,
        return_pred_samples_pm: chex.Array,
    ) -> Dict[str, chex.Array]:
        """Compute detailed info for testing purposes"""
        return {
            "cm_am": self._build_confusion_matrix(
                demand_am, return_samples_am, return_pred_samples_am
            ),
            "cm_pm": self._build_confusion_matrix(
                demand_pm, return_samples_pm, return_pred_samples_pm
            ),
        }

    @classmethod
    def calculate_kpis(cls, rollout_results: Dict) -> Dict[str, float]:
        """Calculate KPIs for each rollout, using the output of a rollout from RolloutWrapper"""
        service_level = (
            rollout_results["info"]["demand"] - rollout_results["info"]["shortage"]
        ).sum(axis=-1) / rollout_results["info"]["demand"].sum(axis=-1)
        # Denominator for expiries and slippage should include both routine and emergency orders
        # So we add in shortage

        total_received = rollout_results["action"].sum(axis=(-1)) + rollout_results[
            "info"
        ]["shortage"].sum(axis=-1)
        expiries = rollout_results["info"]["expiries"].sum(axis=-1) / total_received
        slippage = rollout_results["info"]["slippage"].sum(axis=-1) / total_received

        holding_units = rollout_results["info"]["holding"].mean(axis=-1)
        demand = rollout_results["info"]["demand"].mean(axis=-1)
        order_q = rollout_results["action"].mean(axis=-1)
        order_made = (rollout_results["action"] > 0).mean(axis=-1)

        return {
            "service_level_%": service_level * 100,
            "expiries_%": expiries * 100,
            "slippage_%": slippage * 100,
            "total_wastage_%": (expiries + slippage) * 100,
            "holding_units": holding_units,
            "demand": demand,
            "order_quantity": order_q,
            "shortage_units": rollout_results["info"]["shortage"].mean(axis=-1),
            "expiries_units": rollout_results["info"]["expiries"].mean(axis=-1),
            "slippage_units": rollout_results["info"]["slippage"].mean(axis=-1),
            "total_wastage_units": rollout_results["info"]["expiries"].mean(axis=-1)
            + rollout_results["info"]["slippage"].mean(axis=-1),
            "order_made_%": order_made,
        }
