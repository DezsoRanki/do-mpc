"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

import do_mpc
from casadi import *

import matplotlib.pyplot as plt
import matplotlib as mpl


class PathFollowingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # model
        self.model_type = 'continuous' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(self.model_type)

        # states
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
        self.x_path = self.model.set_variable(var_type='_x', var_name='x_path', shape=(1,1))
        self.y_path = self.model.set_variable(var_type='_x', var_name='y_path', shape=(1,1))
        self.phi_path = self.model.set_variable(var_type='_x', var_name='phi_path', shape=(1,1))
        self.x = self.model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        self.y = self.model.set_variable(var_type='_x', var_name='y', shape=(1,1))
        self.phi = self.model.set_variable(var_type='_x', var_name='phi', shape=(1,1))
        self.v = self.model.set_variable(var_type='_x', var_name='v', shape=(1,1))
        self.v_ref = self.model.set_variable(var_type='_x', var_name='v_ref', shape=(1,1))
        self.delta_s = self.model.set_variable(var_type='_x', var_name='delta_s', shape=(1,1))
        self.delta_s_ref = self.model.set_variable(var_type='_x', var_name='delta_s_ref', shape=(1,1))

        # inputs
        self._theta = self.model.set_variable(var_type='_u', var_name='_theta')
        self.a_ref = self.model.set_variable(var_type='_u', var_name='a_ref')
        self.w_s_ref = self.model.set_variable(var_type='_u', var_name='w_s_ref')

        # parameters
        self.k0 = -0.44
        self.k1 = 928
        self.k2 = -2.39
        self.lr = 1.51
        self.lf = 1.13
        self.T_v = 1.2
        self.T_delta = 0.15
        self._theta_ref = 10

        self.delta_a = (self.delta_s - self.k0) / (self.k1 + self.k2 * self.v)
        self.beta = np.arctan((self.lr / (self.lf + self.lr)) * np.tan(self.delta_a))

        # states at which to fail the episode
        # terminate at simulate time
        self.x_threshold = 100
        self.y_threshold = 100
        self.phi_threshold = 90


        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # observation
        high = np.array(
            [
                self.x_threshold * 2,
                self.y_threshold * 2,
                self.phi_threshold * 2,
            ],
            dtype=np.float32,
        )
        self.min_action = 1.0
        self.max_action = 10e5

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x0 = self.state
        _path_error = self.path_error

        # dynamics
        self.model.set_rhs('theta', self.theta)
        self.model.set_rhs('x_path', self.theta)
        self.model.set_rhs('y_path', 2 * self.theta * self.theta)
        self.model.set_rhs('phi_path', (1 / (1 + 4 * self.theta * self.theta)) * 2 * self.theta * self.theta)
        # model.set_rhs('theta', _theta)
        # model.set_rhs('x_path', (1 + 2 * theta + 3 * theta * theta) * _theta) # x=theta+theta^2+theta^3
        # model.set_rhs('y_path', (1 - 2 * theta + 3 * theta * theta - 5 * theta * theta * theta * theta) * _theta)
        # model.set_rhs('phi_path', np.arctan(((1 - 2 * theta + 6 * theta * theta - 5 * theta * theta * theta * theta) * _theta) / ((1 + 2 * theta + 3 * theta * theta) * _theta)))
        self.model.set_rhs('x', self.v * np.cos(self.phi + self.beta))
        self.model.set_rhs('y', self.v * np.sin(self.phi + self.beta))
        self.model.set_rhs('phi', (self.v / self.lr) * np.sin(self.beta))
        self.model.set_rhs('v', 1 / self.T_v * (self.v_ref - self.v))
        self.model.set_rhs('v_ref', self.a_ref)
        self.model.set_rhs('delta_s', 1 / self.T_delta * (self.delta_s_ref - self.delta_s))
        self.model.set_rhs('delta_s_ref', self.w_s_ref)

        self.h1 = self.x + self.lf * np.cos(self.phi)
        self.h2 = self.y + self.lf * np.sin(self.phi)
        self.h3 = self.phi

        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)
        self.setup_mpc = {
            'n_horizon': 30,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        self.mpc.set_param(**self.setup_mpc)

        # object function
        self.p_x = 2.5e4
        self.p_y = 2.5e4
        self.q_x = 2.5e4
        self.q_y = 2.5e4
        self.q_phi = 2700
        self.p_phi = 2700
        self.r_a = 700
        self.r_w = 0.2
        self.r_theta = 1000
        self.p_a = action
        self.q_a = action        

        self.mterm = self.p_x * ((self.x + self.lf * np.cos(self.phi) - self.x_path) ** 2) + self.p_y * ((self.y + self.lf * np.sin(self.phi) - self.y_path) ** 2) + self.p_phi * ((self.phi - self.phi_path) ** 2) + \
                self.p_a * ((self.v * ((self.v / self.lr) * np.sin(np.arctan((self.lr / (self.lf + self.lr)) * np.tan((self.delta_s - self.k0) / (self.k1 + self.k2 * self.v)))))) ** 2)
        self.lterm = self.q_x * ((self.x + self.lf * np.cos(self.phi) - self.x_path) ** 2) + self.q_y * ((self.y + self.lf * np.sin(self.phi) - self.y_path) ** 2) + self.q_phi * ((self.phi - self.phi_path) ** 2) + \
                self.q_a * ((self.v * ((self.v / self.lr) * np.sin(np.arctan((self.lr / (self.lf + self.lr)) * np.tan((self.delta_s - self.k0) / (self.k1 + self.k2 * self.v)))))) ** 2) + \
                self.r_a * (self.a_ref ** 2) + self.r_w * (self.w_s_ref ** 2) + self.r_theta * ((self._theta - self._theta_ref) ** 2)

        self.mpc.set_objective(mterm=self.mterm, lterm=self.lterm)

        # constraints
        self.mpc.bounds['lower','_x', 'v_ref'] = 0
        self.mpc.bounds['lower','_x', 'delta_s_ref'] = -460
        self.mpc.bounds['lower','_x', 'theta'] = 0

        self.mpc.bounds['upper','_x', 'v_ref'] = 60
        self.mpc.bounds['upper','_x', 'delta_s_ref'] = 460
        self.mpc.bounds['upper','_x', 'theta'] = 60

        self.mpc.bounds['lower','_u', 'a_ref'] = -3
        self.mpc.bounds['lower','_u', 'w_s_ref'] = -250
        self.mpc.bounds['lower','_u', '_theta'] = 0

        self.mpc.bounds['upper','_u', 'a_ref'] = 2
        self.mpc.bounds['upper','_u', 'w_s_ref'] = 250
        self.mpc.bounds['upper','_u', '_theta'] = 60

        self.mpc.setup()

        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step = 0.1)

        self.simulator.setup()

        x0 = np.array([1, 1, 1, np.arctan(2), 1, 1, np.arctan(2), 1, 1, 1, 1]).reshape(-1,1)

        self.simulator.x0 = x0
        self.mpc.x0 = x0

        self.mpc.set_initial_guess()

        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        mpc_graphics = do_mpc.graphics.Graphics(self.mpc.data)
        sim_graphics = do_mpc.graphics.Graphics(self.simulator.data)

        self.simulator.reset_history()
        self.simulator.x0 = x0
        self.mpc.reset_history()

        for i in range(50):
            u0 = self.mpc.make_step(x0)
            x0 = self.simulator.make_step(u0)

        self.state = x0

        self.path_error = (x0[1][0] - x0[4][0]) ** 2 + (x0[2][0] - x0[5][0]) ** 2 + (x0[2][0] - x0[6][0]) ** 2

        fig, ax = plt.subplots(1, sharex=True, figsize=(16,9))
        ax.plot(sim_graphics.data['_x', 'x_path'],sim_graphics.data['_x', 'y_path'])
        ax.plot(sim_graphics.data['_x', 'x'],sim_graphics.data['_x', 'y'])
        plt.show()

        terminated = bool(
            x0[1][0] < -self.x_threshold
            or x0[1][0] > self.x_threshold
            or x0[2][0] < -self.y_threshold
            or x0[2][0] > self.y_threshold
            or x0[3][0] < -self.phi_threshold
            or x0[3][0] > self.phi_threshold
        )


        # reward
        if self.path_error < _path_error:
            reward = 1.0
        else:
            reward = 0.0
        # if not terminated:
        #     reward = 1.0
        # elif self.steps_beyond_terminated is None:
        #     # Pole just fell!
        #     self.steps_beyond_terminated = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_terminated == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned terminated = True. You "
        #             "should always call 'reset()' once you receive 'terminated = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_terminated += 1
        #     reward = 0.0

        if self.render_mode == "human":
            self.render()
        return self.state, self.path_error, reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = np.array([1, 1, 1, np.arctan(2), 1, 1, np.arctan(2), 1, 1, 1, 1]).reshape(-1,1)
        self.path_error = inf
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

