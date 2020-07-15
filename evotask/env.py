''' Code to simulate various environments, each of which might have parameters describing
    a number of task variants. '''

import string

import numpy as np
import tensorflow as tf

class NullEnv:
    ''' A model of the most simple environment that could exist. '''
    def __init__(self, evolution, obs_dim, action_dim):
        ''' Receives a reference to the evolution class and any keyword arguments,
            and saves them for future use. '''

        self.ev = evolution
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def generate_parameters(self):
        ''' Initializes parameters; can be nested tuples/lists of any shape or form.
            Each parameter must have axis 0 align with the task index.
            Uses self.ev.genotype_random as the source for any randomness. '''
        return tf.zeros((self.ev.n_tasks,1))

    def use_parameters(self, parameters):
        ''' Receives a list of parameters and saves them as instance variables for use
            elsewhere. Each parameter will have axis 0 align with the task index.
            All tuples will have been transformed into lists. '''
        pass

    @tf.function
    def init(self, parameters):
        ''' Initializes a generation and starts the first episode.
            Returns the following as a tuple:
            - The environment's internal state (nested lists of tensors) -- uses
              self.ev.rand for any randomness needed.
            - An observation for the agent (a single float tensor with dimension
              self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions x obs_dim)
            - The reward for this timestep (a single float tensor, practically always 0,
              with dimension self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions)
            - The terminal state at this timestep (a single bool tensor, practically always False,
              with dimension self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions) '''

        self.use_parameters(parameters)

        H = tuple()
        obs = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.obs_dim))
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), dtype=tf.bool)

        return H, obs, dR, T


    @tf.function
    def reset(self, parameters, H, training=True):
        ''' Starts a new episode, possibly using an index from the environment's internal state.
            When initial conditions are uncorrelated, may directly call self.init(parameters).
            Receives optional boolean value training to determine whether to start using a training
            or test condition.
            Returns the following as a tuple:
            - The environment's internal state (nested lists of tensors) -- uses
              self.ev.rand for any randomness needed.
            - An observation for the agent (a single float tensor with dimension
              self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions x obs_dim)
            - The reward for this timestep (a single float tensor, practically always 0,
              with dimension self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions)
            - The terminal state at this timestep (a single bool tensor, practically always False,
              with dimension self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions) '''

        return self.init(parameters)

    @tf.function
    def step(self, parameters, H, A):
        ''' Performs a single time-step.
            In addition to the parameters, receives:
            - The environment's previous internal state (nested lists of tensors)
            - The agent's most recent action (a single float tensor with dimension
              self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions x action_dim)
            Returns the following as a tuple:
            - The environment's new internal state (nested lists of tensors)
            - An observation for the agent (a single float tensor with dimension
              self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions x obs_dim)
            - The reward for this timestep (a single float tensor representing the change in reward,
              with dimension self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions)
            - The terminal state at this timestep (a single bool tensor True when terminal,
              with dimension self.ev.n_individuals x self.ev.n_tasks x self.ev.n_conditions) '''

        self.use_parameters(parameters)

        nextH = H
        obs = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.obs_dim))
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), dtype=tf.bool)

        return H, obs, dR, T

    def _render(self, target):
        ''' This is an intentionally meaningless function. '''
        print(target)

    @tf.function
    def render(self, parameters, H, A, target, prefix=None):
        ''' Renders the trajectory denoted by target (specifies individual, task, and condition).
            Optionally saves the render under prefix, usually {prefix}-{timestep}.png.
            Typically calls a Python function using tf.py_function. '''

        self.use_parameters(parameters)

        # the second [] is the expected return from the function
        tf.py_function(self._render, [target], [])

class LandscapeEnv:
    ''' A simple static landscape evaluator of the action. Optionally returns the gradient of the landscape as an observation.
        Otherwise, returns zeros in the dimension of the landscape as an observation.
        module.evaluate should be a tf.function taking a variable of dimension dim, a parameter (such as K), and a seed,
        and returning a fitness score (ideally in the range [0, 1]). '''
    def __init__(self, evolution, module, dim, parameter, want_gradient=False):
        self.ev = evolution
        self.landscape_module = module
        self.landscape_parameter = parameter
        self.landscape_dim = dim
        self.landscape_seed = self.ev.seed
        self.want_gradient = False

        # perform any caching behavior necessary
        self.landscape_module.evaluate(tf.zeros((1, self.landscape_dim)), self.landscape_parameter, self.landscape_seed)

    def generate_parameters(self):
        return tf.zeros((self.ev.n_tasks,1))

    def use_parameters(self, parameters):
        pass

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)

        H = tuple()
        obs = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.landscape_dim))
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), dtype=tf.bool)

        return H, obs, dR, T

    @tf.function
    def reset(self, parameters, H, training=True):
        return self.init(parameters)

    @tf.function
    def step(self, parameters, H, A):
        self.use_parameters(parameters)

        nextH = H
        shaped_A = tf.reshape(A, (-1, self.landscape_dim))
        scaled_A = (shaped_A + 1) / 2 # [-1, 1] => [0, 1]

        score = self.landscape_module.evaluate(scaled_A, self.landscape_parameter, self.landscape_seed)

        dR = tf.reshape(score, (self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), dtype=tf.bool)

        if self.want_gradient:
            score_gradient = tf.gradients([tf.reduce_sum(score)], [shaped_A])[0]
            obs = tf.reshape(score_gradient, (self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.landscape_dim))
        else:
            obs = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.landscape_dim))

        return H, obs, dR, T

    @tf.function
    def render(self, parameters, H, A, target, prefix=None):
        pass

class PendulumEnv:
    ''' The inverted pendulum environment, based on OpenAI gym's implementation. '''
    def __init__(self, evolution, default_difficulty=1):
        ''' Difficulty refers to the spread of initial conditions.
            1 is the default and corresponds with the distribution used in other papers. '''
        self.ev = evolution
        self.dt = self.ev.dt
        self.default_difficulty = default_difficulty

        self.viewer = None

    def generate_parameters(self):
        target = tf.zeros((self.ev.n_tasks))
        difficulty = tf.ones((self.ev.n_tasks))*self.default_difficulty
        g = tf.ones((self.ev.n_tasks))*10
        m = tf.ones((self.ev.n_tasks))
        l = tf.ones((self.ev.n_tasks))
        return target, difficulty, g, m, l

    def use_parameters(self, parameters):
        target, difficulty, g, m, l = parameters
        # so that axes can be (individual, task)
        self.target = tf.expand_dims(target, 0)
        self.difficulty = tf.expand_dims(difficulty, 0)
        self.g = tf.expand_dims(g, 0)
        self.m = tf.expand_dims(m, 0)
        self.l = tf.expand_dims(l, 0)

    def _get_obs(self, theta, omega):
        s_th = tf.math.sin(theta)
        c_th = tf.math.cos(theta)

        return tf.stack((c_th, s_th, omega), axis=3)

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)

        theta_conditions = self.ev.random.uniform((self.ev.n_conditions,), -np.pi, np.pi)
        theta_tasks = tf.stack([theta_conditions]*self.ev.n_tasks, axis=0)
        theta = tf.stack([theta_tasks]*self.ev.n_individuals, axis=0)
        theta *= self.difficulty

        omega_conditions = self.ev.random.uniform((self.ev.n_conditions,), -1, 1)
        omega_tasks = tf.stack([omega_conditions]*self.ev.n_tasks, axis=0)
        omega = tf.stack([omega_tasks]*self.ev.n_individuals, axis=0)
        omega *= self.difficulty

        R = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))

        obs = self._get_obs(theta, omega)
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        H = (theta, omega, R)

        return H, obs, dR, T

    @tf.function
    def reset(self, parameters, H, training=True):
        return self.init(parameters)

    @tf.function
    def step(self, parameters, H, A):
        self.use_parameters(parameters)

        theta, omega, R = H

        torque = tf.squeeze(A, axis=-1) * 2

        theta = tf.where(theta-self.target < -np.pi, theta + 2 * np.pi, theta)
        theta = tf.where(theta-self.target >= np.pi, theta - 2 * np.pi, theta)

        cost = (theta-self.target)**2 + .1*omega**2 + .001*torque**2

        alpha = (-3*self.g/(2*self.l) * tf.math.sin(theta + np.pi) + 3./(self.m*self.l**2)*torque)
        omega += alpha * self.dt
        theta += omega * self.dt

        omega = tf.clip_by_value(omega, -8., 8.)

        obs = self._get_obs(theta, omega)

        dR = -cost*(self.dt/.05)
        R += dR

        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        nextH = theta, omega, R

        return nextH, obs, dR, T

    @tf.function
    def render(self, parameters, H, A, target, prefix=''):
        self.use_parameters(parameters)

        if self.viewer is None:
            from .render import SystemRenderer
            self.viewer = SystemRenderer(1, 1, 1)

        theta, omega, R = H

        tf.py_function(self.viewer.render, [A[target], tf.reshape(theta[target], (1,))+np.pi/2, tf.reshape(omega[target], (1,))/8, R[target], prefix], [])


class SystematicPendulumEnv(PendulumEnv):
    ''' A modified version of the Pendulum Environment designed to test on a constant,
        deterministic distribution of starting conditions for the test episodes. '''
    @tf.function
    def init(self, parameters):
        H, obs, dR, T = super().init(parameters)

        testing_episode_index = tf.constant(0, dtype=tf.int32)

        return (testing_episode_index,)+H, obs, dR, T

    @tf.function
    def reset(self, parameters, H, training=True):
        if training:
            return self.init(parameters)

        self.use_parameters(parameters)
        testing_episode_index = H[0]

        # this math is not taking place inside of tensorflow.
        # these will multiply perfectly so long as testing_episodes is an even square;
        # otherwise, the most ccw starting angular velocity
        # will be biased towards cw starting angle
        n_thetas = np.floor(np.sqrt(self.ev.testing_episodes)*2)
        n_omegas = np.ceil(self.ev.testing_episodes/n_thetas)

        # back to tensorflow
        n_thetas = tf.constant(n_thetas, dtype=tf.int32)
        n_omegas = tf.constant(n_omegas, dtype=tf.int32)
        omega_idx = testing_episode_index // n_thetas
        theta_idx = testing_episode_index % n_thetas

        omega_root = omega_idx / (n_omegas-1) * 2 - 1
        omega_root = tf.cast(omega_root, tf.float32)
        omega_conditions = tf.stack([omega_root]*self.ev.n_conditions, axis=0)
        omega_tasks = tf.stack([omega_conditions]*self.ev.n_tasks, axis=0)
        omega = tf.stack([omega_tasks]*self.ev.n_individuals, axis=0)
        omega *= self.difficulty

        theta_root = theta_idx / (n_thetas-1) * (2 * np.pi) - np.pi
        theta_root = tf.cast(theta_root, tf.float32)
        theta_conditions = tf.stack([theta_root]*self.ev.n_conditions, axis=0)
        theta_tasks = tf.stack([theta_conditions]*self.ev.n_tasks, axis=0)
        theta = tf.stack([theta_tasks]*self.ev.n_individuals, axis=0)
        theta *= self.difficulty

        R = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))

        obs = self._get_obs(theta, omega)
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        H = (testing_episode_index+1, theta, omega, R)

        return H, obs, dR, T

    @tf.function
    def step(self, parameters, H, A):
        testing_episode_index, *_H = H
        nextH, obs, dR, T = super().step(parameters, _H, A)

        return (testing_episode_index,)+nextH, obs, dR, T

    @tf.function
    def render(self, parameters, H, A, target, prefix=''):
        testing_episode_index, *_H = H
        return super().render(parameters, _H, A, target, prefix)

class OneLeggerEnv:
    ''' An environment simulating a one-legged walking agent.
        Written by Madhavun Candadai. '''
    def __init__(self, evolution):
        self.ev = evolution
        self.dt = self.ev.dt
        self.obs_dim = 3
        self.action_dim = 2

        self.viewer = None

    def generate_parameters(self):
        leg_len = tf.ones((self.ev.n_tasks))*15
        max_force = tf.ones((self.ev.n_tasks))*0.05
        max_fwd_angle = tf.ones((self.ev.n_tasks))*(np.pi/6)
        max_bwd_angle = tf.ones((self.ev.n_tasks))*(-np.pi/6)
        max_vel = tf.ones((self.ev.n_tasks))*6
        max_tque = tf.ones((self.ev.n_tasks))*0.5
        max_omega = tf.ones((self.ev.n_tasks))*1.0
        return leg_len, max_force, max_fwd_angle, max_bwd_angle, max_vel, max_tque, max_omega

    def use_parameters(self, parameters):
        leg_len, max_force, max_fwd_angle, max_bwd_angle, max_vel, max_tque, max_omega = parameters
        self.leg_len = tf.expand_dims(leg_len, 0)
        self.max_force = tf.expand_dims(max_force, 0)
        self.max_fwd_angle = tf.expand_dims(max_fwd_angle, 0)
        self.max_bwd_angle = tf.expand_dims(max_bwd_angle, 0)
        self.max_vel = tf.expand_dims(max_vel, 0)
        self.max_tque = tf.expand_dims(max_tque, 0)
        self.max_omega = tf.expand_dims(max_omega, 0)

    def _get_obs(self, angle, omega, foot_state):
        return tf.stack((angle, omega, tf.cast(foot_state,tf.float32)), axis=3)

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)

        # insect position
        cx = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        cy = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        vx = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))

        # leg joint states
        joint_X = tf.identity(cx) # copy
        joint_Y = cy + 12.5
        # stack up so each individual / task has the same set of conditions
        angle = self.ev.random.uniform((self.ev.n_conditions,), 0., 1.)
        angle = tf.stack([angle]*self.ev.n_tasks,axis=0)
        angle = tf.stack([angle]*self.ev.n_individuals,axis=0)
        angle = angle * (self.max_fwd_angle - self.max_bwd_angle) + self.max_bwd_angle
        omega = self.ev.random.uniform((self.ev.n_conditions,), -1., 1.)
        omega = tf.stack([omega]*self.ev.n_tasks,axis=0)
        omega = tf.stack([omega]*self.ev.n_individuals,axis=0)
        omega *= self.max_omega

        # foot position
        foot_X = joint_X + self.leg_len * tf.sin(angle)
        foot_Y = joint_Y + self.leg_len * tf.cos(angle)
        foot_state = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        obs = self._get_obs(angle, omega, foot_state)
        R = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        H = (cx, cy, vx, joint_X, joint_Y, angle, omega, foot_X, foot_Y, foot_state, R)

        return H, obs, dR, T

    @tf.function
    def reset(self, parameters, H, training=True):
        return self.init(parameters)

    @tf.function
    def step(self, parameters, H, A):
        self.use_parameters(parameters)

        cx, cy, vx, joint_X, joint_Y, angle, omega, foot_X, foot_Y, foot_state, R = H
        old_cx = tf.identity(cx)

        # update leg effector - is foot on ground?
        foot_state = (A[:,:,:,0] > 0) # simpler than foot_state = tf.where(A[:,:,:,0] > 0, tf.ones_like(foot_state), tf.zeros_like(foot_state))

        # estimating force
        force = A[:,:,:,1]*self.max_force
        # added foot_state to where logic instead of giving it its own multiply
        force = tf.where(
            (((angle >= self.max_bwd_angle) & (angle <= self.max_fwd_angle))
            | ((angle < self.max_bwd_angle) & (force < 0))
            | ((angle > self.max_fwd_angle) & (force > 0)))
            & foot_state,
            force,
            0.0,
        ) # need to use bitwise symbols

        # Update position of body
        vx = vx + self.dt * force
        vx = tf.clip_by_value(vx, -self.max_vel, self.max_vel)
        cx = cx + self.dt * vx

        ## Update leg geometry
        joint_X = joint_X + self.dt * vx

        # if foot is on
        tmp_foot_on_angle = tf.atan2(foot_X - joint_X, foot_Y - joint_Y)
        foot_on_omega = (tmp_foot_on_angle - angle)/self.dt
        foot_on_angle = tmp_foot_on_angle

        # if foot is off
        foot_off_omega = omega + self.dt * self.max_tque * A[:,:,:,1]
        foot_off_omega = tf.clip_by_value(foot_off_omega, -self.max_omega, self.max_omega)
        foot_off_angle = angle + self.dt * foot_off_omega
        foot_off_omega = tf.where((foot_off_angle < self.max_bwd_angle) | (foot_off_angle > self.max_fwd_angle), 0., foot_off_omega) # 0. for type consistency
        foot_off_angle = tf.clip_by_value(foot_off_angle, self.max_bwd_angle, self.max_fwd_angle)

        # Final updates to leg geometry
        # foot_state is a bool so it can be a direct condition
        angle = tf.where(foot_state, foot_on_angle, foot_off_angle)
        omega = tf.where(foot_state, foot_on_omega, foot_off_omega)
        foot_X = tf.where(foot_state, foot_X, joint_X + self.leg_len * tf.sin(angle))
        foot_Y = tf.where(foot_state, foot_Y, joint_Y + self.leg_len * tf.cos(angle))

        # If the foot is too far back, the body becomes "unstable" and forward motion ceases
        # If the foot is "off", vx is also set to 0.
        vx = tf.where((cx - foot_X > 20) | ~foot_state, 0.0, vx)

        # reward is distance moved, total fitness should be normalized by duration
        dR = cx - old_cx
        R += dR

        # there is no terminal state?
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        obs = self._get_obs(angle, omega, foot_state)

        nextH = (cx, cy, vx, joint_X, joint_Y, angle, omega, foot_X, foot_Y, foot_state, R)

        return nextH, obs, dR, T

    def to_render(self, *args):
        if self.viewer is None:
            from .render import SystemRenderer
            self.viewer = SystemRenderer(2, 1, 4)
        return self.viewer.render(*args)

    @tf.function
    def render(self, parameters, H, A, target, prefix=''):
        self.use_parameters(parameters)

        cx, cy, vx, joint_X, joint_Y, angle, omega, foot_X, foot_Y, foot_state, R = H
        # angle=0 denotes down;
        # cy and joint_Y are fixed; joint_X=cx; foot_X and foot_Y are determined by angle and joint_X and joint_Y;
        # foot state denotes whether the foot is on the ground
        biconstrained_vars = tf.stack([omega/self.max_omega, cx, vx/self.max_vel, tf.cast(foot_state,tf.float32)], axis=-1)

        tf.py_function(self.to_render, [A[target], tf.reshape(angle[target] - np.pi/2, (1,)), biconstrained_vars[target], R[target], prefix], [])
