''' Code to run a number of gym environment instances, in series, in python functions for use
    within evotask. '''

import string
import types

import numpy as np
import tensorflow as tf

class GymEnv:
    def __init__(self, evolution, gym_env, discrete_as_probs=True, frames_per_render=1):
        self.ev = evolution
        if type(gym_env) == str:
            gym_env = [gym_env]
        assert (self.ev.n_tasks == len(gym_env))
        import gym
        self.env = [[[gym.make(name) for c in range(self.ev.n_conditions)] for name in gym_env] for i in range(self.ev.n_individuals)]
        self.env = np.array(self.env)
        self.low_action = np.array([[[cond.action_space.low for cond in task] for task in ind] for ind in self.env])
        self.high_action = np.array([[[cond.action_space.high for cond in task] for task in ind] for ind in self.env])
        self.action_range = self.high_action - self.low_action
        self.reseed()

        self._reset = np.vectorize(lambda env: env.reset(), signature='()->(n)')
        def single_env_step(env, action, last_obs, last_T):
            if not last_T:
                obs, dR, T, _ = env.step(action)
                return (obs.flatten(), dR, T)
            else:
                return (last_obs, 0., last_T)
        self._step = np.vectorize(single_env_step, signature='(),(a),(n),()->(n),(),()')

        self.obs_shape = self.env[0][0][0].observation_space.shape
        self.obs_dim = 1
        for i in self.obs_shape: self.obs_dim *= i
        #if discrete_as_probs:
        #    self.action_shape =
        self.action_shape = self.env[0][0][0].action_space.shape
        self.actions_shape = self.env.shape + self.action_shape
        self.action_dim = 1
        for i in self.action_shape: self.action_dim *= i

        self.frames_per_render = frames_per_render
        self.render_frame_counter = 0

        print(self.obs_dim, self.action_dim)

    def reseed(self):
        for ind in self.env:
            for task in ind:
                for i, cond in enumerate(task):
                    cond.seed(self.ev.seed+i*3977)

    def generate_parameters(self):
        ''' Initializes parameters; can be nested tuples/lists of any shape or form.
            Each parameter must have axis 0 align with the task.
            Uses self.ev.genotype_random as the source for any randomness. '''
        return tf.zeros((self.ev.n_tasks,1))

    @tf.function
    def init(self, parameters):
        tf.py_function(self.reseed, [], [])
        return self.reset(parameters)

    @tf.function
    def reset(self, parameters, H=None):
        def to_reset():
            return self._reset(self.env)
        obs, = tf.py_function(to_reset, [], [self.env[0][0][0].observation_space.dtype])
        obs = tf.reshape(obs, (self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.obs_dim))
        dR = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions), tf.bool)

        return (obs, T), obs, dR, T

    @tf.function
    def step(self, parameters, H, A):
        A = (tf.reshape(A, self.actions_shape) + 1) / 2
        action = self.low_action + A * self.action_range

        last_obs, last_T = H

        def to_step(action, last_obs, last_T):
            return self._step(self.env, action, last_obs, last_T)

        obs, dR, T = tf.py_function(to_step, [action, last_obs, last_T], [self.env[0][0][0].observation_space.dtype, tf.float32, tf.bool])
        obs = tf.reshape(obs, (self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.obs_dim))
        dR = tf.reshape(dR, (self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))
        T = tf.reshape(T, (self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))

        return (obs, T), obs, dR, T

    def to_render(self, target):
        self.render_frame_counter += 1
        if self.render_frame_counter % self.frames_per_render == 0:
            self.env[target].render()

    @tf.function
    def render(self, parameters, H, A, target, prefix=''):
        if target is not None:
            tf.py_function(self.to_render, [target], [])

class HumanAgent:
    ''' An agent that uses the mouse to control the first two dimensions of the action. '''
    USES_VARS = False

    def __init__(self, evolution, action_dim):
        assert action_dim in (1, 2)
        self.action_dim = action_dim

        self.ev = evolution

        import pyglet

        self.window = pyglet.window.Window(width=500, height=500)
        self.x = self.y = 0
        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.x = (x-250) / 250
            self.y = (y-250) / 250

    def generate_parameters(self):
        return (self.ev.genotype_random.truncated_normal((self.ev.n_individuals, 1))*self.init_noise,)

    def use_parameters(self, parameters):
        self.action_constant, = parameters

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)
        H = tuple()
        return H

    def get_action(self):
        self.window.switch_to()
        self.window.dispatch_events()
        return self.x, self.y

    @tf.function
    def step(self, parameters, H, S, R, T, training):
        self.use_parameters(parameters)

        x, y = tf.py_function(self.get_action, [], [tf.float32, tf.float32])

        if self.action_dim == 1:
            A = tf.stack([y], axis=0)
        else:
            A = tf.stack([x, y], axis=0)

        A_by_tasks_conds = tf.reshape(A, (1, 1, 1, self.action_dim))
        A_copied = tf.tile(A_by_tasks_conds, [self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, 1])
        A = tf.clip_by_value(A_copied, -1, 1)
        return H, A

    def set_vars(self, state):
        ''' Sets the state variables based on nested list of arrays state. '''
        pass

    def get_vars(self):
        ''' Returns a nested list of tensors that contains the current state variables. '''
        pass
