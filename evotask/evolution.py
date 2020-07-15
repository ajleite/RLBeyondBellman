#!/usr/bin/python3

''' Code to handle the agent-environment interaction and call an evolutionary algorithm. '''

import tensorflow as tf
import numpy as np


def _expand_genotype(genotype, structure):
    if isinstance(structure, list):
        expanded = []
        for piece in structure:
            expanded_piece, genotype = _expand_genotype(genotype, piece)
            expanded.append(expanded_piece)
    else:
        size = 1
        for dim in structure[1:]:
            size *= dim
        expanded = tf.reshape(genotype[:, :size], structure)
        genotype = genotype[:, size:]
    return expanded, genotype

def expand_genotype(genotype, structure):
    expanded, _ = _expand_genotype(genotype, structure)
    return expanded

def to_genotype(parameters):
    if hasattr(parameters, 'shape'):
        return tf.reshape(parameters, (parameters.shape[0], -1))
    return tf.concat([to_genotype(subpar) for subpar in parameters], axis=1)

def to_genotype_structure(parameters):
    if hasattr(parameters, 'shape'):
        return tuple(parameters.shape)
    return [to_genotype_structure(subpar) for subpar in parameters]

def to_n_params(parameters):
    if hasattr(parameters, 'size'):
        return parameters.size
    return sum(to_n_params(subpar) for subpar in parameters)


class Evolution:
    def set_args(self, dimensions=(1,1,1), seed=0, dt=.05, episode_length=10, training_episodes=10, testing_episodes=10, cycles_per_generation=1, render_target=None, render_prefix=''):
        self.n_individuals, self.n_tasks, self.n_conditions = dimensions
        self.seed = seed
        self.dt = dt
        self.episode_length = episode_length
        self.training_episodes = training_episodes
        self.testing_episodes = testing_episodes
        self.n_episodes = training_episodes + testing_episodes
        self.cycles_per_generation = cycles_per_generation
        self.render_target = render_target
        self.render_prefix = render_prefix

    def __init__(self, evolution_args,
                 AgentClass, agent_args, agent_fitness, agent_strategy, agent_strategy_args,
                 EnvClass, env_args, env_fitness=None, env_strategy=None, env_strategy_args=None):
        self.set_args(**evolution_args)

        self.random = tf.random.experimental.Generator(tf.random.experimental.get_global_generator())
        self.random.reset_from_seed(self.seed)

        self.genotype_random = tf.random.experimental.Generator(tf.random.experimental.get_global_generator())
        self.genotype_random.reset_from_seed(self.seed+1234)

        self.agent = AgentClass(self, **agent_args)
        self.agent_fitness = agent_fitness
        self.agent_strategy = agent_strategy
        self.agent_strategy_args = agent_strategy_args

        self.env = EnvClass(self, **env_args)
        self.env_fitness = env_fitness
        self.env_strategy = env_strategy
        self.env_strategy_args = env_strategy_args

        self.initialize_parameters()

    def initialize_parameters(self):
        self.env_parameters = self.env.generate_parameters()
        self.agent_parameters = self.agent.generate_parameters()

    @property
    def env_parameters(self):
        return expand_genotype(self.env_genotype, self.env_p_structure)

    @env_parameters.setter
    def env_parameters(self, env_parameters):
        self.env_genotype = to_genotype(env_parameters)
        self.env_p_structure = to_genotype_structure(env_parameters)

    @property
    def agent_parameters(self):
        return expand_genotype(self.agent_genotype, self.agent_p_structure)

    @agent_parameters.setter
    def agent_parameters(self, agent_parameters):
        self.agent_genotype = to_genotype(agent_parameters)
        self.agent_p_structure = to_genotype_structure(agent_parameters)

    @tf.function
    def init(self, env_parameters, agent_parameters):
        env_init_H, init_obs, init_R, init_T = self.env.init(env_parameters)
        agent_init_H = self.agent.init(agent_parameters)

        init_reward = tf.zeros((self.n_individuals, self.n_tasks, self.n_conditions), tf.float32)
        init_t = tf.zeros(tuple(), tf.float32)

        init_state = (init_t, env_init_H, agent_init_H, init_reward, init_obs, init_R, init_T)

        return init_state

    @tf.function
    def new_episode(self, env_parameters, agent_parameters, state, training=True):
        t, env_H, agent_H, reward, obs, R, T = state

        t = tf.zeros(tuple(), tf.float32)
        T = tf.ones((self.n_individuals, self.n_tasks, self.n_conditions), tf.bool)

        if training:
            agent_R = R
        else:
            agent_R = tf.zeros_like(R)

        agent_H, A = self.agent.step(agent_parameters, agent_H, obs, agent_R, T, training)
        env_H, obs, R, T = self.env.reset(env_parameters, env_H, training)
        if self.render_target is not None:
            self.env.render(self.env_parameters, env_H, A, self.render_target, self.render_prefix)

        # reset reward
        reward = R

        return (t, env_H, agent_H, reward, obs, R, T)

    @tf.function
    def step(self, env_parameters, agent_parameters, state, training=True):
        t, env_H, agent_H, reward, obs, R, T = state

        t += self.dt

        if training:
            agent_R = R
        else:
            agent_R = tf.zeros_like(R)

        agent_H, A = self.agent.step(agent_parameters, agent_H, obs, agent_R, T, training)
        env_H, obs, R, T = self.env.step(env_parameters, env_H, A)

        if self.render_target is not None:
            self.env.render(self.env_parameters, env_H, A, self.render_target, self.render_prefix)

        reward += R

        return (t, env_H, agent_H, reward, obs, R, T)

    @tf.function
    def run_steps(self, env_parameters, agent_parameters, state, n_steps, training=True):
        def func(*_state):
            return self.step(env_parameters, agent_parameters, _state, training)
        def cond(*_state):
            t = _state[0]
            return t < self.episode_length

        return tf.while_loop(cond, func, state, parallel_iterations=1, back_prop=False, maximum_iterations=n_steps)

    def start_run(self):
        self.random.reset_from_seed(self.seed)
        state = self.init(self.env_parameters, self.agent_parameters)
        return state

    def assess_fitness(self, steps_per_batch=50):
        state = self.start_run()

        # run an episode
        training_rewards = []
        testing_rewards = []

        for episode in range(self.n_episodes*self.cycles_per_generation):
            if episode % self.n_episodes < self.training_episodes:
                training = True
            else:
                training = False

            if episode > 0:
                state = self.new_episode(self.env_parameters, self.agent_parameters, state, training)

            while state[0] < self.episode_length:
                state = self.run_steps(self.env_parameters, self.agent_parameters, state, steps_per_batch, training)

            reward = state[3]
            if training:
                training_rewards.append(reward)
            else:
                testing_rewards.append(reward)

            print(f'Mean/Max Reward: {np.mean(reward.numpy())!s}/{np.max(reward.numpy())!s} | Episode: {episode}')

        if training_rewards:
            training_rewards = tf.stack(training_rewards, axis=3)
        if testing_rewards:
            testing_rewards = tf.stack(testing_rewards, axis=3)

        if self.agent_fitness is not None:
            ind_fitness = self.agent_fitness(training_rewards, testing_rewards)
        else:
            ind_fitness = None

        if self.env_fitness is not None:
            task_fitness = self.env_fitness(training_rewards, testing_rewards)
        else:
            task_fitness = None

        return ind_fitness, task_fitness

    def run_generation(self):
        ind_fitness, task_fitness = self.assess_fitness()

        if ind_fitness is not None:
            print('Individual fitness:', str(ind_fitness))
            print(f'Mean/Max Fitness: {np.mean(ind_fitness)!s}/{np.max(ind_fitness)!s}')
            if self.agent_strategy is not None:
                self.agent_genotype = self.agent_strategy(self.genotype_random, self.agent_genotype, ind_fitness, **self.agent_strategy_args)

        if task_fitness is not None:
            print('Task fitness:', str(task_fitness))
            if self.env_strategy is not None:
                self.env_genotype = self.env_strategy(self.genotype_random, self.env_genotype, task_fitness, **self.env_strategy_args)

    def evolution(self, n_generations):
        for generation in range(n_generations):
            print(f'Generation {generation}:')
            self.run_generation()
