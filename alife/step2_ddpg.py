''' Step 2:
    Perform the DDPG algorithm to obtain Bellman-fitted Q-maps.
    Due to instability with such small critics and to match populations,
    we are using the same population size as eQ and will extract the best
    critic from each run.
    Default values are the ones used for our conference paper.
    Repeat with different seeds! (We did 0-20, exclusive of 20.)

    If you do not have a GPU, reduce the population size (meaning you will
    select a less unusually lucky run) and the number of training episodes.
    Takes ~6h to complete one run on powerful hardware (GTX 1080 or V100). '''

import pickle
import sys
import os

import tensorflow as tf
import numpy as np

import evotask as et

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0, help="integer seed to reproduce calulations")
parser.add_argument("-d", "--difficulty", type=float, default=1., choices=(0.,.1,.2,.5,1.), help="proportion of initial condition range to use")
parser.add_argument("-t", "--train", type=int, default=5000, help="number of training episodes")
args = parser.parse_args()

seed = args.seed
difficulty = args.difficulty
training_episodes = args.train

population_size = 160
critic_hidden_layers = [20,10]

directory = f'alife-results/ddpg-{critic_hidden_layers}-pendulum-{difficulty}'
if not os.path.isdir(directory):
    os.makedirs(directory, exist_ok=True)
suffix = f'{seed}'

def agent_fitness(training_episodes, testing_episodes):
    np.save(f'{directory}/training_rewards-{suffix}.npy', training_episodes)
    np.save(f'{directory}/testing_rewards-{suffix}.npy', testing_episodes)
    return tf.reduce_mean(testing_episodes, axis=[1,2,3])

evolution_args = {'dimensions': (population_size,1,1), 'seed': seed, 'dt': .05, 'episode_length': 10, 'training_episodes': training_episodes, 'testing_episodes': 100}
agent = et.agent.DDPGAgent
agent_args = {'obs_dim': 3, 'action_dim': 1, 'critic_hidden_layers': critic_hidden_layers, 'actor_hidden_layers': [400,300], 'init_noise': 'Lillicrap'}
agent_fitness = agent_fitness
agent_strategy = et.strategy.basic_ev
agent_strategy_args = {}

environment = et.env.PendulumEnv
env_args = {'default_difficulty': difficulty}
env_fitness = None
env_strategy = None
env_strategy_args = {}


if tf.test.is_gpu_available():
    device = tf.test.gpu_device_name()
else:
    device = '/CPU:0'

with tf.device(device):
    ev = et.Evolution(evolution_args, agent, agent_args, agent_fitness, agent_strategy, agent_strategy_args,
                      environment, env_args, env_fitness, env_strategy, env_strategy_args)

    # preform a single generation of "evolution"
    ev.evolution(1)
    pickle.dump(ev.agent.get_vars(), open(f'{directory}/vars-{suffix}.pnpy', 'wb'))
    np.save(f'{directory}/critic-{suffix}.npy', et.evolution.to_genotype(ev.agent.critic_network))
    np.save(f'{directory}/actor-{suffix}.npy', et.evolution.to_genotype(ev.agent.actor_network))
