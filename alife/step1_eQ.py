''' Step 1:
    Perform the eQ algorithm to obtain adaptive Q-maps.
    Default values are the ones used for our conference paper.
    Repeat with different seeds! (We did 0-20, exclusive of 20.)

    Unfortunately, a GPU is practically required to replicate in any reasonable
    amount of time, although you can experiment with tiny actor architectures.
    Takes ~10h to complete one run on powerful hardware (GTX 1080 or V100). '''

import sys
import os
import pickle

import tensorflow as tf
import numpy as np

import evotask as et

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=0, help="integer seed to reproduce calulations")
parser.add_argument("-d", "--difficulty", type=float, default=1., choices=(0.,.1,.2,.5,1.), help="proportion of initial condition range to use")
parser.add_argument("--train", type=int, default=50, help="number of training episodes, used each generation")
parser.add_argument("--test", type=int, default=100, help="number of testing episodes, used each generation")
parser.add_argument("--gens", type=int, default=200, help="number of generations to evolve")
args = parser.parse_args()

seed = args.seed
difficulty = args.difficulty
training_episodes = args.train
testing_episodes = args.test
n_generations = args.gens

critic_hidden_layers = [20,10]
population_size = 160

out_dir = f'alife-results/eQ-{critic_hidden_layers}-train{training_episodes}-test{testing_episodes}-gens{n_generations}-pendulum-{difficulty}'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)

fn = out_dir+f'/critic-{seed}.npy'
provisional_fn = out_dir+f'/critic-gen{n_generations//2}-{seed}.npy'
actor_fn = out_dir+f'/actor-{seed}.npy'
vars_fn = out_dir+f'/vars-{seed}.pnpy'
log_fn = out_dir+f'/log-{seed}.txt'
train_fn = out_dir+f'/training_rewards-{seed}.npy'
test_fn = out_dir+f'/testing_rewards-{seed}.npy'
sys.stdout = open(log_fn, 'a')

all_training_rewards = []
all_testing_rewards = []

def agent_fitness(training_episodes, testing_episodes):
    all_training_rewards.append(training_episodes)
    all_testing_rewards.append(testing_episodes)
    return tf.reduce_mean(testing_episodes, axis=[1,2,3])


evolution_args = {'dimensions': (population_size,1,1), 'seed': seed, 'dt': .05, 'episode_length': 10, 'training_episodes': training_episodes, 'testing_episodes': testing_episodes}
agent = et.agent.eQAgent
agent_args = {'obs_dim': 3, 'action_dim': 1, 'critic_hidden_layers': critic_hidden_layers, 'actor_hidden_layers': [400,300], 'init_noise': 'Lillicrap', 'noise_fixed': False}
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

    # saving halfway through because I'm nervous
    ev.evolution(n_generations//2)
    np.save(provisional_fn, ev.agent_genotype)
    ev.evolution(n_generations-n_generations//2)

    np.save(fn, ev.agent_genotype)
    np.save(train_fn, all_training_rewards)
    np.save(test_fn, all_testing_rewards)
    np.save(actor_fn, et.evolution.to_genotype(ev.agent.actor_network))
    pickle.dump(ev.agent.get_vars(), open(vars_fn, 'wb'))
