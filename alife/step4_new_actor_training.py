''' Step 4:
    Re-train fresh actors using the extracted critics.
    Produce the data required to generate training curves by training
    for 25-episode periods and then assessing on a set of standard conditions
    for 100 episodes, repeating until convergence.
    Repeat for tiny 5-hidden unit actors, small (20, 10)-hidden unit actors,
    and standard (400, 300)-hidden unit (as in DDPG) actors.

    To feasibly run on CPU, reduce n_conditions (the number of actors; 2 is not
    unreasonable) and wait fewer cycles for convergence. (We found a crossover
    point on (400, 300) actors after 250 episodes of training, so perform at
    the very least 12 train/test cycles for 300 episodes of training.) I would
    expect the (400, 300) case to take a number of hours even after making
    these changes if you must run on CPU.
    In the standard conditions, takes ~3h to complete on powerful hardware
    (GTX 1080 or V100). '''

import sys
import os

import tensorflow as tf
import numpy as np

import evotask as et

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("critic_genotype", nargs='?', default="alife-results/best-critic-genotypes.npy", help="numpy file containing flattened critic network")
parser.add_argument("-s", "--seed", type=int, default=0, help="integer seed to reproduce calulations")
parser.add_argument("-d", "--difficulty", type=float, default=1., choices=(0.,.1,.2,.5,1.), help="proportion of initial condition range to use")
parser.add_argument("-a", "--actor", type=int, default=400300, choices=(5,2010,400300), help="hidden layer architecture of actor ([5], [20,10], [400,300])")
args = parser.parse_args()

seed = args.seed
difficulty = args.difficulty

n_conditions = 2
critic_hidden_layers = [20, 10]
training_episodes = 25
testing_episodes = 100

# 200 cycles -> 5000 training eps for [5] and [20, 10] actor (25000 total)
#  40 cycles -> 1000 training eps for [400, 300] actor        (5000 total)
if args.actor == 5:
    cycles_per_generation = 200
    actor_hidden_layers = [5]
elif args.actor == 2010:
    cycles_per_generation = 200
    actor_hidden_layers = [20,10]
else:
    cycles_per_generation = 40
    actor_hidden_layers = [400, 300]

fn = args.critic_genotype
safe_fn = fn.replace('.npy','').replace('/','_').replace('.','_').replace('-','_').replace(' ','')
critic_genotype = np.load(fn)
dimensions = (critic_genotype.shape[0], 1, n_conditions)

directory = f'alife-results/retraining-{difficulty}-{actor_hidden_layers}'
if not os.path.isdir(directory):
    os.makedirs(directory, exist_ok=True)
suffix = f'{safe_fn}-{seed}'


def agent_fitness(training_rewards, testing_rewards):
    np.save(f'{directory}/training_rewards-{suffix}.npy', training_rewards)
    np.save(f'{directory}/testing_rewards-{suffix}.npy', testing_rewards)
    return tf.reduce_mean(testing_rewards, axis=[1,2,3])

evolution_args = {'dimensions': dimensions, 'seed': seed, 'dt': .05, 'episode_length': 10, 'training_episodes': training_episodes, 'testing_episodes': testing_episodes, 'cycles_per_generation': cycles_per_generation, 'render_target': None}
agent = et.agent.eQAgent
agent_args = {'obs_dim': 3, 'action_dim': 1, 'critic_hidden_layers': critic_hidden_layers, 'actor_hidden_layers': actor_hidden_layers, 'init_noise': 'Lillicrap'}
agent_fitness = agent_fitness
agent_strategy = None
agent_strategy_args = {}

environment = et.env.SystematicPendulumEnv
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

    ev.agent_genotype = critic_genotype
    # preform a single generation of "evolution"
    ev.evolution(1)
    np.save(f'{directory}/actor-{suffix}.npy', et.evolution.to_genotype(ev.agent.actor_network))
