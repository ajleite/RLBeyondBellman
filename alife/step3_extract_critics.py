''' Step 3:
    Combine DDPG and eQ critic genotypes into a single file for further analysis. '''

import argparse

import numpy as np
import matplotlib.pyplot as plt

def extract_eq_critic(eq_folder, seed):
    # best genotype from final generation.
    filename = f'{eq_folder}/critic-{seed}.npy'
    critics = np.load(filename)
    return critics[0]

def extract_ddpg_critic(ddpg_folder, seed):
    rewards_filename = f'{ddpg_folder}/testing_rewards-{seed}.npy'
    critics_filename = f'{ddpg_folder}/critic-{seed}.npy'

    all_rewards = np.load(rewards_filename)[:,0,0,:]
    per_critic_fitness = np.mean(all_rewards,axis=1)
    best_critic = np.argmax(per_critic_fitness)
    critics = np.load(critics_filename)
    return critics[best_critic]

def concatenate_critics(eq_folder, eq_seeds, ddpg_folder, ddpg_seeds, outfile):
    critics = []
    for seed in eq_seeds:
        critics.append(extract_eq_critic(eq_folder, seed))
    for seed in ddpg_seeds:
        critics.append(extract_ddpg_critic(ddpg_folder, seed))
    np.save(outfile, critics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-seeds", type=int, default=20, help="number of seeds (seeds are [0-N), excluding N) used for eQ and DDPG")
    args = parser.parse_args()

    seeds = range(args.n_seeds)
    ddpg_folder = 'alife-results/ddpg-[20, 10]-pendulum-1.0'
    eQ_folder = 'alife-results/eQ-[20, 10]-train50-test100-gens200-pendulum-1.0'
    outfile = 'alife-results/best-critic-genotypes.npy'

    concatenate_critics(eQ_folder, seeds, ddpg_folder, seeds, outfile)
