''' Produce training curves from obtaining DDPG and eQ critics.
    DDPG training curve is based on the best overall individual from the
    population to depict standard training, rather than the individual
    that ended up being selected (which often skew towards best behavior
    at the end). '''

import argparse

import numpy as np
import matplotlib.pyplot as plt

def extract_eq_trainingline(eq_folder, seed):
    ''' Extract a training curve from a single eQ run. '''
    rewards_filename = f"{eq_folder}/testing_rewards-{seed}.npy"

    all_rewards = np.load(rewards_filename)[
        :, :, 0, 0, :
    ]  # generation, individual, episode
    all_fitness = np.mean(all_rewards, axis=2)
    best_fitness = np.max(all_fitness, axis=1)

    n_generations = best_fitness.size
    # training_rewards_filename = f'{eq_folder}/training_rewards-{seed}.npy'
    n_training_episodes = 50  # np.load(training_rewards_filename).shape[-1]
    n_testing_episodes = all_rewards.shape[-1]
    n_episodes = n_training_episodes + n_testing_episodes
    num_episodes = np.arange(
        n_training_episodes,
        n_training_episodes + n_episodes * n_generations,
        n_episodes,
    )

    return num_episodes, best_fitness


def extract_ddpg_trainingline(ddpg_folder, seed, use_testing=True):
    ''' Extract a smoothed training curve from a single DDPG run.
        Selects critic to display based on run-end test score if
        available, otherwise by overall training performance. '''
    training_filename = f"{ddpg_folder}/training_rewards-{seed}.npy"
    testing_filename = f"{ddpg_folder}/testing_rewards-{seed}.npy"

    training_rewards = np.load(training_filename)[:, 0, 0, :]

    testing_rewards = training_rewards
    if use_testing:
        try:
            testing_rewards = np.load(testing_filename)[:, 0, 0, :]
        except:
            pass

    per_critic_fitness = np.mean(testing_rewards, axis=1)
    best_critic = np.argmax(per_critic_fitness)

    used_training_rewards = training_rewards[best_critic]
    median_training_rewards = np.median(training_rewards, axis=0)

    smoothed_training_rewards = np.convolve(
        used_training_rewards, np.ones(100), "same"
    ) / np.convolve(np.ones_like(used_training_rewards), np.ones(100), "same")
    num_episodes = np.arange(smoothed_training_rewards.size)

    return num_episodes, smoothed_training_rewards


def plot_critic_training(
    eq_folder, eq_seeds, ddpg_folder, ddpg_seeds, subplot_locs, title, task_name
):
    eq_runs = []
    ddpg_runs = []

    for seed in eq_seeds:
        x, y = extract_eq_trainingline(eq_folder, seed)
        eq_runs.append((x, y))
    for seed in ddpg_seeds:
        x, y = extract_ddpg_trainingline(ddpg_folder, seed, False)
        ddpg_runs.append((x, y))

    #####
    # RL
    #####
    plt.subplot(subplot_locs[0])
    plt.fill_between(
        np.mean([x for x, y in ddpg_runs], axis=0),
        np.mean([y for x, y in ddpg_runs], axis=0)
        - np.std([y for x, y in ddpg_runs], axis=0),
        np.mean([y for x, y in ddpg_runs], axis=0)
        + np.std([y for x, y in ddpg_runs], axis=0),
        color="C0",
        alpha=0.5,
    )
    plt.plot(
        np.mean([x for x, y in ddpg_runs], axis=0),
        np.mean([y for x, y in ddpg_runs], axis=0),
        linewidth=1,
        color="C0",
        label="DDPG training reward (critic selected by final performance)",
    )

    # plt.xscale("log")
    plt.title(title[0])
    plt.ylim([-1500, -100])
    plt.yticks(np.arange(-1500, -100, 250))

    plt.xlabel("Training episodes")
    # plt.ylabel("{}\nRewards".format(task_name))
    plt.ylabel("Rewards")

    #####
    # ERL
    #####
    plt.subplot(subplot_locs[1])
    plt.fill_between(
        np.mean([x for x, y in eq_runs], axis=0) / 100,
        np.mean([y for x, y in eq_runs], axis=0)
        - np.std([y for x, y in eq_runs], axis=0),
        np.mean([y for x, y in eq_runs], axis=0)
        + np.std([y for x, y in eq_runs], axis=0),
        color="C2",
        alpha=0.3,
    )
    plt.plot(
        np.mean([x for x, y in eq_runs], axis=0) / 100,
        np.mean([y for x, y in eq_runs], axis=0),
        linewidth=1,
        color="C2",
        label="eQ fitness",
    )

    # plt.xscale("log")
    plt.title(title[1])
    plt.ylim([-1500, -100])
    plt.yticks(np.arange(-1500, -100, 250))

    plt.xlabel("Generations")
    plt.ylabel("Fitness")


def make_figure_1(
    eq_folders,
    eq_seeds,
    ddpg_folders,
    ddpg_seeds,
    outfile,
    titles=[[""] * 3] * 2,
    task_names=[""] * 2,
):
    plt.figure(figsize=[8.5, 2.5])
    subplts = [[132, 133]]
    for i in range(len(eq_folders)):
        plot_critic_training(
            eq_folders[i],
            eq_seeds[i],
            ddpg_folders[i],
            ddpg_seeds[i],
            subplts[i],
            titles[i],
            task_names[i],
        )

    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.savefig(outfile + ".png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-seeds", type=int, default=20, help="number of seeds (seeds are [0-N), excluding N) used for eQ and DDPG")
    args = parser.parse_args()

    seeds = range(args.n_seeds)
    ddpg_folder = 'alife-results/ddpg-[20, 10]-pendulum-1.0'
    eQ_folder = 'alife-results/eQ-[20, 10]-train50-test100-gens200-pendulum-1.0'
    outfile = 'alife-results/all_critic_genotypes.npy'

    make_figure_1(
        [eQ_folder],
        [seeds],
        [ddpg_folder],
        [seeds],
        "alife-results/figure_1",
        titles=[["DDPG", "eQ"]],
    )
