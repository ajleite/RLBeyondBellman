#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def plot_actor_retraining(infile, title=""):
    retesting_rewards = np.load(infile)  # critic, null, actor, episode
    n_eQ = retesting_rewards.shape[0]//2
    n_actors = retesting_rewards.shape[2]
    eQ_retraining_rewards = retesting_rewards[:n_eQ].reshape((n_eQ*n_actors, -1, 100))
    eQ_retraining_rewards = np.mean(eQ_retraining_rewards, axis=-1)
    ddpg_retraining_rewards = retesting_rewards[n_eQ:].reshape((n_eQ*n_actors, -1, 100))
    ddpg_retraining_rewards = np.mean(ddpg_retraining_rewards, axis=-1)
    n_cycles = ddpg_retraining_rewards.shape[1]

    m_eQ = np.mean(eQ_retraining_rewards, axis=0)
    s_eQ = np.std(eQ_retraining_rewards, axis=0)
    m_ddpg = np.mean(ddpg_retraining_rewards, axis=0)
    s_ddpg = np.std(ddpg_retraining_rewards, axis=0)

    plt.plot(
        np.arange(25, 25 + 25 * n_cycles, 25),
        m_eQ,
        linewidth=1,
        color="#ff0099",
        label="eQ",
    )
    plt.plot(
        np.arange(25, 25 + 25 * n_cycles, 25),
        m_ddpg,
        linewidth=1,
        color="#0099ff",
        label="DDPG",
    )
    plt.fill_between(
        np.arange(25, 25 + 25 * n_cycles, 25),
        m_eQ - s_eQ,
        m_eQ + s_eQ,
        color="#ff0099",
        alpha = 0.3,
    )
    plt.fill_between(
        np.arange(25, 25 + 25 * n_cycles, 25),
        m_ddpg - s_ddpg,
        m_ddpg + s_ddpg,
        color="#0099ff",
        alpha = 0.3,
    )

    plt.xlabel("Training episodes")
    plt.legend(loc="lower right")

if __name__ == "__main__":
    plt.figure(figsize=[9,2])
    for i,arch in enumerate(["[400, 300]", "[20, 10]", "[5]"]):
        plt.subplot(1,3,i+1)
        filename = f"alife-results/retraining-1.0-{arch}/testing_rewards-alife_results_best_critic_genotypes-0.npy"
        plot_actor_retraining(
            filename,
            title=f'Training new {arch.replace("[","[3, ").replace("]",", 1]")} actors on inverted pendulum task',
        )
        if i == 0:
            plt.ylabel("Mean reward on 100\ntest episodes")


    plt.tight_layout()
    plt.savefig("alife-results/figure_2_v2.pdf")
    plt.savefig("alife-results/figure_2_v2.png")
    plt.show()

