#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def plot_actor_retraining_systematic(infile, outfile, title=''):
    retesting_rewards = np.load(infile) # critic, null, actor, episode
    n_eQ = retesting_rewards.shape[0]//2
    n_actors = retesting_rewards.shape[2]
    eQ_retraining_rewards = retesting_rewards[:n_eQ].reshape((n_eQ*n_actors, -1, 100))
    eQ_retraining_rewards = np.mean(eQ_retraining_rewards, axis=-1)
    ddpg_retraining_rewards = retesting_rewards[n_eQ:].reshape((n_eQ*n_actors, -1, 100))
    ddpg_retraining_rewards = np.mean(ddpg_retraining_rewards, axis=-1)
    n_cycles = ddpg_retraining_rewards.shape[1]

    # interspersing to be fair
    for dy, ey in zip(ddpg_retraining_rewards, eQ_retraining_rewards):
        plt.plot(np.arange(25,25+25*n_cycles,25), dy, marker='.', markersize=1, linewidth=.1, color='#44aaff', alpha=.2)
        plt.plot(np.arange(25,25+25*n_cycles,25), ey, marker='.', markersize=1, linewidth=.1, color='#ff44aa', alpha=.2)

    plt.plot(np.arange(25,25+25*n_cycles,25), np.mean(eQ_retraining_rewards, axis=0), linewidth=1, color='#ff0099', label='eQ retraining reward')
    plt.plot(np.arange(25,25+25*n_cycles,25), np.mean(ddpg_retraining_rewards, axis=0), linewidth=1, color='#0099ff', label='DDPG retraining reward')

    plt.xlabel('Training episodes')
    plt.ylabel('Mean reward on 100 test episodes')
    plt.title(title)
    plt.legend()
    if type(outfile) in [list, tuple]:
        for outfile_i in outfile:
            plt.savefig(outfile_i)
    else:
        plt.savefig(outfile)
    plt.close()

def plot_new_actor_tests(infile, outfile, title=''):
    retesting_rewards = np.load(infile) # critic, null, actor, episode
    n_eQ = retesting_rewards.shape[0]//2

    retesting_outcomes = np.mean(retesting_rewards[:,0,:,-100:], axis=-1) # critic, actor
    eQ_outcomes = retesting_outcomes[:n_eQ]
    ddpg_outcomes = retesting_outcomes[n_eQ:]
    import scipy.stats
    print(infile)
    print(' At convergence')
    print('  eQ mean:', np.mean(eQ_outcomes), '+/-', np.std(np.mean(eQ_outcomes,axis=1)),
          '\nddpg mean:', np.mean(ddpg_outcomes), '+/-', np.std(np.mean(ddpg_outcomes,axis=1)))
    # The statistical unit here is the critic, not the retrained actor.
    # This makes us less likely to find a significant result.
    print(scipy.stats.ttest_ind(np.mean(eQ_outcomes,axis=1), np.mean(ddpg_outcomes,axis=1), equal_var=False))

    retesting_fifties = np.mean(retesting_rewards[:,0,:,100:200], axis=-1) # critic, actor
    eQ_fifties = retesting_fifties[:n_eQ]
    ddpg_fifties = retesting_fifties[n_eQ:]
    import scipy.stats
    print(' After first 50 eps')
    print('  eQ mean:', np.mean(eQ_fifties), '+/-', np.std(np.mean(eQ_fifties,axis=1)),
          '\nddpg mean:', np.mean(ddpg_fifties), '+/-', np.std(np.mean(ddpg_fifties,axis=1)))
    # The statistical unit here is the critic, not the retrained actor.
    # This makes us less likely to find a significant result.
    print(scipy.stats.ttest_ind(np.mean(eQ_fifties,axis=1), np.mean(ddpg_fifties,axis=1), equal_var=False))
    print('----\n')

    ax1=plt.subplot(2,2,1)
    plt.title('Actors trained by eQ critics')
    plt.ylabel('100-episode mean reward')
    for critic_outcomes in eQ_outcomes:
        x_shift = .5 + np.random.random(critic_outcomes.size)
        plt.scatter(x_shift, critic_outcomes, marker='.', s=1)
    plt.violinplot(eQ_outcomes.flatten())

    plt.subplot(2,2,2,sharey=ax1)
    plt.title('Actors trained by DDPG critics')
    plt.ylabel('100-episode mean reward')
    for critic_outcomes in ddpg_outcomes:
        x_shift = .5 + np.random.random(critic_outcomes.size)
        plt.scatter(x_shift, critic_outcomes, marker='.', s=1)
    plt.violinplot(ddpg_outcomes.flatten())

    gridspec = ax1.get_gridspec()

    plt.subplot(gridspec[1,:])
    plt.boxplot([eQ_outcomes.flatten(), ddpg_outcomes.flatten()], labels=['eQ', 'ddpg'], widths=.8, showfliers=False)

    plt.suptitle(title)
    if type(outfile) in [list, tuple]:
        for outfile_i in outfile:
            plt.savefig(outfile_i)
    else:
        plt.savefig(outfile)
    plt.close()

if __name__ == '__main__':
    for arch in ['[400, 300]', '[20, 10]', '[5]']:
        training_curve_plots = [] # [f'alife-results/retraining1.0-{arch}-sys.pdf', f'alife-results/retraining1.0-{arch}-sys.png']
        test_curve_plots = [] # [f'alife-results/retraining1.0-{arch}-sys-test.pdf', f'alife-results/retraining1.0-{arch}-sys-test.png']
        plot_actor_retraining_systematic(f'alife-results/retraining-1.0-{arch}/testing_rewards-alife_results_best_critic_genotypes-0.npy',
            training_curve_plots,
            title=f'Training new {arch.replace("[","[3, ").replace("]",", 1]")} actors on inverted pendulum task')
        plot_new_actor_tests(f'alife-results/retraining-1.0-{arch}/testing_rewards-alife_results_best_critic_genotypes-0.npy',
            test_curve_plots,
            title=f'Training new {arch.replace("[","[3, ").replace("]",", 1]")} actors on inverted pendulum task')

