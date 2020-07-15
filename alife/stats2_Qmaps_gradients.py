#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def sim_matrix(spectrum):
    # thanks to Waylon Flinn
    spectrum = np.reshape(spectrum, (spectrum.shape[0], -1))
    matrix = []
    for x, val1 in enumerate(spectrum):
        line = []
        for y, val2 in enumerate(spectrum):
            sim = np.corrcoef(val1, val2)[0,1]
            # I have also experimented with Spearman correlation, used in
            # Evolved Policy Gradients (Houthooft et al)
            # sim = scipy.stats.spearmanr(val1, val2).correlation
            line.append(sim)
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix

def all_crosses(sim_matrix):
    n_inds = sim_matrix.shape[0]
    n_eQ = n_inds // 2
    eQ_range = range(n_eQ)
    ddpg_range = range(n_eQ, n_inds)
    eQ_crosses = set((x,y) for x in eQ_range for y in range(x))
    ddpg_crosses = set((x,y) for x in ddpg_range for y in range(n_eQ,x))
    eQ_ddpg_crosses = set((x,y) for x in eQ_range for y in ddpg_range)
    ddpg_eQ_crosses = set((x,y) for x in ddpg_range for y in eQ_range)
    eQ_scores = [sim_matrix[cross] for cross in eQ_crosses]
    ddpg_scores = [sim_matrix[cross] for cross in ddpg_crosses]
    eQ_ddpg_scores = [sim_matrix[cross] for cross in eQ_ddpg_crosses]

    ddpg_eQ_scores = eQ_ddpg_scores
    # include if you use a non-symmetric similarity measure
    # ddpg_eQ_scores = [sim_matrix[cross] for cross in ddpg_eQ_crosses]

    print('  eQ mean:', np.mean(eQ_scores), '+/-', np.std(eQ_scores),
          '\nddpg mean:', np.mean(ddpg_scores), '+/-', np.std(ddpg_scores),
          '\ncross mean:', np.mean(eQ_ddpg_scores), '+/-', np.std(eQ_ddpg_scores))
    print('eQ-eQ vs DDPG-DDPG', scipy.stats.ttest_ind(eQ_scores, ddpg_scores, equal_var=False))
    print('eQ-eQ vs eQ-DDPG', scipy.stats.ttest_ind(eQ_scores, eQ_ddpg_scores, equal_var=False))
    print('DDPG-DDPG vs eQ-DDPG', scipy.stats.ttest_ind(ddpg_scores, eQ_ddpg_scores, equal_var=False))
    print('----\n')
    return eQ_scores, ddpg_scores, eQ_ddpg_scores, ddpg_eQ_scores


if __name__ == '__main__':
    Q_spectrum = np.load('alife-results/best-Qmaps.npy')
    Q_spectrum -= np.mean(Q_spectrum, axis=(1,2,3), keepdims=True)
    grad_spectrum = np.load('alife-results/best-training-signals.npy')
    

    print('Direct representation (Q-map) analysis:')
    sims = sim_matrix(Q_spectrum)
    print([np.mean(i) for i in all_crosses(sims)])
    plt.imshow(sims, vmin=-1, vmax=1)
    plt.colorbar().set_label('Correlation coefficient')
    plt.xlabel('Critic #')
    plt.ylabel('Critic #')
    plt.title('Correlation matrix of SMQ activation')
    #plt.show()
    print('======================\n\n')


    print('Training signal (dQ/dA) analysis:')
    sims = sim_matrix(grad_spectrum)
    print([np.mean(i) for i in all_crosses(sims)])
    plt.imshow(sims, vmin=-1, vmax=1)
    plt.colorbar().set_label('Correlation coefficient')
    plt.xlabel('Critic #')
    plt.ylabel('Critic #')
    plt.title('Correlation matrix of training signal')
    #plt.show()

