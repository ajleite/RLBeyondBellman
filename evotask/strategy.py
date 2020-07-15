''' A number of different evolutionary strategies. basic_ev is a simple but effective
    multi-rate hill-climbing algorithm, and is the primary one we are currently using.
    It is not very efficient because it tries multiple jump radii at once, but this allows
    it to work effectively on a variety of landscapes with practically no tuning. '''

import tensorflow as tf
import numpy as np

@tf.function
def basic_ev(rand, genotypes, fitness, n_parents=8, cohort_factor=.8, mu=0.08):
    ''' {n_parents} parents are preserved as an elitist cohort.
        The children's mutation rates will be calculated as follows:
        The ith cohort of children has a mutation rate of ((i + 1)**{cohort_factor} - 1)*{mu}'''
    n_cohorts=(genotypes.shape[0]//n_parents)+1
    elite = tf.argsort(-fitness)[:n_parents]
    tf.print(elite, summarize=-1)
    indices = tf.concat([elite] * n_cohorts, axis=0)[:genotypes.shape[0]]

    parents_gene = tf.gather(genotypes, indices)

    mutation_factor = tf.reshape(tf.constant([((i // n_parents + 1)**cohort_factor - 1) for i in range(genotypes.shape[0])]), (genotypes.shape[0], 1)) # each cohort mutates a little more than the last
    children_gene = parents_gene + rand.normal(parents_gene.shape) * mutation_factor * mu

    return children_gene

@tf.function
def basic_ev_big_mu(rand, genotypes, fitness, n_parents=8, mu=0.08, big_mu=1.):
    ''' Like basic_ev, but sets cohort_factor based on the desired maximum mutation rate. '''
    n_cohorts=(genotypes.shape[0]//n_parents)+1
    cohort_factor = np.log(big_mu/mu) / np.log(n_cohorts-1)

    return basic_ev(rand, genotypes, fitness, n_parents, cohort_factor, mu)

def nes(rand, genotypes, fitness, alpha=.001, mu=.05):
    ''' Implementation of Natural Evolutionary Strategies, an explicitly
        gradient-estimating population-based approach from the ML community. '''
    fitness = (fitness - tf.math.reduce_mean(fitness)) / tf.math.reduce_std(fitness)

    mean_by_gene = tf.math.reduce_mean(genotypes, axis=0, keepdims=True)
    std_by_gene = tf.math.reduce_std(genotypes, axis=0, keepdims=True)
    dev_from_mean = (genotypes - std_by_gene) / std_by_gene
    gradient = tf.math.reduce_mean(tf.reshape(fitness, (-1,1)) * dev_from_mean, axis=0, keepdims=True) / std_by_gene

    new_mean = mean_by_gene + alpha * gradient

    children_genotypes = new_mean + (rand.normal(genotypes.shape) * mu)

    return children_genotypes

def rotate(rand, genotypes, fitness):
    ''' Meaningless, debugging strategy. '''
    return tf.concat((genotypes[1:], genotypes[:1]), axis=0)

def null(rand, genotypes, fitness):
    ''' Meaningless, debugging strategy. '''
    return genotypes

@tf.function
def naive_mga(rand, genotypes, fitness, transfection_rate=.5, mutation_rate=0.1, return_winner_fitness=False):
    ''' Molecular genetic algorithm based on contests of 2 individuals,
        with transfection and mutation of the loser. '''
    shuffled_indices = tf.argsort(rand.uniform((genotypes.shape[0],)))
    m1, m2 = shuffled_indices[:genotypes.shape[0] // 2], shuffled_indices[genotypes.shape[0] // 2:(genotypes.shape[0] // 2)*2]

    m1_genes = tf.gather(genotypes, m1)
    m2_genes = tf.gather(genotypes, m2)
    m1_fitness = tf.gather(fitness, m1)
    m2_fitness = tf.gather(fitness, m2)

    select_m1_genes = tf.expand_dims(m1_fitness > m2_fitness, axis=1)
    winner_genes = tf.where(select_m1_genes, m1_genes, m2_genes)
    loser_genes = tf.where(select_m1_genes, m2_genes, m1_genes)

    transfection = rand.uniform(loser_genes.shape) < transfection_rate
    mutation = rand.normal(loser_genes.shape) * mutation_rate
    transfected_genes = tf.where(transfection, winner_genes, loser_genes) + mutation

    if return_winner_fitness:
        winner_fitness = tf.where(m1_fitness > m2_fitness, m1_fitness, m2_fitness)
        if genotypes.shape[0] % 2 == 1:
            free_winner = shuffled_indices[-1:]
            free_winner_genes = tf.gather(genotypes, free_winner)
            free_winner_fitness = tf.gather(fitness, free_winner)
            winner_genes = tf.concat((winner_genes, free_winner_genes), axis=0)
            winner_fitness = tf.concat((winner_fitness, free_winner_fitness), axis=0)
        return winner_genes, winner_fitness, transfected_genes

    if genotypes.shape[0] % 2 == 1:
        free_winner = shuffled_indices[-1:]
        free_winner_genes = tf.gather(genotypes, free_winner)
        return tf.concat((winner_genes, free_winner_genes, transfected_genes), axis=0)
    else:
        return tf.concat((winner_genes, transfected_genes), axis=0)


def lazy_mga(rand, genotypes, fitness, transfection_rate=.5, mutation_rate=0.15):
    ''' MGA that stores already-assessed genotypes in the namespace of the function.
        Need to manually pull best solution out of strategy function's namespace. '''
    if not hasattr(lazy_mga, 'winner_fitness'):
        last_winner_fitness = fitness
        last_winner_genotypes = genotypes
    else:
        last_winner_fitness = lazy_mga.winner_fitness
        last_winner_genotypes = lazy_mga.winner_genotypes

    all_genotypes = tf.concat((last_winner_genotypes, genotypes), axis=0)
    all_fitness = tf.concat((last_winner_fitness, fitness), axis=0)
    tf.print('True best fitness: ', tf.reduce_max(all_fitness))

    lazy_mga.winner_genotypes, lazy_mga.winner_fitness, transfected_genes = naive_mga(rand, all_genotypes, all_fitness, transfection_rate, mutation_rate, return_winner_fitness=True)

    return transfected_genes

@tf.function
def brutal_ev_sub(rand, genotypes, fitness, n_parents, mu):
    n_cohorts=(genotypes.shape[0]//n_parents)+1
    elite = tf.argsort(-fitness)[:n_parents]
    indices = tf.concat([elite] * n_cohorts, axis=0)[:genotypes.shape[0]]

    parents_gene = tf.gather(genotypes, indices)
    cohort_factor = np.log(5) / np.log(n_cohorts)
    mutation_factor = tf.reshape(tf.constant([((i // n_parents + 1)**cohort_factor - 1) for i in range(genotypes.shape[0])]), (genotypes.shape[0], 1))
    children_gene = parents_gene + rand.normal(parents_gene.shape) * mutation_factor * mu

    interesting = tf.reduce_any(elite != tf.range(n_parents))

    return interesting, elite, children_gene

def brutal_ev(rand, genotypes, fitness, n_parents=4, mus=[1.0, 0.2, 0.04, 0.008], tolerance=2):
    ''' Tries locally hill-climbing many different hills in sequence. '''
    if not hasattr(brutal_ev, 'initialized'):
        brutal_ev.initialized = True
        brutal_ev.mu_index = 0
        brutal_ev.boredom = 0

    interesting, winners, children_gene = brutal_ev_sub(rand, genotypes, fitness, n_parents, mus[brutal_ev.mu_index])

    if winners[0] != 0:
        brutal_ev.boredom = 0
        tf.print('Interesting! The winners were', winners)
    elif interesting:
        tf.print('Interesting... the winners were', winners)
    else:
        tf.print('That wasn\'t very interesting... The winners were', winners, '...again.')
        brutal_ev.boredom += 1

    winner_index = winners[0]
    if not hasattr(brutal_ev, 'favorite_fitness') or fitness[winner_index] > brutal_ev.favorite_fitness:
        brutal_ev.favorite = genotypes[winner_index]
        brutal_ev.favorite_fitness = fitness[winner_index]
    elif fitness[winner_index] < brutal_ev.favorite_fitness:
        tf.print('Back in my day, there were scores like', brutal_ev.favorite_fitness)

    if brutal_ev.boredom == tolerance:
        brutal_ev.boredom = 0
        brutal_ev.mu_index += 1
        if brutal_ev.mu_index == len(mus):
            print('SO bored. Moving on.')
            brutal_ev.mu_index = 0
            children_gene = tf.stack([brutal_ev.favorite]*genotypes.shape[0], axis=0) + rand.normal(genotypes.shape)*mus[brutal_ev.mu_index]
        else:
            print(f'Bored. Trying smaller mu: {mus[brutal_ev.mu_index]}')
            interesting, winners, children_gene = brutal_ev_sub(rand, genotypes, fitness, n_parents, mus[brutal_ev.mu_index])

    return children_gene

@tf.function
def deme_ev(rand, genotypes, fitness, n_parents=3, n_demes=5, mu=0.08, big_mu=1.):
    ''' Splits population up into demes, only crossing between them with low
        probability. '''
    pop_size = genotypes.shape[0]
    deme_size = pop_size//n_demes
    n_cohorts=(deme_size//n_parents)+1
    cohort_factor = np.log(big_mu/mu) / np.log(n_cohorts-1)
    mutation_factor = tf.reshape(tf.constant([((i // n_parents + 1)**cohort_factor - 1) for i in range(deme_size)]), (deme_size, 1)) # each cohort mutates a little more than the last

    all_indices = []
    all_mutation_factors = []
    for i, deme_fitness in enumerate(tf.split(fitness, n_demes)):
        elite = tf.argsort(-deme_fitness)[:n_parents] + i*deme_size
        tf.print('Deme', i, elite - i*deme_size, tf.gather(fitness, elite), summarize=-1)
        indices = tf.concat([elite]*(n_cohorts-1) + [rand.uniform((n_parents,), 0, pop_size, dtype=tf.int32)], axis=0)[:deme_size] # one cohort's worth of heavily mutated exchange between demes
        mutation_factor = tf.reshape(tf.constant([((i // n_parents + 1)**cohort_factor - 1) for i in range(deme_size)]), (deme_size, 1)) # each cohort mutates a little more than the last
        all_indices.append(indices)
        all_mutation_factors.append(mutation_factor)

    indices = tf.concat(all_indices, axis=0)
    mutation_factor = tf.concat(all_mutation_factors, axis=0)

    parents_gene = tf.gather(genotypes, indices)
    children_gene = parents_gene + rand.normal(parents_gene.shape) * mutation_factor * mu

    return children_gene
