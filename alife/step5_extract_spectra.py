''' Step 5:
    Extract Q-map and gradient spectra from the best critics. '''

#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def derive_sensorimotor_maps(actorfile, SM_outfile, actors_per_critic=10):
    # takes actorfile of genotypes (flattened parameter sets) indexed by actor
    # saves SM_outfile storing sensorimotor values under the following axes: (critic, actor, omega, theta)
    import tensorflow as tf
    import evotask as et

    actor_hidden_layers = [400,300]

    actor_genotype = np.load(actorfile)
    n_critics = actor_genotype.shape[0]
    network_template = et.agent.generate_network_Lillicrap((n_critics,actors_per_critic), [3] + actor_hidden_layers + [1], tf.random)
    network_structure = et.evolution.to_genotype_structure(network_template)
    actor_networks = et.evolution.expand_genotype(actor_genotype, network_structure)

    if tf.test.is_gpu_available():
        device = tf.test.gpu_device_name()
    else:
        device = '/CPU:0'

    with tf.device(device):
        # desired axes: critic, omega, theta, torque.

        omega_values = tf.linspace(-8., 8., 9)
        theta_values = tf.linspace(np.pi, -np.pi, 200)

        expanded_omega_values = tf.tile(tf.reshape(omega_values,(1,1,9,1)),(n_critics,actors_per_critic,1,200))
        expanded_theta_values = tf.tile(tf.reshape(theta_values,(1,1,1,200)),(n_critics,actors_per_critic,9,1))

        state = tf.stack([tf.math.cos(expanded_theta_values), tf.math.sin(expanded_theta_values), expanded_omega_values], axis=-1)

        @tf.function
        def get_A(s):
            return tf.squeeze(et.agent.apply_network(s, actor_networks),axis=-1)
        SM_map = get_A(state)
    np.save(SM_outfile, SM_map)
    return SM_map

def derive_sensorimotor_quality_spectra(criticfile, Q_outfile, grad_outfile):
    # takes criticfile of genotypes (flattened parameter sets) indexed by critic
    # saves Q_outfile storing Q values under the following axes: (critic, omega, theta, torque)
    # & grad_outfile storing gradients under the following axes: (critic, omega, theta, torque)
    import tensorflow as tf
    import evotask as et

    critic_hidden_layers = [20,10]
    actor_hidden_layers = [400,300]

    critic_genotype = np.load(criticfile)
    n_critics = critic_genotype.shape[0]
    network_template = et.agent.generate_network_Lillicrap((n_critics,), [3, (20, 1), 10, 1], tf.random)
    network_structure = et.evolution.to_genotype_structure(network_template)
    critic_networks = et.evolution.expand_genotype(critic_genotype, network_structure)

    if tf.test.is_gpu_available():
        device = tf.test.gpu_device_name()
    else:
        device = '/CPU:0'

    with tf.device(device):
        # desired axes: critic, omega, theta, torque.

        omega_values = tf.linspace(-8., 8., 9)
        theta_values = tf.linspace(np.pi, -np.pi, 200)
        torque_values = tf.linspace(-1., 1., 40)

        expanded_omega_values = tf.tile(tf.reshape(omega_values,(1,9,1,1)),(n_critics,1,200,40))
        expanded_theta_values = tf.tile(tf.reshape(theta_values,(1,1,200,1)),(n_critics,9,1,40))
        expanded_torque_values = tf.tile(tf.reshape(torque_values,(1,1,1,40)),(n_critics,9,200,1))

        state = tf.stack([tf.math.cos(expanded_theta_values), tf.math.sin(expanded_theta_values), expanded_omega_values], axis=-1)
        action = tf.stack([expanded_torque_values], axis=-1)

        @tf.function
        def get_Q_with_grad(s, a):
            Q_values = et.agent.apply_critic_network(s, a, critic_networks)
            total_Q = tf.reduce_sum(Q_values)
            # IMPORTANT: each action contributes ONLY to total Q through its appropriate Q value.
            grads = tf.gradients(total_Q, [a])[0]
            return tf.squeeze(Q_values,axis=-1), tf.squeeze(grads,axis=-1)
        Q_spectrum, grad_spectrum = get_Q_with_grad(state, action)
    np.save(Q_outfile, Q_spectrum)
    np.save(grad_outfile, grad_spectrum)
    return Q_spectrum, grad_spectrum

def plot_spectrum(spectrum, outfile):
    # takes a spectrum with the following axes: (omega, theta, torque) as defined above
    # and saves it to a nicely formatted file.
    max_val = np.max(np.abs(spectrum))
    for idx, omega in enumerate(np.linspace(-8,8,9)):
        plt.subplot(3,3,idx+1)
        plt.imshow(np.transpose(spectrum[idx]), cmap='bwr', vmin=-max_val, vmax=max_val, origin='lower', extent=(np.pi, -np.pi, -2, 2))
        plt.xlabel('CCW theta')
        plt.ylabel('CCW torque')
        plt.title(f'Omega (CCW) = {omega}')
    plt.show()

def plot_SM_map(SM, outfile):
    # takes a spectrum with the following axes: (omega, theta, torque) as defined above
    # and saves it to a nicely formatted file.
    for idx, omega in enumerate(np.linspace(-8,8,9)):
        plt.subplot(3,3,idx+1)
        plt.plot(np.linspace(np.pi, -np.pi, 200), SM[idx]*2)
        plt.ylim((-2.05, 2.05))
        plt.xlim((np.pi, -np.pi))
        plt.xlabel('CCW theta')
        plt.ylabel('CCW torque')
        plt.title(f'Omega (CCW) = {omega}')
    plt.show()


def sim_matrix(spectrum):
    # thanks to Waylon Flinn
    spectrum = np.reshape(spectrum, (spectrum.shape[0], -1))
    matrix = []
    for x, val1 in enumerate(spectrum):
        line = []
        for y, val2 in enumerate(spectrum):
            sim = np.corrcoef(val1, val2)[0,1]
            # sim = np.dot(val1, val2)/np.sqrt(np.dot(val1, val1)*np.dot(val2, val2))
            # sim = -np.log(np.dot(val1-val2, val1-val2))
            # sim = np.where(sim == np.inf, 0, sim)
            line.append(sim)
        matrix.append(line)
    matrix = np.array(matrix)
    print(matrix)
    return matrix


if __name__ == '__main__':
    Q_spectrum, grad_spectrum = derive_sensorimotor_quality_spectra('alife-results/best-critic-genotypes.npy', 'alife-results/best-Qmaps.npy', 'alife-results/best-training-signals.npy')

    # No longer used in paper but it may be of interest to compare retrained
    # actors perhaps of different architectures, to DDPG actors and to eQ
    # actors trained in the original architecture and conditions.
    # SM_map = derive_sensorimotor_maps('alife-results/retraining-1.0-[20, 10]/actor-alife_results_best_critic_genotypes-0.npy', 'alife-results/SM-pendulum-1.0.npy')

