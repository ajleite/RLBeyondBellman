''' Code to simulate various agents whose behavior is determined by parameters. '''

import itertools

import tensorflow as tf
import numpy as np

def network_vars(instances, architecture, name='network_vars'):
    ''' Generates a set of variables able to store neural
        networks indexed along the axes defined by instances,
        using the provided architecture.
        If a layer in the architecture is a tuple, the first
        element is the number of hidden units and the second
        is the dimension of a new input. '''
    with tf.name_scope(name):
        network = []
        for n, (size_1, size_2) in enumerate(zip(architecture[:-1], architecture[1:])):
            if type(size_2) == tuple:
                size_2 = size_2[0]
            if type(size_1) == tuple:
                size_1 = sum(size_1)

            W_shape = list(instances)+[size_1, size_2]
            W = tf.Variable(tf.zeros(W_shape, tf.float32), name=f'W{n}')

            b_shape = list(instances)+[size_2]
            b = tf.Variable(tf.zeros(b_shape, tf.float32), name=f'b{n}')

            network.append((W, b))
        return network

@tf.function
def assign_network(network_vars, network_vals):
    for (W_var, b_var), (W, b) in zip(network_vars, network_vals):
        W_var.assign(W)
        b_var.assign(b)

@tf.function
def read_network(network_vars):
    network = []
    for (W_var, b_var) in network_vars:
        network.append((W_var.read_value(), b_var.read_value()))
    return network

def duplicate_network(network, axis, copies):
    new_network = []
    for W, b in network:
        new_W = tf.stack([W]*copies, axis=axis)
        new_b = tf.stack([b]*copies, axis=axis)
        new_network.append((new_W, new_b))
    return new_network

@tf.function
def generate_network(instances, architecture, random, noise):
    ''' Generates a group of neural networks indexed along
        the axes defined by instances, using the provided
        architecture. Generates initial weights using a tf
        random Generator random and the scalar noise.
        If a layer in the architecture is a tuple, the first
        element is the number of hidden units and the second
        is the dimension of a new input. '''
    network = []
    for n, (size_1, size_2) in enumerate(zip(architecture[:-1], architecture[1:])):
        if type(size_2) == tuple:
            size_2 = size_2[0]
        if type(size_1) == tuple:
            size_1 = sum(size_1)

        W_std = noise/(size_1*size_2)
        W_shape = list(instances)+[size_1, size_2]
        W = random.truncated_normal(W_shape, stddev=W_std)

        b_std = noise/size_2
        b_shape = list(instances)+[size_2]
        b = random.truncated_normal(b_shape, stddev=b_std)

        network.append((W, b))
    return network

@tf.function
def generate_network_Emami(instances, architecture, random, trunc_norm_stdev=0.02, uniform_max=0.003):
    ''' Generates a group of neural networks indexed along
        the axes defined by instances, using the provided
        architecture. Generates initial weights using a tf
        random Generator random and the scalar noise.
        If a layer in the architecture is a tuple, the first
        element is the number of hidden units and the second
        is the dimension of a new input.
        Mimics Patrick Emami's implementation of the DDPG
        algorithm, which uses a truncated normal distribution
        during the first layers and a uniform distribution for
        the final ones. (The minimum of the uniform distribution
        is -uniform_max, not 0.) '''
    network = []
    for n, (size_1, size_2) in enumerate(zip(architecture[:-1], architecture[1:])):
        if type(size_2) == tuple:
            size_2 = size_2[0]
        if type(size_1) == tuple:
            size_1 = sum(size_1)

        W_shape = list(instances)+[size_1, size_2]
        if n == len(architecture)-2:
            W = random.uniform(W_shape, -uniform_max, uniform_max)
        else:
            W = random.truncated_normal(W_shape, stddev=trunc_norm_stdev)

        b_shape = list(instances)+[size_2]
        if n == len(architecture)-2:
            b = random.uniform(b_shape, -uniform_max, uniform_max)
        else:
            b = random.truncated_normal(b_shape, stddev=trunc_norm_stdev)

        network.append((W, b))
    return network

@tf.function
def generate_network_Lillicrap(instances, architecture, random, uniform_max=0.003):
    ''' Generates a group of neural networks indexed along
        the axes defined by instances, using the provided
        architecture. Generates initial weights using a tf
        random Generator random and the scalar noise.
        If a layer in the architecture is a tuple, the first
        element is the number of hidden units and the second
        is the dimension of a new input.
        Mimics Lillicrap et al's specification of the DDPG
        algorithm, which uses a uniform distribution 1/sqrt(fan_in)
        during the first layers and a uniform distribution for
        the final one. (The minimum of the uniform distribution
        is -uniform_max, not 0.) '''
    network = []
    for n, (size_1, size_2) in enumerate(zip(architecture[:-1], architecture[1:])):
        if type(size_2) == tuple:
            size_2 = size_2[0]
        if type(size_1) == tuple:
            size_1 = sum(size_1)

        W_shape = list(instances)+[size_1, size_2]
        if n == len(architecture)-2:
            W = random.uniform(W_shape, -uniform_max, uniform_max)
        else:
            W = random.uniform(W_shape, -1/size_1**.5, 1/size_1**.5)

        b_shape = list(instances)+[size_2]
        if n == len(architecture)-2:
            b = random.uniform(b_shape, -uniform_max, uniform_max)
        else:
            b = random.uniform(b_shape, -1/size_1**.5, 1/size_1**.5)

        network.append((W, b))
    return network

@tf.function
def apply_network(x, network, signal_function=tf.nn.relu, final_function=tf.nn.tanh, name='apply_network'):
    ''' Applies a series of (W, b) pairs to the input, with the
        provided signal function.
        Uses dimension of network and input to infer batching:
        any dimensions of network prior to its 2D weights matrix
        and 1D bias are assumed to correspond with the first
        dimensions of input, and then the network is batched over
        any additional dimensions of input preceding its final
        dimension. '''
    # batch inference
    net_instances = network[0][0].shape[:-2]
    instance_axes = 'itcnmopqrs'[:len(net_instances)]

    assert x.shape[:len(net_instances)] == net_instances
    batches = x.shape[len(net_instances):-1]
    batch_axes = 'itcnmopqrs'[len(net_instances):][:len(batches)]


    with tf.name_scope(name):
        next_input = x
        for layer, (W, b) in enumerate(network):
            for batch_axis in batch_axes:
                b = tf.expand_dims(b, len(net_instances), name=f'bias_l{layer}')
            einsum_notation = f'{instance_axes}xy,{instance_axes}{batch_axes}x->{instance_axes}{batch_axes}y'
            activation = tf.einsum(einsum_notation, W, next_input, name=f'matmul_l{layer}')
            activation = tf.add(activation, b, name=f'activation_l{layer}')
            next_input = signal_function(activation, name=f'signal_l{layer}')
        # pull the activation from the final layer, rather than the signal
    return final_function(activation, name=name)

@tf.function
def apply_critic_network(s, a, network, signal_function=tf.nn.relu, final_function=tf.identity, name='apply_network'):
    ''' Applies a series of (W, b) pairs to the input, with the
        provided signal function.
        Uses dimension of network and input to infer batching:
        any dimensions of network prior to its 2D weights matrix
        and 1D bias are assumed to correspond with the first
        dimensions of input, and then the network is batched over
        any additional dimensions of input preceding its final
        dimension.
        As a critic network, this accepts a proposed action
        as an input to the second layer. '''
    # batch inference
    net_instances = network[0][0].shape[:-2]
    instance_axes = 'itcnmopqrs'[:len(net_instances)]

    assert s.shape[:len(net_instances)] == net_instances
    batches = s.shape[len(net_instances):-1]
    batch_axes = 'itcnmopqrs'[len(net_instances):][:len(batches)]


    with tf.name_scope(name):
        next_input = s
        for layer, (W, b) in enumerate(network):
            if layer == 1:
                next_input = tf.concat([next_input, a], axis=-1)
            for batch_axis in batch_axes:
                b = tf.expand_dims(b, len(net_instances), name=f'bias_l{layer}')
            einsum_notation = f'{instance_axes}xy,{instance_axes}{batch_axes}x->{instance_axes}{batch_axes}y'
            activation = tf.einsum(einsum_notation, W, next_input, name=f'matmul_l{layer}')
            activation = tf.add(activation, b, name=f'activation_l{layer}')
            next_input = signal_function(activation, name=f'signal_l{layer}')
        # pull the activation from the final layer, rather than the signal
    return final_function(activation, name=name)

class ReplayBuffer:
    def __init__(self, evolution, obs_dim, action_dim, buffer_length, batch_size):
        self.ev = evolution
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_length = buffer_length
        self.batch_size = batch_size

        self.n_instances = self.ev.n_individuals * self.ev.n_tasks * self.ev.n_conditions
        self.instance_shape = [self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions]

    @tf.function
    def init(self):
        state_buffer = tf.zeros((self.n_instances, self.buffer_length, self.obs_dim))
        action_buffer = tf.zeros((self.n_instances, self.buffer_length, self.action_dim))
        reward_buffer = tf.zeros((self.n_instances, self.buffer_length))
        terminal_buffer = tf.zeros((self.n_instances, self.buffer_length), tf.bool)

        SA_pointer = tf.zeros((self.n_instances), tf.int32)
        RT_pointer = tf.zeros((self.n_instances), tf.int32)
        last_T = tf.ones((self.n_instances), tf.bool)
        return (state_buffer, action_buffer, reward_buffer, terminal_buffer, SA_pointer, RT_pointer, last_T)

    @tf.function
    def store(self, H, state, action, reward, terminal):
        # SA_pointer is one ahead of the last guaranteed S,A pair and RT_pointer
        # is one ahead of the last guaranteed R,T pair.
        # R[0] is the reward for the S[0]->S[1] transition.

        state_buffer, action_buffer, reward_buffer, terminal_buffer, SA_pointer, RT_pointer, last_T = H
        state = tf.reshape(state, (self.n_instances, self.obs_dim))
        action = tf.reshape(action, (self.n_instances, self.action_dim))
        reward = tf.reshape(reward, (self.n_instances,))
        terminal = tf.reshape(terminal, (self.n_instances,))

        individual_indices = tf.reshape(tf.range(self.n_instances, dtype=tf.int32), (-1, 1))
        SA_step_indices = tf.reshape(SA_pointer % self.buffer_length, (-1, 1))
        RT_step_indices = tf.reshape(RT_pointer % self.buffer_length, (-1, 1))
        augmented_SA_pointer = tf.concat([individual_indices, SA_step_indices], axis=1)
        augmented_RT_pointer = tf.concat([individual_indices, RT_step_indices], axis=1)

        state_buffer = tf.tensor_scatter_nd_update(state_buffer, augmented_SA_pointer, state)
        action_buffer = tf.tensor_scatter_nd_update(action_buffer, augmented_SA_pointer, action)
        reward_buffer = tf.tensor_scatter_nd_update(reward_buffer, augmented_RT_pointer, reward)
        terminal_buffer = tf.tensor_scatter_nd_update(terminal_buffer, augmented_RT_pointer, terminal)

        SA_pointer = tf.where(terminal, SA_pointer, SA_pointer + 1)
        RT_pointer = tf.where(last_T, RT_pointer, RT_pointer + 1)

        return state_buffer, action_buffer, reward_buffer, terminal_buffer, SA_pointer, RT_pointer, terminal

    @tf.function
    def sample(self, H):
        # Want a SARTS transition t->t+1 where t < the current RT_pointer
        # (RT_pointer is one ahead of the last guaranteed R,T pair)
        # If S(t+1) will be overwritten, it also won't be used, so this is safe.

        # Ideal strategy: if RT_pointer is less than buffer length, go for a t in [0, RT_pointer)
        #                 if RT_pointer is >= buffer length, go for a t in [RT_ptr - buffer_length + 1, RT_pointer)
        #                 and then mod it by buffer_length

        # What we can do: take random in [-buffer_length+1, -0), add it to RT_pointer,
        #                 mod by RT_pointer, and then mod by buffer_length.
        #      This would sometimes significantly bias in favor of more recent frames.

        state_buffer, action_buffer, reward_buffer, terminal_buffer, SA_pointer, RT_pointer, last_T = H

        deltas = self.ev.random.uniform((1, self.batch_size), minval=-self.buffer_length+1, maxval=0, dtype=tf.dtypes.int32)
        # these are same from gen to gen
        pointers = tf.reshape(RT_pointer, (-1, 1)) - deltas
        # sometimes things will be meaningless,
        # but no errors if we haven't filled enough of the buffer
        safe_modulos = tf.maximum(RT_pointer, 1)
        pointers %= tf.reshape(safe_modulos, (-1, 1))

        # guaranteed not to hit the current (not necessarily fixed) RT_pointer
        current_frame_pointers = pointers % self.buffer_length
        # the only time the next S is not certain is when the current T is True, so we won't use it
        next_frame_pointers = (pointers + 1) % self.buffer_length

        S = tf.gather(state_buffer, current_frame_pointers, batch_dims=1)
        A = tf.gather(action_buffer, current_frame_pointers, batch_dims=1)
        R = tf.gather(reward_buffer, current_frame_pointers, batch_dims=1)
        T = tf.gather(terminal_buffer, current_frame_pointers, batch_dims=1)
        S2 = tf.gather(state_buffer, next_frame_pointers, batch_dims=1)

        S = tf.reshape(S, self.instance_shape+[self.batch_size, -1])
        A = tf.reshape(A, self.instance_shape+[self.batch_size, -1])
        R = tf.reshape(R, self.instance_shape+[self.batch_size])
        T = tf.reshape(T, self.instance_shape+[self.batch_size])
        S2 = tf.reshape(S2, self.instance_shape+[self.batch_size, -1])

        return S, A, R, T, S2

class ConstantAgent:
    ''' An agent that always returns a constant action signal, parameterized directly by its genotype. '''
    USES_VARS = False

    def __init__(self, evolution, action_dim, init_noise):
        self.action_dim = action_dim
        self.init_noise = init_noise

        self.ev = evolution

    def generate_parameters(self):
        return (self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.action_dim))*self.init_noise,)

    def use_parameters(self, parameters):
        self.action_constant, = parameters

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)
        H = tuple()
        return H

    @tf.function
    def step(self, parameters, H, S, R, T, training):
        self.use_parameters(parameters)
        A_by_tasks_conds = tf.reshape(self.action_constant, (self.ev.n_individuals, 1, 1, self.action_dim))
        A_copied = tf.tile(A_by_tasks_conds, [1, self.ev.n_tasks, self.ev.n_conditions, 1])
        A = tf.clip_by_value(A_copied, -1, 1)
        print(A.shape)
        return H, A

    def set_vars(self, state):
        ''' Sets the state variables based on nested list of arrays state. '''
        pass

    def get_vars(self):
        ''' Returns a nested list of tensors that contains the current state variables. '''
        pass

class OnePolicyAgent:
    ''' A simple feed-forward neural network that uses a single policy for the duration
        of its lifespan. '''
    USES_VARS = False

    def __init__(self, evolution, obs_dim, action_dim, hidden_layers, init_noise, signal_function=tf.nn.relu, final_function=tf.nn.tanh, action_noise=0, action_noise_theta=0.15, noise_fixed=False):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.init_noise = init_noise
        self.signal_function = signal_function
        self.final_function = final_function

        self.action_noise = action_noise
        self.action_noise_theta = action_noise_theta
        if noise_fixed:
            self.noise_random = evolution.random
        else:
            self.noise_random = evolution.genotype_random


        self.ev = evolution

    def generate_parameters(self):
        architecture = [self.obs_dim] + self.hidden_layers + [self.action_dim]
        return generate_network((self.ev.n_individuals,), architecture, random=self.ev.genotype_random, noise=self.init_noise)

    def use_parameters(self, parameters):
        self.network = parameters

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)
        if self.action_noise == 0:
            # no hidden state for this network
            H = tuple()
        else:
            noise = tf.zeros((1, 1, self.ev.n_conditions, self.action_dim))
            H = (noise,)
            return H
        return H

    @tf.function
    def step(self, parameters, H, S, R, T, training):
        self.use_parameters(parameters)
        # neural network function of S
        A = apply_network(S, self.network, self.signal_function, self.final_function)
        if self.action_noise:
            noise, = H
            new_noise = self.noise_random.normal((1, 1, self.ev.n_conditions, self.action_dim)) * self.action_noise
            noise = noise - (self.action_noise_theta * noise * 0.01) + (0.1 * new_noise)

            A = tf.clip_by_value(A + noise, -1, 1)
            nextH = (noise,)
        else:
            nextH = tuple()
        return nextH, A

    def set_vars(self, state):
        ''' Sets the state variables based on nested list of arrays state. '''
        pass

    def get_vars(self):
        ''' Returns a nested list of tensors that contains the current state variables. '''
        pass

class DDPGAgent:
    ''' An actor/critic pair of feed-forward neural networks that use the DDPG algorithm
        to learn a Q-map and fit an actor. '''
    USES_VARS = True

    def __init__(self, evolution, obs_dim, action_dim, critic_hidden_layers, actor_hidden_layers, init_noise, tau=0.001, gamma=0.99, signal_function=tf.nn.relu, final_function=tf.nn.tanh, action_noise=0.2, action_noise_theta=0.15, actor_learning_rate=0.0001, critic_learning_rate=0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.signal_function = signal_function
        self.final_function = final_function
        self.init_noise = init_noise

        self.tau = tau
        self.gamma = gamma
        self.action_noise = action_noise
        self.action_noise_theta = action_noise_theta

        self.ev = evolution

        self.critic_architecture = [self.obs_dim+self.action_dim] + critic_hidden_layers + [1]
        self.critic_architecture = [self.obs_dim, (critic_hidden_layers[0], self.action_dim)] + critic_hidden_layers[1:] + [1]
        self.actor_architecture = [self.obs_dim] + actor_hidden_layers + [self.action_dim]

        self.replay = ReplayBuffer(self.ev, self.obs_dim, self.action_dim, 10000, 64)

        self.critic_network = network_vars((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions),
                                            self.critic_architecture, 'critic_network')
        self.actor_network = network_vars((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions),
                                            self.actor_architecture, 'actor_network')
        self.critic_target = network_vars((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions),
                                            self.critic_architecture, 'critic_target')
        self.actor_target = network_vars((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions),
                                            self.actor_architecture, 'actor_target')

        self.actor_optimizer = tf.optimizers.Adam(actor_learning_rate)
        self.critic_optimizer = tf.optimizers.Adam(critic_learning_rate)

        self.qmax = tf.Variable(tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions)))

        # this is necessary for optimizer saving / loading
        actor_params = list(itertools.chain(*self.actor_network))
        self.actor_optimizer.iterations
        self.actor_optimizer._create_slots(actor_params)
        self.default_actor_optimizer_weights = self.actor_optimizer.get_weights()

        critic_params = list(itertools.chain(*self.critic_network))
        self.critic_optimizer.iterations
        self.critic_optimizer._create_slots(critic_params)
        self.default_critic_optimizer_weights = self.critic_optimizer.get_weights()

    def generate_parameters(self):
        return tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions))

    def use_parameters(self, parameters):
        pass

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)

        for weight_var, weight_val in zip(self.actor_optimizer.weights, self.default_actor_optimizer_weights):
            weight_var.assign(weight_val)
        for weight_var, weight_val in zip(self.critic_optimizer.weights, self.default_critic_optimizer_weights):
            weight_var.assign(weight_val)

        if self.init_noise == 'Emami':
            critic_network = generate_network_Emami((self.ev.n_individuals,),
                                                self.critic_architecture, self.ev.random)
            actor_network = generate_network_Emami((self.ev.n_conditions,),
                                                self.actor_architecture, self.ev.random)
        elif self.init_noise == 'Lillicrap':
            critic_network = generate_network_Lillicrap((self.ev.n_individuals,),
                                                self.critic_architecture, self.ev.random)
            actor_network = generate_network_Lillicrap((self.ev.n_conditions,),
                                                self.actor_architecture, self.ev.random)
        else:
            critic_network = generate_network((self.ev.n_individuals,),
                                                self.critic_architecture, self.ev.random, self.init_noise)
            actor_network = generate_network((self.ev.n_conditions,),
                                                self.actor_architecture, self.ev.random, self.init_noise)
        critic_network = duplicate_network(critic_network, copies=self.ev.n_tasks, axis=1)
        critic_network = duplicate_network(critic_network, copies=self.ev.n_conditions, axis=2)
        actor_network = duplicate_network(actor_network, copies=self.ev.n_tasks, axis=0)
        actor_network = duplicate_network(actor_network, copies=self.ev.n_individuals, axis=0)
        assign_network(self.critic_network, critic_network)
        assign_network(self.critic_target, critic_network)
        assign_network(self.actor_network, actor_network)
        assign_network(self.actor_target, actor_network)

        replay_H = self.replay.init()
        noise = tf.zeros((1, 1, self.ev.n_conditions, self.action_dim))
        H = (replay_H, noise)
        return H

    def update_target(self):
        for (W, b), (Wt, bt) in zip(itertools.chain(self.actor_network, self.critic_network),
                                    itertools.chain(self.actor_target, self.critic_target)):
            Wt.assign(W * self.tau + Wt * (1 - self.tau))
            bt.assign(b * self.tau + bt * (1 - self.tau))

    @tf.function
    def step(self, parameters, H, S, R, T, training):
        self.use_parameters(parameters)
        replay_H, noise = H

        A = apply_network(S, self.actor_network, self.signal_function, final_function=self.final_function)

        if training:
            new_noise = self.ev.random.normal((1, 1, self.ev.n_conditions, self.action_dim)) * self.action_noise
            noise = noise - (self.action_noise_theta * noise * 0.01) + (0.1 * new_noise)

            A = tf.clip_by_value(A + noise, -1, 1)
            replay_H = self.replay.store(replay_H, S, A, R, T)

            self.train_on(self.replay.sample(replay_H))

        nextH = (replay_H, noise)
        return nextH, A

    def train_on(self, SARTS):
        state, action, reward, terminal, next_state = SARTS

        # train critic to predict the ideal Q values
        next_action = apply_network(next_state, self.actor_target, name='next_action')
        if type(self.critic_architecture[1]) == tuple:
            next_q_value = apply_critic_network(next_state, next_action, self.critic_target, final_function=tf.identity, name='next_q_value')
        else:
            next_q_value = apply_network(tf.concat((next_state, next_action), axis=-1), self.critic_target, final_function=tf.identity, name='next_q_value')
        next_q_value = tf.squeeze(next_q_value, axis=-1)
        Bellman_equation = reward + self.gamma * next_q_value
        target_q_value = tf.where(terminal, reward, Bellman_equation, name='target_q_value')

        if type(self.critic_architecture[1]) == tuple:
            predicted_q_value = apply_critic_network(state, action, self.critic_network, final_function=tf.identity, name='predicted_q_value')
        else:
            predicted_q_value = apply_network(tf.concat((state, action), axis=-1), self.critic_network, final_function=tf.identity, name='predicted_q_value')
        predicted_q_value = tf.squeeze(predicted_q_value, axis=-1)
        self.qmax.assign_add(tf.reduce_max(predicted_q_value, axis=-1))

        critic_squared_error = tf.square(target_q_value - predicted_q_value)
        critic_losses = tf.reduce_mean(critic_squared_error, axis=-1)

        simultaneous_critic_loss = tf.reduce_sum(critic_losses)

        critic_params = list(itertools.chain(*self.critic_network))
        critic_grads = tf.gradients(simultaneous_critic_loss, critic_params)

        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_params))


        # train actor to maximize the critic's evaluation
        new_action = apply_network(state, self.actor_network, self.signal_function, final_function=self.final_function, name='new_action')
        if type(self.critic_architecture[1]) == tuple:
            new_q_value = apply_critic_network(next_state, new_action, self.critic_network, final_function=tf.identity, name='new_q_value')
        else:
            new_q_value = apply_network(tf.concat((next_state, new_action), axis=-1), self.critic_network, final_function=tf.identity, name='new_q_value')
        new_q_value = tf.squeeze(new_q_value, axis=-1)
        aptitude = tf.reduce_mean(new_q_value, axis=-1)
        simultaneous_aptitude = tf.reduce_sum(aptitude)

        actor_params = list(itertools.chain(*self.actor_network))
        actor_grads = tf.gradients(-simultaneous_aptitude, actor_params)
        #tf.print(tf.reduce_mean(tf.stack([tf.reduce_mean(i**2) for i in actor_grads])))

        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

        self.update_target()

    def set_vars(self, state):
        ''' Returns a tensorflow op to set the state variables based on nested list of arrays state. '''
        actor_network, actor_target, actor_optimizer_weights, critic_network, critic_target, critic_optimizer_weights = state
        assign_network(self.critic_network, critic_network)
        assign_network(self.critic_target, critic_target)
        self.critic_optimizer.set_weights(critic_optimizer_weights)
        assign_network(self.actor_network, actor_network)
        assign_network(self.actor_target, actor_target)
        self.actor_optimizer.set_weights(actor_optimizer_weights)

    def get_vars(self):
        ''' Returns a nested list of tensors that contains the current state variables. '''
        critic_network = read_network(self.critic_network)
        critic_target = read_network(self.critic_target)
        critic_optimizer_weights = self.critic_optimizer.get_weights()
        actor_network = read_network(self.actor_network)
        actor_target = read_network(self.actor_target)
        actor_optimizer_weights = self.actor_optimizer.get_weights()

        state = (actor_network, actor_target, actor_optimizer_weights, critic_network, critic_target, critic_optimizer_weights)
        return state

class eQAgent:
    ''' An actor/critic pair of feed-forward neural networks where the Q-map is evolved
        and the actor is fitted to the Q-map as in DDPG. '''
    USES_VARS = True

    def __init__(self, evolution, obs_dim, action_dim, critic_hidden_layers, actor_hidden_layers, init_noise, actor_optimizer=tf.optimizers.Adam, actor_learning_rate=0.0001, signal_function=tf.nn.relu, final_function=tf.nn.tanh, action_noise=0.2, action_noise_theta=0.15, noise_fixed=False, actor_fixed=True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.signal_function = signal_function
        self.final_function = final_function
        self.init_noise = init_noise

        self.ev = evolution

        self.action_noise = action_noise
        self.action_noise_theta = action_noise_theta
        if noise_fixed:
            self.noise_random = self.ev.random
        else:
            self.noise_random = self.ev.genotype_random

        if actor_fixed:
            self.actor_random = self.ev.random
        else:
            self.actor_random = self.ev.genotype_random

        #self.critic_architecture = [self.obs_dim+self.action_dim] + critic_hidden_layers + [1]
        self.critic_architecture = [self.obs_dim, (critic_hidden_layers[0], self.action_dim)] + critic_hidden_layers[1:] + [1]
        self.actor_architecture = [self.obs_dim] + actor_hidden_layers + [self.action_dim]

        self.replay = ReplayBuffer(self.ev, self.obs_dim, self.action_dim, 10000, 64)

        self.actor_network = network_vars((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions),
                                            self.actor_architecture, 'actor_network')

        self.actor_optimizer = actor_optimizer(actor_learning_rate)

        # this is necessary for optimizer saving / loading
        actor_params = list(itertools.chain(*self.actor_network))
        self.actor_optimizer.iterations
        self.actor_optimizer._create_slots(actor_params)

        self.default_actor_optimizer_weights = self.actor_optimizer.get_weights()

    def generate_parameters(self):
        if self.init_noise == 'Emami':
            critic_network = generate_network_Emami((self.ev.n_individuals,),
                                                self.critic_architecture, self.ev.genotype_random)
        elif self.init_noise == 'Lillicrap':
            critic_network = generate_network_Lillicrap((self.ev.n_individuals,),
                                                self.critic_architecture, self.ev.genotype_random)
        else:
            critic_network = generate_network((self.ev.n_individuals,),
                                                self.critic_architecture, self.ev.genotype_random, self.init_noise)
        return critic_network,

    def use_parameters(self, parameters):
        critic_network, = parameters
        self.critic_network = critic_network

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)

        for weight_var, weight_val in zip(self.actor_optimizer.weights, self.default_actor_optimizer_weights):
            weight_var.assign(weight_val)

        if self.init_noise == 'Emami':
            actor_network = generate_network_Emami((self.ev.n_conditions,),
                                                self.actor_architecture, self.actor_random)
        elif self.init_noise == 'Lillicrap':
            actor_network = generate_network_Lillicrap((self.ev.n_conditions,),
                                                self.actor_architecture, self.actor_random)
        else:
            actor_network = generate_network((self.ev.n_conditions),
                                                self.actor_architecture, self.actor_random, self.init_noise)

        actor_network = duplicate_network(actor_network, copies=self.ev.n_tasks, axis=0)
        actor_network = duplicate_network(actor_network, copies=self.ev.n_individuals, axis=0)
        assign_network(self.actor_network, actor_network)

        replay_H = self.replay.init()
        noise = tf.zeros((1, 1, self.ev.n_conditions, self.action_dim))
        H = (replay_H, noise)
        return H

    @tf.function
    def step(self, parameters, H, S, R, T, training):
        self.use_parameters(parameters)
        replay_H, noise = H

        A = apply_network(S, self.actor_network, self.signal_function, final_function=self.final_function)

        if training:
            if self.action_noise:
                new_noise = self.noise_random.normal((1, 1, self.ev.n_conditions, self.action_dim)) * self.action_noise
                noise = noise - (self.action_noise_theta * noise * 0.01) + (0.1 * new_noise)

                A = tf.clip_by_value(A + noise, -1, 1)
            replay_H = self.replay.store(replay_H, S, A, R, T)

            self.train_on(self.replay.sample(replay_H))

        nextH = (replay_H, noise)
        return nextH, A

    def train_on(self, SARTS):
        state, action, reward, terminal, next_state = SARTS

        # train actor to maximize the critic's evaluation
        new_action = apply_network(state, self.actor_network, self.signal_function, final_function=self.final_function, name='new_action')
        if type(self.critic_architecture[1]) == tuple:
            new_q_value = apply_critic_network(state, new_action, self.critic_network, final_function=tf.identity, name='new_q_value')
        else:
            new_q_value = apply_network(tf.concat((state, new_action), axis=-1), self.critic_network, final_function=tf.identity, name='new_q_value')
        aptitude = tf.reduce_mean(new_q_value, axis=tf.range(1, new_q_value.shape.rank))
        simultaneous_aptitude = tf.reduce_sum(aptitude, axis=0)

        # if you want to see the action gradients, they are here:
        # action_grads = tf.gradients(-simultaneous_aptitude, new_action)

        actor_params = list(itertools.chain(*self.actor_network))
        actor_grads = tf.gradients(-simultaneous_aptitude, actor_params)

        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))

    def set_vars(self, state):
        ''' Returns a tensorflow op to set the state variables based on nested list of arrays state. '''
        actor_network, actor_optimizer_weights = state
        assign_network(self.actor_network, actor_network)
        self.actor_optimizer.set_weights(actor_optimizer_weights)

    def get_vars(self):
        ''' Returns a nested list of tensors that contains the current state variables. '''
        actor_network = read_network(self.actor_network)
        actor_optimizer_weights = self.actor_optimizer.get_weights()

        state = (actor_network, actor_optimizer_weights)
        return state

class CTRNNAgent:
    ''' A continuous-time neural network agent. Not currently production-level because it is capable of
        evolving numerically unstable time-constants. Set min_time_constant to a value at LEAST your timestep.
        Based on http://www.cs.uvm.edu/~jbongard/2014_CS206/Beer_CTRNNs.pdf '''
    USES_VARS = False

    def __init__(self, evolution, obs_dim, action_dim, hidden_dim, init_noise, signal_function=tf.nn.sigmoid, final_function=tf.nn.tanh, min_time_constant=0.01, mean_time_constant=0.2, activity_noise=0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.init_noise = init_noise
        self.signal_function = signal_function
        self.final_function = final_function
        self.min_time_constant = min_time_constant
        self.mean_time_constant = mean_time_constant
        self.tc_exp = np.log(self.mean_time_constant - self.min_time_constant)

        self.activity_noise = activity_noise

        self.ev = evolution

    def generate_parameters(self):
        sensory_time_constant = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.obs_dim), stddev=self.init_noise)
        hidden_time_constant = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.hidden_dim), stddev=self.init_noise)
        motor_time_constant = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.action_dim), stddev=self.init_noise)

        sensory_output_bias = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.obs_dim), stddev=self.init_noise)
        hidden_output_bias = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.hidden_dim), stddev=self.init_noise)

        sensorihidden_weights = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.obs_dim, self.hidden_dim), stddev=self.init_noise)
        hiddenhidden_weights = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.hidden_dim, self.hidden_dim), stddev=self.init_noise)
        sensorimotor_weights = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.obs_dim, self.action_dim), stddev=self.init_noise)
        hiddenmotor_weights = self.ev.genotype_random.truncated_normal((self.ev.n_individuals, self.hidden_dim, self.action_dim), stddev=self.init_noise)

        return (sensory_time_constant, hidden_time_constant, motor_time_constant), (sensory_output_bias, hidden_output_bias), (sensorihidden_weights, hiddenhidden_weights, sensorimotor_weights, hiddenmotor_weights)

    def use_parameters(self, parameters):
        time_constants, biases, weights = parameters
        broadcasted_time_constants = (tf.reshape(self.min_time_constant+tf.math.exp(term + self.tc_exp), (self.ev.n_individuals, 1, 1, -1)) for term in time_constants)
        broadcasted_biases = (tf.reshape(term, (self.ev.n_individuals, 1, 1, -1)) for term in biases)

        self.sensory_time_constant, self.hidden_time_constant, self.motor_time_constant = broadcasted_time_constants
        self.sensory_output_bias, self.hidden_output_bias = broadcasted_biases

        self.sensorihidden_weights, self.hiddenhidden_weights, self.sensorimotor_weights, self.hiddenmotor_weights = weights

    @tf.function
    def init(self, parameters):
        self.use_parameters(parameters)
        sensory_state = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.obs_dim))
        hidden_state = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.hidden_dim))
        motor_state = tf.zeros((self.ev.n_individuals, self.ev.n_tasks, self.ev.n_conditions, self.action_dim))
        network_state = (sensory_state, hidden_state, motor_state)
        return network_state

    @tf.function
    def step(self, parameters, H, S, R, T, training):
        self.use_parameters(parameters)
        sensory_state, hidden_state, motor_state = H
        # neural network function of S
        sensory_output = self.signal_function(sensory_state + self.sensory_output_bias)
        hidden_output = self.signal_function(hidden_state + self.hidden_output_bias)
        sensory_target = S
        hidden_target = tf.einsum('itcx,ixy->itcy', sensory_output, self.sensorihidden_weights) + tf.einsum('itcx,ixy->itcy', hidden_output, self.hiddenhidden_weights)
        motor_target = tf.einsum('itcx,ixy->itcy', sensory_output, self.sensorimotor_weights) + tf.einsum('itcx,ixy->itcy', hidden_output, self.hiddenmotor_weights)

        dsensory_dt = (sensory_target - sensory_state)/self.sensory_time_constant
        dhidden_dt = (hidden_target - hidden_state)/self.hidden_time_constant
        dmotor_dt = (motor_target - motor_state)/self.motor_time_constant

        next_sensory = self.ev.dt*dsensory_dt + sensory_state
        next_hidden = self.ev.dt*dhidden_dt + hidden_state
        next_motor = self.ev.dt*dmotor_dt + motor_state

        action = self.final_function(next_motor)

        return (next_sensory, next_hidden, next_motor), action

    def set_vars(self, state):
        ''' Sets the state variables based on nested list of arrays state. '''
        pass

    def get_vars(self):
        ''' Returns a nested list of tensors that contains the current state variables. '''
        pass

