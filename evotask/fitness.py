''' A number of fitness functions designed to measure agents' performance, potentially
    across multiple tasks. '''

import tensorflow as tf
import numpy as np

def direct_agent_fitness(training_rewards, testing_rewards):
    return tf.reduce_mean(testing_rewards, axis=[1,2,3])

def standard_agent_fitness(training_rewards, testing_rewards):
    ''' Normalize over tasks so that being unusually
        good at each is equally important. '''
    agent_task = tf.reduce_mean(testing_rewards, axis=[2,3])
    agent_task -= tf.math.reduce_mean(agent_task, axis=0, keepdims=True)
    agent_task /= tf.math.reduce_std(agent_task, axis=0, keepdims=True)
    return tf.reduce_mean(agent_task, axis=1)


def standard_env_fitness(training_rewards, testing_rewards):
    ''' Adversarial approach to task fitness. '''
    agent_task = tf.reduce_mean(testing_rewards, axis=[2,3])
    agent_task -= tf.math.reduce_mean(agent_task, axis=1, keepdims=True)
    agent_task /= tf.math.reduce_std(agent_task, axis=1, keepdims=True)

    return -tf.reduce_mean(agent_task, axis=0)
