import collections
import logging
import math
import numpy as np
import operator
import os
import tensorflow as tf


# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


# Simple namedtuple to create the different variables in the actor and critic network.
Variable = collections.namedtuple('Variable', ['tensor', 'regularize', 'copy_as_is'])


class Model(object):

  def __init__(self, action_shape, observation_shape, checkpoint_directory, options, restore=False):
    self.session = tf.Session()
    self.action_shape = action_shape
    self.observation_shape = observation_shape
    # Create networks.
    self.options = options
    if not self.options.layer_size:
      self.options.layer_size.extend([400, 300])
    self.Create()
    # Create saver.
    self.saver = tf.train.Saver(max_to_keep=1)
    self.checkpoint_directory = checkpoint_directory
    checkpoint = tf.train.latest_checkpoint(checkpoint_directory)
    if checkpoint and restore:
      LOG.info('Restoring from previous checkpoint: %s', checkpoint)
      self.saver.restore(self.session, checkpoint)
    else:
      tf.initialize_all_variables().run(session=self.session)  # To be replaced with global_variables_initializer.
    tf.train.SummaryWriter(checkpoint_directory, self.session.graph)
    self.session.graph.finalize()

  def __del__(self):
    self.session.close()

  def Save(self, step):
    return self.saver.save(self.session, os.path.join(self.checkpoint_directory, 'model.ckpt'), global_step=step)

  def Create(self):
    # Contains all operations to reset the networks when an episode starts.
    self.reset_ops = []

    with tf.device(self.options.device):
      # Create parameters (both for the regular and target networks).
      parameters_actor = self.ActorNetworkParameters(name='actor_parameters')
      parameters_critic = self.CriticNetworkParameters(name='critic_parameters')
      parameters_target_actor, update_target_actor = PropagateToTargetNetwork(
          parameters_actor, 1. - self.options.tau, name='target_actor_parameters')
      parameters_target_critic, update_target_critic = PropagateToTargetNetwork(
          parameters_critic, 1. - self.options.tau, name='target_critic_parameters')

      # Non-trainable actor network.
      single_input_observation = tf.placeholder(tf.float32, shape=(1,) + self.observation_shape)
      single_action = self.ActorNetwork(single_input_observation, parameters_actor, name='single_actor')
      # Add exploration (using Ornstein-Uhlenbeck process) - only used when a single action is given.
      action_noise, reset_op = OrnsteinUhlenbeckProcess(
          (1,) + self.action_shape, self.options.exploration_noise_theta, self.options.exploration_noise_sigma)
      self.reset_ops.append(reset_op)
      noisy_action = single_action + action_noise
      # The model is really only insterested in actions between -1 and 1. Hence, we mirror actions if they
      # saturate over these limits.
      # TODO: Investigate the next 5 lines.
      upper_bound = 1.1  # Add some margins so that saturated actions are explored more often.
      lower_bound = -1.1
      noisy_action = upper_bound - tf.abs(upper_bound - noisy_action)  # Mirror upper-bound.
      noisy_action = lower_bound + tf.abs(lower_bound - noisy_action)  # Mirror lower-bound.
      noisy_action = tf.maximum(tf.minimum(noisy_action, 1.), -1.)  # Saturate just in case.

      # Training actor.
      input_observation = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape)
      action = self.ActorNetwork(input_observation, parameters_actor, is_training=True, name='actor')
      with tf.variable_scope('actor_loss'):
        q_value = self.CriticNetwork(action, input_observation, parameters_critic, name='fixed_critic')
        loss = -tf.reduce_mean(q_value, 0)  # Maximize Q-value.
        optimizer = tf.train.AdamOptimizer(learning_rate=self.options.actor_learning_rate)
        # Only update parameters of the actor network.
        gradients = optimizer.compute_gradients(loss, var_list=[p.tensor for p in parameters_actor])
        train_op = optimizer.apply_gradients(gradients)
        # Training the actor include propagating the learned parameters.
        with tf.control_dependencies([train_op]):
          train_actor = tf.group(update_target_actor)

      # Critic network.
      input_action = tf.placeholder(tf.float32, shape=(None,) + self.action_shape)
      input_reward = tf.placeholder(tf.float32, shape=(None,))
      input_done = tf.placeholder(tf.bool, shape=(None,))
      input_next_observation = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape)
      input_weight = tf.placeholder(tf.float32, shape=(None,))
      q_value = self.CriticNetwork(input_action, input_observation, parameters_critic, is_training=True, name='critic')

      with tf.variable_scope('critic_loss'):
        q_value_target = self.CriticNetwork(
            self.ActorNetwork(input_next_observation, parameters_target_actor, name='target_actor'),
            input_next_observation, parameters_target_critic, name='target_critic')
        q_value_target = tf.stop_gradient(  # Gradient are not propagated to the target networks.
            tf.select(input_done, input_reward, input_reward + self.options.discount_factor * q_value_target))
        # Training critic.
        td_error = q_value - q_value_target
        loss = tf.reduce_sum(input_weight * tf.square(td_error), 0)  # Minimize weighted TD-error.
        loss += tf.add_n([self.options.critic_weight_decay * tf.nn.l2_loss(p.tensor) for p in parameters_critic if p.regularize])
        optimizer = tf.train.AdamOptimizer(learning_rate=self.options.critic_learning_rate)
        gradients = optimizer.compute_gradients(loss, var_list=[p.tensor for p in parameters_critic])
        train_op = optimizer.apply_gradients(gradients)
        # Training the critic include propagating the learned parameters.
        with tf.control_dependencies([train_op]):
          train_critic = tf.group(update_target_critic)

      # Create relevant functions.
      with self.session.as_default():
        self.Reset = WrapComputationalGraph([], self.reset_ops)
        self._NoisyAct = WrapComputationalGraph(single_input_observation, noisy_action)
        self._Act = WrapComputationalGraph(single_input_observation, single_action)
        self.Train = WrapComputationalGraph(
            [input_action, input_observation, input_reward, input_done, input_next_observation, input_weight],
            [train_actor, train_critic, td_error], return_only=2)

  def Act(self, observation, add_noise=False):
    observation = np.expand_dims(observation, axis=0)
    action = self._NoisyAct(observation) if add_noise else self._Act(observation)
    return action[0, ...]

  def ActorNetworkParameters(self, name='actor'):
    params = []
    with tf.variable_scope(name):
      # Input is flattened.
      previous_size = reduce(operator.mul, self.observation_shape, 1)
      if self.options.use_actor_batch_normalization:
        # This is not strictly needed but helps given that the weights of the next layer are
        # always initialized with the same variance.
        params.extend(BatchNormalizationParameters((previous_size,), scale=False, center=False))
      # Layers.
      for i, layer_size in enumerate(self.options.layer_size):
        with tf.variable_scope('layer_%d' % i):
          initializer = tf.random_uniform_initializer(minval=-1.0 / math.sqrt(previous_size),
                                                      maxval=1.0 / math.sqrt(previous_size))
          w = tf.get_variable('w', (previous_size, layer_size), initializer=initializer)
          b = tf.get_variable('b', (layer_size,), initializer=initializer)
          params.extend([Variable(w, regularize=False, copy_as_is=False),
                         Variable(b, regularize=False, copy_as_is=False)])
          if self.options.use_actor_batch_normalization:
            params.extend(BatchNormalizationParameters((layer_size,), scale=False))
          previous_size = layer_size
      # Output action.
      with tf.variable_scope('output'):
        output_size = reduce(operator.mul, self.action_shape, 1)
        initializer = tf.random_uniform_initializer(minval=-self.options.initialization_range_last_layer, maxval=self.options.initialization_range_last_layer)
        w = tf.get_variable('w', (previous_size, output_size), initializer=initializer)
        b = tf.get_variable('b', (output_size,), initializer=initializer)
        params.extend((Variable(w, regularize=False, copy_as_is=False),
                       Variable(b, regularize=False, copy_as_is=False)))
        if self.options.use_actor_batch_normalization:
          params.extend(BatchNormalizationParameters((output_size,), scale=False))
        return params

  def ActorNetwork(self, input_observation, params, is_training=False, name='actor'):
    index = 0
    with tf.variable_scope(name):
      # Input is flattened.
      flat_input_observation = tf.contrib.layers.flatten(input_observation)
      previous_input = flat_input_observation
      if self.options.use_actor_batch_normalization:
        bn = params[index: index + 4]
        index += 4
        previous_input = BatchNormalization(previous_input, bn, is_training=is_training,
                                            decay=self.options.batch_normalization_decay)
      # Layers.
      for i, layer_size in enumerate(self.options.layer_size):
        with tf.variable_scope('layer_%d' % i):
          w = params[index].tensor
          b = params[index + 1].tensor
          index += 2
          previous_input = tf.nn.xw_plus_b(previous_input, w, b)
          if self.options.use_actor_batch_normalization:
            bn = params[index: index + 4]
            index += 4
            previous_input = BatchNormalization(previous_input, bn, is_training=is_training,
                                                decay=self.options.batch_normalization_decay)
          previous_input = tf.nn.relu(previous_input)
      # Output action.
      with tf.variable_scope('output'):
        w = params[index].tensor
        b = params[index + 1].tensor
        index += 2
        # TODO: Reshape to requested shape.
        previous_input = tf.nn.xw_plus_b(previous_input, w, b)
        if self.options.use_actor_batch_normalization:
          bn = params[index: index + 4]
          index += 4
          previous_input = BatchNormalization(previous_input, bn, is_training=is_training,
                                              decay=self.options.batch_normalization_decay)
        return tf.nn.tanh(previous_input)

  def CriticNetworkParameters(self, name='critic'):
    params = []
    with tf.variable_scope(name):
      # Input is flattened.
      previous_size = reduce(operator.mul, self.observation_shape, 1)
      if self.options.use_critic_batch_normalization:
        params.extend(BatchNormalizationParameters((previous_size,), scale=False, center=False))
      # Layers.
      for i, layer_size in enumerate(self.options.layer_size):
        with tf.variable_scope('layer_%d' % i):
          if i == self.options.critic_action_inserted_at_layer:
            # Input action in the second layer.
            previous_size += reduce(operator.mul, self.action_shape, 1)
          initializer = tf.random_uniform_initializer(minval=-1.0 / math.sqrt(previous_size),
                                                      maxval=1.0 / math.sqrt(previous_size))
          w = tf.get_variable('w', (previous_size, layer_size), initializer=initializer)
          b = tf.get_variable('b', (layer_size,), initializer=initializer)
          params.extend([Variable(w, regularize=True, copy_as_is=False),
                         Variable(b, regularize=True, copy_as_is=False)])
          # The original paper batch normalizes each layer until the layer that
          # includes the action (exluded). I found that the training behaved
          # more consistently by only batch normalizing the observation input.
          previous_size = layer_size
      # Output q-value.
      with tf.variable_scope('output'):
        output_size = 1
        initializer = tf.random_uniform_initializer(minval=-self.options.initialization_range_last_layer, maxval=self.options.initialization_range_last_layer)
        w = tf.get_variable('w', (previous_size, output_size), initializer=initializer)
        b = tf.get_variable('b', (output_size,), initializer=initializer)
        params.extend((Variable(w, regularize=True, copy_as_is=False),
                       Variable(b, regularize=True, copy_as_is=False)))
        return params

  def CriticNetwork(self, input_action, input_observation, params, is_training=False, name='critic'):
    index = 0
    with tf.variable_scope(name):
      # Input is flattened.
      flat_input_observation = tf.contrib.layers.flatten(input_observation)
      previous_input = flat_input_observation
      if self.options.use_critic_batch_normalization:
        bn = params[index: index + 4]
        index += 4
        previous_input = BatchNormalization(previous_input, bn, is_training=is_training,
                                            decay=self.options.batch_normalization_decay)
      # Layers.
      for i, layer_size in enumerate(self.options.layer_size):
        with tf.variable_scope('layer_%d' % i):
          if i == 1:
            # Input action in the second layer.
            flat_input_action = tf.contrib.layers.flatten(input_action)
            previous_input = tf.concat(1, [previous_input, flat_input_action])
          w = params[index].tensor
          b = params[index + 1].tensor
          index += 2
          previous_input = tf.nn.xw_plus_b(previous_input, w, b)
          previous_input = tf.nn.relu(previous_input)
      # Output q-value.
      with tf.variable_scope('output'):
        w = params[index].tensor
        b = params[index + 1].tensor
        index += 2
        return tf.squeeze(tf.nn.xw_plus_b(previous_input, w, b))


class WrapComputationalGraph(object):
  def __init__(self, inputs, outputs, return_only=None, session=None):
    self._inputs = inputs if isinstance(inputs, list) else [inputs]
    self._outputs = outputs if isinstance(outputs, list) else [outputs]
    assert self._outputs
    self._session = session or tf.get_default_session()
    self._return_only = return_only or range(len(self._outputs))
    self._return_only = self._return_only if isinstance(self._return_only, list) else [self._return_only]

  def __call__(self, *args):
    feed_dict = {}
    for (argpos, arg) in enumerate(args):
      feed_dict[self._inputs[argpos]] = arg
    outputs = self._session.run(self._outputs, feed_dict)
    final_outputs = []
    for i in self._return_only:
      final_outputs.append(outputs[i])
    return final_outputs if len(final_outputs) > 1 else final_outputs[0]


def PropagateToTargetNetwork(params, decay=0.99, name='moving_average'):
  ema = tf.train.ExponentialMovingAverage(decay=decay, name=name)
  tensors_for_ema = [p.tensor for p in params if not p.copy_as_is]
  op = ema.apply(tensors_for_ema)
  target_params = []
  for p in params:
    if p.copy_as_is:
      target_params.append(p)
    else:
      target_params.append(Variable(ema.average(p.tensor), regularize=None, copy_as_is=None))
  return target_params, op


def OrnsteinUhlenbeckProcess(shape, theta, sigma):
  with tf.variable_scope('output'):
    initial_noise = tf.zeros(shape)
    noise = tf.get_variable('noise', shape, initializer=tf.constant_initializer(0.), trainable=False)
    reset_op = noise.assign(initial_noise)
    return noise.assign_sub(theta * noise - tf.random_normal(shape, stddev=sigma)), reset_op


# We define our own batch normalization layer so it is easier to construct it from parameters.
def BatchNormalizationParameters(shape, center=True, scale=True, name=None):
  with tf.variable_scope(name or 'bn'):
    beta = tf.get_variable('beta', shape, initializer=tf.constant_initializer(0.), trainable=center)
    gamma = tf.get_variable('gamma', shape, initializer=tf.constant_initializer(1.), trainable=scale)
    # Moving averages are stored in these variables.
    average_mean = tf.get_variable('mean', shape, initializer=tf.constant_initializer(0.), trainable=False)
    average_var = tf.get_variable('var', shape, initializer=tf.constant_initializer(1.), trainable=False)
  return [Variable(beta, regularize=False, copy_as_is=False),
          Variable(gamma, regularize=False, copy_as_is=False),
          Variable(average_mean, regularize=False, copy_as_is=True),  # Make an exact copy.
          Variable(average_var, regularize=False, copy_as_is=True)]


def BatchNormalization(input_tensor, params, decay=0.99,
                       epsilon=1e-3, is_training=False, name=None):
  with tf.variable_scope('bn'):
    beta, gamma, average_mean, average_var = params
    if is_training:
      # Create moving average and store in average_mean and average_var.
      ops = []
      batch_mean, batch_var = tf.nn.moments(input_tensor, [0], name='moments')
      ops.append(average_mean.tensor.assign_sub((1. - decay) * (average_mean.tensor - batch_mean)))
      ops.append(average_var.tensor.assign_sub((1. - decay) * (average_var.tensor - batch_var)))
      with tf.control_dependencies(ops):
        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
    else:
      mean, var = average_mean.tensor, average_var.tensor
    output_tensor = tf.nn.batch_normalization(input_tensor, mean, var, beta.tensor, gamma.tensor, epsilon)
  return output_tensor
