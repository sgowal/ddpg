import math
import operator
import tensorflow as tf

_LEARNING_RATE_POLICY = 1e-4
_LEARNING_RATE_VALUE = 1e-3
_L2_WEIGHT_DECAY = 1e-2  # Only for Q.
_LAYERS = [400, 300]
_LAST_LAYER_INIT = 3e-3  # Other layers are initialize with 1/sqrt(F)
_DISCOUNT_FACTOR = .99
_EXPLORATION_NOISE_THETA = 0.1  # Ornstein-Uhlenbeck process.
_EXPLORATION_NOISE_SIGMA = 0.2
_TAU = 1e-3  # Leaky-integrator for parameters.


class Model(object):

  def __init__(self, action_shape, observation_shape):
    self.session = tf.Session()
    self.action_shape = action_shape
    self.observation_shape = observation_shape
    # Create networks.
    self.Create()

  def __del__(self):
    self.session.close()

  def Create(self):
    # Contains all operations to reset the networks when an episode starts.
    self.reset_ops = []

    # Create parameters (both for the regular and target networks).
    parameters_actor = self.ActorNetworkParameters()
    parameters_critic = self.CriticNetworkParameters()
    parameters_target_actor, update_target_actor = ExponentialMovingAverage(parameters_actor, 1. - _TAU)
    parameters_target_critic, update_target_critic = ExponentialMovingAverage(parameters_critic, 1. - _TAU)

    # Actor network.
    input_observation = tf.placeholder(tf.float32, shape=(None,) + self.observation_shape)
    action = self.ActorNetwork(input_observation, parameters_actor)

    # Add exploration (using Ornstein-Uhlenbeck process) - only used when a single action is given.
    action_noise, reset_op = OrnsteinUhlenbeckProcess(
        (1,) + self.action_shape, _EXPLORATION_NOISE_THETA, _EXPLORATION_NOISE_SIGMA)
    self.reset_ops.append(reset_op)
    noisy_action = action + action_noise

    # Training actor.
    q_value = self.CriticNetwork(action, input_observation, parameters_critic)
    loss = -tf.reduce_mean(q_value, 0)  # Maximize Q-value.
    optimizer = tf.train.AdamOptimizer(learning_rate=_LEARNING_RATE_POLICY)
    # Only update parameters of the actor network.
    gradients = optimizer.compute_gradients(loss, var_list=parameters_actor)
    train_actor = optimizer.apply_gradients(gradients)
    # Training the actor include propagating the learned parameters.
    with tf.control_dependencies([train_actor]):
      train_actor = update_target_actor

    # Training critic.
    input_action = tf.placeholder(tf.float32, shape=(None,) + self.action_shape)
    input_reward = tf.placeholder(tf.float32, shape=(None,))
    input_done = tf.placeholder(tf.float32, shape=(None,))
    # input_reward = tf.placeholder(tf.float32, shape=(None,))

    # TODO TODO

    # Create relevant functions.
    self.Reset = WrapComputationalGraph(None, self.reset_ops)



  def ActorNetworkParameters(self, device='/cpu:0'):
    params = []
    with tf.device(device):
      with tf.variable_scope('actor'):
        # Input is flattened.
        previous_size = reduce(operator.mul, self.observation_shape, 1)
        # Layers.
        for i, layer_size in enumerate(_LAYERS):
          with tf.variable_scope('layer_%d' % i):
            initializer = tf.random_uniform_initializer(minval=1.0 / math.sqrt(previous_size),
                                                        maxval=1.0 / math.sqrt(previous_size))
            w = tf.get_variable('w', (previous_size, layer_size), initializer=initializer)
            b = tf.get_variable('b', (layer_size,), initializer=initializer)
            params.extend((w, b))
            previous_size = layer_size
        # Output action.
        with tf.variable_scope('output'):
          output_size = reduce(operator.mul, self.action_shape, 1)
          initializer = tf.random_uniform_initializer(minval=-_LAST_LAYER_INIT, maxval=_LAST_LAYER_INIT)
          w = tf.get_variable('w', (previous_size, output_size), initializer=initializer)
          b = tf.get_variable('b', (output_size,), initializer=initializer)
          params.extend((w, b))
    return params

  def ActorNetwork(self, input_observation, params, device='/cpu:0'):
    index = 0
    with tf.device(device):
      with tf.variable_scope('actor'):
        # Input is flattened.
        flat_input_observation = tf.contrib.layers.flatten(input_observation)
        previous_input = flat_input_observation
        # Layers.
        for i, layer_size in enumerate(_LAYERS):
          with tf.variable_scope('layer_%d' % i):
            w = params[index]
            b = params[index + 1]
            index += 2
            previous_input = tf.nn.relu(tf.nn.xw_plus_b(previous_input, w, b))
        # Output action.
        with tf.variable_scope('output'):
          w = params[index]
          b = params[index + 1]
          index += 2
          # TODO: Reshape to requested shape.
          return tf.nn.tanh(tf.nn.xw_plus_b(previous_input, w, b))

  def CriticNetworkParameters(self, device='/cpu:0'):
    params = []
    with tf.device(device):
      with tf.variable_scope('critic'):
        # Input is flattened.
        previous_size = reduce(operator.mul, self.observation_shape, 1)
        # Layers.
        for i, layer_size in enumerate(_LAYERS):
          with tf.variable_scope('layer_%d' % i):
            if i == 1:
              # Input action in the second layer.
              previous_size += reduce(operator.mul, self.action_shape, 1)
            initializer = tf.random_uniform_initializer(minval=1.0 / math.sqrt(previous_size),
                                                        maxval=1.0 / math.sqrt(previous_size))
            w = tf.get_variable('w', (previous_size, layer_size), initializer=initializer)
            b = tf.get_variable('b', (layer_size,), initializer=initializer)
            params.extend((w, b))
            previous_size = layer_size
        # Output q-value.
        with tf.variable_scope('output'):
          output_size = 1
          initializer = tf.random_uniform_initializer(minval=-_LAST_LAYER_INIT, maxval=_LAST_LAYER_INIT)
          w = tf.get_variable('w', (previous_size, output_size), initializer=initializer)
          b = tf.get_variable('b', (output_size,), initializer=initializer)
          params.extend((w, b))
    return params

  def CriticNetwork(self, input_action, input_observation, params, device='/cpu:0'):
    index = 0
    with tf.device(device):
      with tf.variable_scope('critic'):
        # Input is flattened.
        flat_input_observation = tf.contrib.layers.flatten(input_observation)
        previous_input = flat_input_observation
        # Layers.
        for i, layer_size in enumerate(_LAYERS):
          with tf.variable_scope('layer_%d' % i):
            if i == 1:
              # Input action in the second layer.
              flat_input_action = tf.contrib.layers.flatten(input_action)
              previous_input = tf.concat(1, [previous_input, flat_input_action])
            w = params[index]
            b = params[index + 1]
            index += 2
            previous_input = tf.nn.relu(tf.nn.xw_plus_b(previous_input, w, b))
        # Output q-value.
        with tf.variable_scope('output'):
          w = params[index]
          b = params[index + 1]
          index += 2
          return tf.nn.xw_plus_b(previous_input, w, b)


class WrapComputationalGraph(object):
  def __init__(self, inputs, outputs, session=None):
    self._inputs = inputs
    self._outputs = outputs
    self._session = session or tf.get_default_session()

  def __call__(self, *args):
    feed_dict = {}
    for (argpos, arg) in enumerate(args):
      feed_dict[self._inputs[argpos]] = arg
    results = self._session.run(self._outputs, feed_dict)
    return results


def ExponentialMovingAverage(tensors, decay=0.99):
  average = tf.train.ExponentialMovingAverage(decay=decay)
  op = average.apply(tensors)
  output_tensors = [average.average(x) for x in tensors]
  return output_tensors, op


def OrnsteinUhlenbeckProcess(shape, theta, sigma):
  with tf.variable_scope('output'):
    initial_noise = tf.zeros(shape)
    noise = tf.get_variable('noise', shape, initializer=tf.constant_initializer(0.))
    reset_op = noise.assign(initial_noise)
    return noise.assign_sub(theta * noise - tf.random_normal(shape, stddev=sigma)), reset_op