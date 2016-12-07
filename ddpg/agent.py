import logging
import gym
import numpy as np

import model
import replay_memory


_REPLAY_MEMORY_SIZE = 1e6

# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Agent(object):

  def __init__(self, action_space, observation_space, checkpoint_directory):
    self.action_space = action_space
    if isinstance(action_space, gym.spaces.Discrete):
      # The following is a hack that casts the discrete problems to continuous ones.
      # Actions are then selected based on the argmax of the policy network.
      # This works surprising well.
      self.action_space_shape = (action_space.n,)
      self.continuous_actions = False
    else:
      self.action_space_shape = action_space.shape
      self.continuous_actions = True
    observation_space_shape = observation_space.shape
    self.replay_memory = replay_memory.ReplayMemory(_REPLAY_MEMORY_SIZE, self.action_space_shape, observation_space_shape)
    LOG.info('Initialized agent with actions %s and observations %s',
             str(self.action_space_shape), str(observation_space_shape))
    # Tensorflow model.
    self.model = model.Model(self.action_space_shape, observation_space_shape, checkpoint_directory)

  def Reset(self):
    self.model.Reset()

  def Observe(self, observation):
    self.observation = observation

  def Act(self):
    action = self.action_space.sample()
    if self.continuous_actions:
      self.action = action
    else:
      one_hot_encoding = np.zeros(self.action_space_shape)
      one_hot_encoding[action] = 1.
      self.action = one_hot_encoding
    return action

  def GiveReward(self, reward, done, next_observation):
    self.replay_memory.Add(self.action, self.observation, reward, done, next_observation)
    # Train.
    # TODO TODO.

  def Save(self, checkpoint_index):
    filename = self.model.Save(step=checkpoint_index)
    LOG.info('Saving checkpoint %s', filename)
