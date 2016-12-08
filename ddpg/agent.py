import logging
import gym
import numpy as np

import model
import replay_memory


_REPLAY_MEMORY_SIZE = 1e6
_BATCH_SIZE = 64
_WARMUP_TIMESTEPS = _BATCH_SIZE * 10

# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Agent(object):

  def __init__(self, action_space, observation_space, checkpoint_directory, device='/cpu:0'):
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
    self.model = model.Model(self.action_space_shape, observation_space_shape, checkpoint_directory,
                             device=device)

  def Reset(self):
    self.model.Reset()

  def Observe(self, observation):
    self.observation = observation

  def Act(self, is_training=False):
    # Act randomly initially.
    action = self.action = self.model.Act(self.observation, add_noise=is_training)
    if not self.continuous_actions:
      action = np.argmax(self.action)
    return action

  def GiveReward(self, reward, done, next_observation, is_training=False):
    if not is_training:
      return
    self.replay_memory.Add(self.action, self.observation, reward, done, next_observation)
    if len(self.replay_memory) >= _WARMUP_TIMESTEPS:
      batch = self.replay_memory.Sample(_BATCH_SIZE)
      self.model.Train(*batch)

  def Save(self, checkpoint_index):
    filename = self.model.Save(step=checkpoint_index)
    LOG.info('Saving checkpoint %s', filename)
