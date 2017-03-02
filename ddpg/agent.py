import logging
import gym
import numpy as np
import os

import model
import replay_memory


# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Agent(object):

  def __init__(self, action_space, observation_space, checkpoint_directory, options, restore=False):
    self.options = options
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
    if self.options.use_rank_based_replay:
      self.replay_memory = replay_memory.RankBased(self.options.replay_memory_size, self.action_space_shape, observation_space_shape)
    else:
      self.replay_memory = replay_memory.Uniform(self.options.replay_memory_size, self.action_space_shape, observation_space_shape)
    self.last_checkpoint_index = 0
    if restore:
      filename = self.replay_memory.Load(os.path.join(checkpoint_directory, 'memory.ckpt'))
      LOG.info('Restoring replay memory from %s', filename)
      self.last_checkpoint_index = int(filename.rsplit('-', 1)[1])
    LOG.info('Initialized agent with actions %s and observations %s',
             str(self.action_space_shape), str(observation_space_shape))
    # Tensorflow model.
    self.checkpoint_directory = checkpoint_directory
    self.model = model.Model(self.action_space_shape, observation_space_shape, checkpoint_directory,
                             options=options, restore=restore)

  def Reset(self):
    self.model.Reset()

  def Observe(self, observation):
    self.observation = observation

  def Act(self, add_noise=0.):
    # Act randomly initially.
    action = self.action = self.model.Act(self.observation, add_noise=add_noise)
    if not self.continuous_actions:
      action = np.argmax(self.action)
    return action

  def GiveReward(self, reward, done, next_observation, is_training=False):
    if not is_training:
      return
    self.replay_memory.Add(self.action, self.observation, reward, done, next_observation)
    if len(self.replay_memory) >= self.options.warmup_timesteps:
      batch = self.replay_memory.Sample(self.options.batch_size)
      td_error = self.model.Train(*batch)
      self.replay_memory.Update(np.abs(td_error))

  def Save(self, checkpoint_index):
    self.last_checkpoint_index = checkpoint_index
    filename = self.model.Save(step=checkpoint_index)
    self.replay_memory.Save(os.path.join(self.checkpoint_directory, 'memory.ckpt'),
                            step=checkpoint_index)
    LOG.info('Saving checkpoint %s', filename)

  def GetLatestSavedStep(self):
    return self.last_checkpoint_index
