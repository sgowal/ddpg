# -*- coding: utf-8 -*-

import logging
import numpy as np
import os


# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Manager(object):

  def __init__(self, environment, agent, options):
    self.environment = environment
    self.agent = agent
    self.options = options
    environment.monitor.start(os.path.join(options.output_directory, 'monitor'),
                              video_callable=lambda _: False)

  def __del__(self):
    self.environment.monitor.close()

  def Run(self):
    num_training_episodes = 0
    while num_training_episodes < self.options.max_episodes:
      # Test.
      rewards = []
      for i in range(self.environment.spec.trials):
        rewards.append(self.RunEpisode(is_training=False, record_video=i == 0)[0])
      average_reward = np.mean(rewards)
      stddev_reward = np.std(rewards)
      LOG.info(u'%d-episode evaluation average reward (after %d episodes): %.2f ± %.2f',
               self.environment.spec.trials, num_training_episodes, average_reward, stddev_reward)

      # Train.
      training_timesteps = 0
      num_episodes = 0
      rewards = []
      while training_timesteps < self.options.evaluate_after_timesteps:
        r, t = self.RunEpisode(is_training=True)
        rewards.append(r)
        training_timesteps += t
        num_episodes += 1
      num_training_episodes += num_episodes
      average_reward = np.mean(rewards)
      stddev_reward = np.std(rewards)
      LOG.info(u'%d-episode training average reward (after %d episodes): %.2f ± %.2f',
               num_episodes, num_training_episodes, average_reward, stddev_reward)
      self.agent.Save(num_training_episodes)

  def RunEpisode(self, is_training=False, record_video=False, show=False):
    self.environment.monitor.configure(lambda _: record_video)
    total_reward = 0.
    timesteps = 0
    done = False

    observation = self.environment.reset()
    while not done:
      if show:
        self.environment.render()
      self.agent.Observe(observation)
      action = self.agent.Act()
      observation, reward, done, _ = self.environment.step(action)
      total_reward += reward
      timesteps += 1
      done = (timesteps > self.options.max_timesteps_per_episode) or done
      self.agent.GiveReward(reward, done, observation)
    return total_reward, timesteps


def Start(environment, agent, options):
  manager = Manager(environment, agent, options)
  manager.Run()
