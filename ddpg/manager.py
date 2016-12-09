# -*- coding: utf-8 -*-

import logging
import matplotlib.pyplot as plt
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
                              video_callable=lambda _: False, force=True)
    self.mean_rewards_over_timesteps = []
    self.std_rewards_over_timesteps = []
    self.training_timesteps = []

  def __del__(self):
    self.environment.monitor.close()

  def Run(self):
    num_training_timesteps = 0
    while num_training_timesteps < self.options.max_timesteps:
      # Test.
      rewards = []
      for i in range(self.environment.spec.trials):
        rewards.append(self.RunEpisode(is_training=False, record_video=i == 0)[0])
      average_reward = np.mean(rewards)
      stddev_reward = np.std(rewards)
      self.mean_rewards_over_timesteps.append(average_reward)
      self.std_rewards_over_timesteps.append(stddev_reward)
      self.training_timesteps.append(num_training_timesteps)
      LOG.info(u'%d-episode evaluation average reward (after %d timesteps): %.2f ± %.2f',
               self.environment.spec.trials, num_training_timesteps, average_reward, stddev_reward)
      if self.environment.spec.reward_threshold and average_reward > self.environment.spec.reward_threshold:
        LOG.info('Surpassing reward threshold of %.2f. Stopping...', self.environment.spec.reward_threshold)
        break

      # Train.
      training_timesteps = 0
      num_episodes = 0
      rewards = []
      while training_timesteps < self.options.evaluate_after_timesteps:
        r, t = self.RunEpisode(is_training=True)
        rewards.append(r)
        training_timesteps += t
        num_episodes += 1
      num_training_timesteps += training_timesteps
      average_reward = np.mean(rewards)
      stddev_reward = np.std(rewards)
      LOG.info(u'%d-episode training average reward (after %d timesteps): %.2f ± %.2f',
               num_episodes, num_training_timesteps, average_reward, stddev_reward)
      self.agent.Save(num_training_timesteps)
    self.mean_rewards_over_timesteps = np.array(self.mean_rewards_over_timesteps)
    self.std_rewards_over_timesteps = np.array(self.std_rewards_over_timesteps)
    self.training_timesteps = np.array(self.training_timesteps)
    max_reward = np.max(self.mean_rewards_over_timesteps)
    self.mean_rewards_over_timesteps /= max_reward
    self.std_rewards_over_timesteps /= max_reward

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
      action = self.agent.Act(is_training=is_training)
      observation, reward, done, _ = self.environment.step(action)
      total_reward += reward
      timesteps += 1
      done = (timesteps >= self.options.max_timesteps_per_episode) or done
      self.agent.GiveReward(reward, done, observation, is_training=is_training)
    return total_reward, timesteps

  def PlotRewards(self):
    plt.figure()
    plt.plot(self.training_timesteps, self.mean_rewards_over_timesteps, 'k')
    plt.fill_between(
        self.training_timesteps,
        self.mean_rewards_over_timesteps - self.std_rewards_over_timesteps,
        self.mean_rewards_over_timesteps + self.std_rewards_over_timesteps,
        facecolor='black', alpha=0.1)
    plt.ylim(bottom=0)
    plt.xlabel('Steps')
    plt.ylabel('Normalized Reward')
    plt.show()


def Start(environment, agent, options):
  manager = Manager(environment, agent, options)
  manager.Run()
  return manager
