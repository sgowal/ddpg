# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import tensorflow as tf

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
    # Log performance.
    self.test_writer = tf.train.SummaryWriter(os.path.join(options.output_directory, 'test'))
    self.train_writer = tf.train.SummaryWriter(os.path.join(options.output_directory, 'train'))

  def __del__(self):
    self.environment.monitor.close()

  def WriteResultSummary(self, timestep, values, is_training=False):
    values = np.array(values)
    min_value = np.min(values)
    max_value = np.max(values)
    sum_value = np.sum(values)
    sum_squares_value = np.sum(np.square(values))
    mean_value = np.mean(values)
    std_value = np.std(values)
    hist, bin_edges = np.histogram(values, bins=10)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag="Rewards", histo=tf.HistogramProto(
            min=min_value, max=max_value, sum=sum_value, sum_squares=sum_squares_value,
            bucket_limit=bin_edges[1:], bucket=hist)),
        tf.Summary.Value(tag='Average reward', simple_value=mean_value),
        tf.Summary.Value(tag='Standard deviation of reward', simple_value=std_value),
    ])
    if is_training:
      self.train_writer.add_summary(summary, timestep)
    else:
      self.test_writer.add_summary(summary, timestep)
    # Verbose output.
    LOG.info(u'%d-episode %s average reward (after %d timesteps): %.2f ± %.2f',
             self.environment.spec.trials, 'training' if is_training else 'evaluation',
             timestep, mean_value, std_value)
    return mean_value

  def Run(self):
    num_training_timesteps = 0
    while num_training_timesteps < self.options.max_timesteps:
      # Test.
      rewards = []
      for i in range(self.environment.spec.trials):
        rewards.append(self.RunEpisode(is_training=False, record_video=i == 0)[0])
      average_reward = self.WriteResultSummary(num_training_timesteps, rewards, is_training=False)
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
      self.WriteResultSummary(num_training_timesteps, rewards, is_training=True)
      self.agent.Save(num_training_timesteps)
    LOG.info('To visualize results: tensorboard --logdir="%s"' % self.output_directory)

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


def Start(environment, agent, options):
  manager = Manager(environment, agent, options)
  manager.Run()
