# -*- coding: utf-8 -*-

import glob
import logging
import numpy as np
import os
import tensorflow as tf

import options_pb2

# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

try:
  import moviepy.editor
  has_moviepy = True
except ImportError:
  LOG.warn('MoviePy not found. Install with "pip install moviepy"')
  has_moviepy = False


# Summary tags.
AVERAGE_REWARD_TAG = 'Average reward'
STDDEV_REWARD_TAG = 'Standard deviation of reward'


class Manager(object):

  def __init__(self, environment, agent, output_directory, options, restore=False):
    self.environment = environment
    self.agent = agent
    self.output_directory = output_directory
    self.options = options
    self.monitoring_path = os.path.join(output_directory, 'monitor')
    environment.monitor.start(self.monitoring_path,
                              resume=restore,
                              video_callable=lambda _: False)
    # Log performance.
    self.test_writer = tf.train.SummaryWriter(os.path.join(output_directory, 'test'))
    self.train_writer = tf.train.SummaryWriter(os.path.join(output_directory, 'train'))
    # Upon restore, the agent knows the number of training steps done so far.
    self.restored_num_training_timesteps = agent.GetLatestSavedStep()

  def __del__(self):
    self.environment.monitor.close()

  def WriteResultSummary(self, timestep, values, is_training=False, postfix=''):
    values = np.array(values)
    min_value = np.min(values)
    max_value = np.max(values)
    sum_value = np.sum(values)
    sum_squares_value = np.sum(np.square(values))
    mean_value = np.mean(values)
    std_value = np.std(values)
    hist, bin_edges = np.histogram(values, bins=10)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Rewards' + postfix, histo=tf.HistogramProto(
            min=min_value, max=max_value, sum=sum_value, sum_squares=sum_squares_value,
            bucket_limit=bin_edges[1:], bucket=hist)),
        tf.Summary.Value(tag=AVERAGE_REWARD_TAG + postfix, simple_value=mean_value),
        tf.Summary.Value(tag=STDDEV_REWARD_TAG + postfix, simple_value=std_value),
    ])
    if is_training:
      self.train_writer.add_summary(summary, timestep)
    else:
      self.test_writer.add_summary(summary, timestep)
    # Verbose output.
    LOG.info(u'%d-episode %s average reward (after %d timesteps): %.2f ± %.2f%s',
             len(values), 'training' if is_training else 'evaluation',
             timestep, mean_value, std_value, postfix)
    return mean_value

  def WriteMovieSummary(self, timestep):
    if not has_moviepy or self.options.disable_rendering or not self.options.record_gif:
      return
    # Get latest movie.
    movie_filename = max(glob.iglob(os.path.join(self.monitoring_path, 'openaigym.video.*.mp4')),
                         key=os.path.getctime)
    gif_filename = os.path.splitext(movie_filename)[0] + '.gif'
    try:
      clip = moviepy.editor.VideoFileClip(movie_filename)
      clip = clip.subclip(0, min(10, clip.end))  # Only make 10s GIF.
      clip = clip.resize(width=min(clip.w, 640))  # Resize to max-width 640.
      clip.write_gif(gif_filename)
      with open(gif_filename) as fp:
        gif_content = fp.read()
      summary = tf.Summary(value=[
          tf.Summary.Value(tag='Movie', image=tf.Summary.Image(
              height=clip.h, width=clip.w, encoded_image_string=gif_content)),
      ])
      self.test_writer.add_summary(summary, timestep)
    except ValueError:
      LOG.warn('Could not convert movie to gif for Tensorboard.')

  def Run(self):
    num_training_timesteps = self.restored_num_training_timesteps
    while True:
      # Test.
      rewards = []
      for i in range(self.environment.spec.trials):
        rewards.append(self.RunEpisode(is_training=False, record_video=i < self.options.num_recorded_runs)[0])
      average_reward = self.WriteResultSummary(num_training_timesteps, rewards, is_training=False)
      self.WriteMovieSummary(num_training_timesteps)
      if self.environment.spec.reward_threshold and average_reward > self.environment.spec.reward_threshold:
        LOG.info('Surpassing reward threshold of %.2f. Stopping...', self.environment.spec.reward_threshold)
        break
      if num_training_timesteps >= self.options.max_timesteps:
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
        # Break if we go over timestep limit.
        if num_training_timesteps + training_timesteps >= self.options.max_timesteps:
          break
      num_training_timesteps += training_timesteps
      self.WriteResultSummary(num_training_timesteps, rewards, is_training=True)
      self.agent.Save(num_training_timesteps)

  def RunEpisode(self, is_training=False, record_video=False, show=False):
    self.environment.monitor.configure(lambda _: record_video and not self.options.disable_rendering,
                                       mode='training' if is_training else 'evaluation')
    total_reward = 0.
    timesteps = 0
    done = False

    self.agent.Reset()
    observation = self.environment.reset()
    while not done:
      if show and not self.options.disable_rendering:
        self.environment.render()
      self.agent.Observe(observation)
      action = self.agent.Act(is_training=is_training)
      observation, reward, done, _ = self.environment.step(action)
      total_reward += reward
      timesteps += 1
      done = (timesteps >= self.options.max_timesteps_per_episode) or done
      self.agent.GiveReward(reward, done, observation, is_training=is_training)
    return total_reward, timesteps


def Start(environment, agent, output_directory, options, restore=False):
  try:
    environment.HasSpecialFunctions()
    manager = EnvironmentPrivacyManager(environment, agent, output_directory, options, restore=restore)
  except AttributeError:
    manager = Manager(environment, agent, output_directory, options, restore=restore)
  manager.Run()


# IGNORE what is below unless you are interested in the privacy code.
# Special manager that is aware of additional functions used in the private environments.
class EnvironmentPrivacyManager(Manager):

  def __init__(self, environment, agent, output_directory, options, restore=False):
    super(EnvironmentPrivacyManager, self).__init__(environment, agent, output_directory, options, restore)
    self.environment_output_directory = os.path.join(output_directory, 'environment')
    if restore:
      self.environment.Restore(self.environment_output_directory)
    else:
      os.makedirs(self.environment_output_directory)

  def Run(self):
    num_training_timesteps = self.restored_num_training_timesteps
    while True:
      # Test.
      self.environment.SetTrainingMode(is_training=False)
      rewards = []
      performance_rewards = []
      privacy_rewards = []
      for i in range(self.environment.spec.trials):
        self.environment.SetEpisodeNumber(i)
        r, t, performance_reward, privacy_reward = self.RunEpisode(is_training=False, record_video=i < self.options.num_recorded_runs)
        rewards.append(r)
        performance_rewards.append(performance_reward)
        privacy_rewards.append(privacy_reward)
      average_reward = self.WriteResultSummary(num_training_timesteps, rewards, is_training=False)
      self.WriteResultSummary(num_training_timesteps, performance_rewards, is_training=False, postfix=' (preformance)')
      self.WriteResultSummary(num_training_timesteps, privacy_rewards, is_training=False, postfix=' (privacy)')
      self.WriteMovieSummary(num_training_timesteps)
      if self.environment.spec.reward_threshold and average_reward > self.environment.spec.reward_threshold:
        LOG.info('Surpassing reward threshold of %.2f. Stopping...', self.environment.spec.reward_threshold)
        break
      if num_training_timesteps >= self.options.max_timesteps:
        break

      # Train.
      self.environment.SetTrainingMode(is_training=self.options.privacy.mode == options_pb2.Options.PrivacyOptions.SIMULTANEOUS)
      self.environment.SetEpisodeNumber(0)
      training_timesteps = 0
      num_episodes = 0
      rewards = []
      while training_timesteps < self.options.evaluate_after_timesteps:
        self.environment.SetEpisodeNumber(num_episodes)
        r, t, _, _ = self.RunEpisode(is_training=True)
        # When alternating between envrionment training and testing, run another episode without
        # training the agent.
        if self.options.privacy.mode == options_pb2.Options.PrivacyOptions.ALTERNATE:
          self.environment.SetTrainingMode(is_training=True)
          self.RunEpisode(is_training=False)
          self.environment.SetTrainingMode(is_training=False)

        rewards.append(r)
        training_timesteps += t
        num_episodes += 1
        # Break if we go over timestep limit.
        if num_training_timesteps + training_timesteps >= self.options.max_timesteps:
          break
      num_training_timesteps += training_timesteps
      self.WriteResultSummary(num_training_timesteps, rewards, is_training=True)
      self.agent.Save(num_training_timesteps)
      self.environment.Save(self.environment_output_directory, num_training_timesteps)

  def RunEpisode(self, is_training=False, record_video=False, show=False):
    self.environment.monitor.configure(lambda _: record_video and not self.options.disable_rendering,
                                       mode='training' if is_training else 'evaluation')
    total_reward = 0.
    timesteps = 0
    done = False

    total_privacy_reward = 0.
    total_performance_reward = 0.

    self.agent.Reset()
    observation = self.environment.reset()
    while not done:
      if show and not self.options.disable_rendering:
        self.environment.render()
      self.agent.Observe(observation)
      action = self.agent.Act(is_training=is_training)
      observation, reward, done, info = self.environment.step(action)
      total_reward += reward
      total_performance_reward += info['performance_reward']
      total_privacy_reward += info['privacy_reward']
      timesteps += 1
      done = (timesteps >= self.options.max_timesteps_per_episode) or done
      self.agent.GiveReward(reward, done, observation, is_training=is_training)
    return total_reward, timesteps, total_performance_reward, total_privacy_reward
