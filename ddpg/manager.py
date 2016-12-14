# -*- coding: utf-8 -*-

import glob
import logging
import numpy as np
import os
import tensorflow as tf

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

  def __init__(self, environment, agent, output_directory, options):
    self.environment = environment
    self.agent = agent
    self.output_directory = output_directory
    self.options = options
    self.monitoring_path = os.path.join(output_directory, 'monitor')
    environment.monitor.start(self.monitoring_path,
                              video_callable=lambda _: False, force=True)
    # Log performance.
    self.test_writer = tf.train.SummaryWriter(os.path.join(output_directory, 'test'))
    self.train_writer = tf.train.SummaryWriter(os.path.join(output_directory, 'train'))

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
        tf.Summary.Value(tag='Rewards', histo=tf.HistogramProto(
            min=min_value, max=max_value, sum=sum_value, sum_squares=sum_squares_value,
            bucket_limit=bin_edges[1:], bucket=hist)),
        tf.Summary.Value(tag=AVERAGE_REWARD_TAG, simple_value=mean_value),
        tf.Summary.Value(tag=STDDEV_REWARD_TAG, simple_value=std_value),
    ])
    if is_training:
      self.train_writer.add_summary(summary, timestep)
    else:
      self.test_writer.add_summary(summary, timestep)
    # Verbose output.
    LOG.info(u'%d-episode %s average reward (after %d timesteps): %.2f ± %.2f',
             len(values), 'training' if is_training else 'evaluation',
             timestep, mean_value, std_value)
    return mean_value

  def WriteMovieSummary(self, timestep):
    if not has_moviepy or self.options.disable_rendering:
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
    num_training_timesteps = 0
    while num_training_timesteps < self.options.max_timesteps:
      # Test.
      rewards = []
      for i in range(self.environment.spec.trials):
        rewards.append(self.RunEpisode(is_training=False, record_video=i == 0)[0])
      average_reward = self.WriteResultSummary(num_training_timesteps, rewards, is_training=False)
      self.WriteMovieSummary(num_training_timesteps)
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
        # Break if we go over timestep limit.
        if num_training_timesteps + training_timesteps >= self.options.max_timesteps:
          break
      num_training_timesteps += training_timesteps
      self.WriteResultSummary(num_training_timesteps, rewards, is_training=True)
      self.agent.Save(num_training_timesteps)
    LOG.info('To visualize results: tensorboard --logdir="%s"' % self.output_directory)

  def RunEpisode(self, is_training=False, record_video=False, show=False):
    self.environment.monitor.configure(lambda _: record_video and not self.options.disable_rendering)
    total_reward = 0.
    timesteps = 0
    done = False

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


def Start(environment, agent, output_directory, options):
  manager = Manager(environment, agent, output_directory, options)
  manager.Run()
