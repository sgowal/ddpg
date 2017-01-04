import logging
import matplotlib.pylab as plt
import numpy as np
import os
import pickle

from .. import manager
from .. import privacy_options_pb2

import utils

# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


# Special manager that is aware of additional functions used in the private environments.
class EnvironmentPrivacyManager(manager.Manager):

  def __init__(self, environment, agent, output_directory, options, restore=False):
    super(EnvironmentPrivacyManager, self).__init__(environment, agent, output_directory, options, restore)
    self.environment_output_directory = os.path.join(output_directory, 'environment')
    if restore:
      self.environment.Restore(self.environment_output_directory)
    else:
      os.makedirs(self.environment_output_directory)
    self.environment.SetOptions(options.privacy)

  def Run(self):
    num_training_timesteps = self.restored_num_training_timesteps
    while True:
      # Test.
      self.environment.SetTrainingMode(is_training=False)
      rewards = []
      performance_rewards = []
      privacy_rewards = []
      # Store trajectories.
      trajectories = []
      for _ in range(len(self.environment.TargetLabels())):
        trajectories.append([])
      for i in range(self.environment.spec.trials):
        r, t, performance_reward, privacy_reward, trajectory = self.RunEpisode(is_training=False, record_video=i < self.options.num_recorded_runs)
        rewards.append(r)
        performance_rewards.append(performance_reward)
        privacy_rewards.append(privacy_reward)
        trajectories[self.environment.GetChosenTarget()].append(trajectory)
      average_reward = self.WriteResultSummary(num_training_timesteps, rewards, is_training=False)
      self.WriteResultSummary(num_training_timesteps, performance_rewards, is_training=False, postfix=' (preformance)')
      self.WriteResultSummary(num_training_timesteps, privacy_rewards, is_training=False, postfix=' (privacy)')
      self.WriteMovieSummary(num_training_timesteps)
      if self.options.privacy.save_trajectories:
        analyzer = utils.TrajectoryAnalyzer(trajectories=trajectories, labels=self.environment.TargetLabels())
        analyzer.PlotTrajectories()
        plt.savefig(os.path.join(self.monitoring_path, 'trajectories_%06d.png' % num_training_timesteps), format='png')
        analyzer.Save(os.path.join(self.monitoring_path, 'trajectories_%06d.pickle' % num_training_timesteps))
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
        if self.options.privacy.mode == privacy_options_pb2.PrivacyOptions.ALTERNATE:
          self.environment.SetTrainingMode(is_training=False)
          for i in range(self.options.privacy.ddpg_training_episodes):
            r, t, _, _, _ = self.RunEpisode(is_training=True)
            rewards.append(r)
            training_timesteps += t
          self.environment.SetTrainingMode(is_training=True)
          for i in range(self.options.privacy.privacy_training_episodes):
            self.RunEpisode(is_training=False)
          num_episodes += self.options.privacy.ddpg_training_episodes
        else:
          self.environment.SetTrainingMode(is_training=True)
          r, t, _, _ = self.RunEpisode(is_training=True)
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
    trajectory = [self.environment.GetState()]
    actions = []
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
      if not done:
        trajectory.append(self.environment.GetState())
      actions.append(action)
    trajectory = np.vstack(trajectory)
    actions = np.vstack(actions)
    trajectory = np.hstack([trajectory, actions])
    return total_reward, timesteps, total_performance_reward, total_privacy_reward, trajectory
