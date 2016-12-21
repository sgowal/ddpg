from __future__ import print_function

import collections
import glob
import numpy as np
import matplotlib.pylab as plt
import os
import tensorflow as tf
import re

import ddpg

flags = tf.app.flags
flags.DEFINE_string('event_directory', None, 'Directory where TensorFlow results are stored.')
flags.DEFINE_string('show_only', None, 'Comma-separated list of regular expressions that are matched against all subdirectories to include in the report.')
flags.DEFINE_string('group_by', None, 'Comma-separated list of regular expressions that are used to average across multiple runs.')
FLAGS = flags.FLAGS


def Run():
  event_files = list(glob.iglob(os.path.join(FLAGS.event_directory, '**/events.out.tfevents.*')))
  common_directory = os.path.commonprefix(event_files)
  if common_directory and common_directory[-1] != '/':
    common_directory = os.path.dirname(common_directory) + '/'
  whitelist = set(FLAGS.show_only.split(',')) if FLAGS.show_only else None
  # Retrieve all relevant events.
  average_reward = collections.defaultdict(lambda: [])
  stddev_reward = collections.defaultdict(lambda: [])
  for event_file in event_files:
    canonical_name = os.path.dirname(event_file[len(common_directory):])
    if whitelist and not any(re.match(w, canonical_name) for w in whitelist):
      print('Skipping', canonical_name)
      continue
    print('Analyzing', canonical_name)
    for event in tf.train.summary_iterator(event_file):
      for value in event.summary.value:
        if value.tag == ddpg.AVERAGE_REWARD_TAG:
          average_reward[canonical_name].append((event.step, value.simple_value))
        elif value.tag == ddpg.STDDEV_REWARD_TAG:
          stddev_reward[canonical_name].append((event.step, value.simple_value))
  if not average_reward:
    print('No data found.')
    return

  if FLAGS.group_by is not None:
    regexps = FLAGS.group_by.split(',')
    # common_directory = os.path.commonprefix(regexps)
    # if common_directory and common_directory[-1] != '/':
    #   common_directory = os.path.dirname(common_directory) + '/'
    groups = collections.defaultdict(lambda: [])
    for i, (k, v) in enumerate(average_reward.iteritems()):
      valid_regexps = []
      for r in regexps:
        g = re.match(r, k)
        if not g:
          continue
        valid_regexps.append('/'.join(g.groups()))
      timesteps, mean = zip(*sorted(v))
      for r in valid_regexps:
        groups[r].append((np.array(timesteps), np.array(mean)))
    average_reward = collections.defaultdict(lambda: [])
    stddev_reward = collections.defaultdict(lambda: [])
    for k, v in groups.iteritems():
      timesteps = []
      for t, _ in v:
        if len(timesteps) < len(t):
          timesteps = t
      values = []
      for _, m in v:
        expanded_mean = np.empty_like(timesteps, dtype=np.float32)
        expanded_mean[:len(m)] = m
        expanded_mean[len(m):] = m[-1]
        values.append(expanded_mean)
      values = np.vstack(values)
      # mean = np.mean(values, axis=0)
      mean = np.median(values, axis=0)
      std = np.std(values, axis=0)
      for t, m, s in zip(timesteps, mean, std):
        average_reward[k].append((t, m))
        stddev_reward[k].append((t, s))

  # Plot.
  plt.figure()
  colors = _GetColors(len(average_reward))
  for i, (k, v) in enumerate(average_reward.iteritems()):
    timesteps, mean = zip(*sorted(v))
    _, std = zip(*sorted(stddev_reward[k]))
    timesteps = np.array(timesteps)
    mean = np.array(mean)
    std = np.array(std)
    # Timesteps can be duplicated when restoring from prior checkpoints.
    timesteps, unique_indices = np.unique(timesteps, return_index=True)
    mean = mean[unique_indices]
    std = std[unique_indices]
    timesteps = np.array(timesteps)
    mean = np.array(mean)
    std = np.array(std)
    plt.plot(timesteps, mean, color=colors[i % len(colors)], lw=2, label=k)
    plt.fill_between(timesteps, mean - std, mean + std, color=colors[i % len(colors)], alpha=.5)
  if len(average_reward) > 1:
    plt.legend(loc='lower right')
  plt.xlim((0, np.max(timesteps)))
  plt.grid('on')
  plt.xlabel('Step')
  plt.ylabel('Reward')
  plt.show()


def _GetColors(n):
  cm = plt.get_cmap('gist_rainbow')
  return [cm(float(i) / float(n)) for i in range(n)]


if __name__ == '__main__':
  Run()
