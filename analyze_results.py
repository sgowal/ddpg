from __future__ import print_function

import collections
import glob
import numpy as np
import matplotlib.pylab as plt
import os
import tensorflow as tf

import ddpg

flags = tf.app.flags
flags.DEFINE_string('event_directory', None, 'Directory where TensorFlow results are stored.')
flags.DEFINE_string('show_only', None, 'Comma-separated list of subdirectories to include in the report.')
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
    if whitelist and canonical_name not in whitelist:
      print('Skipping', canonical_name)
      continue
    print('Analyzing', canonical_name)
    for event in tf.train.summary_iterator(event_file):
      for value in event.summary.value:
        if value.tag == ddpg.AVERAGE_REWARD_TAG:
          average_reward[canonical_name].append((event.step, value.simple_value))
          pass
        elif value.tag == ddpg.STDDEV_REWARD_TAG:
          stddev_reward[canonical_name].append((event.step, value.simple_value))
  # Plot.
  plt.figure()
  colors = ('coral', 'deepskyblue')
  for i, (k, v) in enumerate(average_reward.iteritems()):
    timesteps, mean = zip(*v)
    _, std = zip(*stddev_reward[k])
    timesteps = np.array(timesteps)
    mean = np.array(mean)
    std = np.array(std)
    plt.plot(timesteps, mean, color=colors[i % len(colors)], lw=2, label=k)
    plt.fill_between(timesteps, mean - std, mean + std, color=colors[i % len(colors)], alpha=.5)
  plt.legend(loc='lower right')
  plt.xlim((0, np.max(timesteps)))
  plt.ylim(bottom=0)
  plt.grid('on')
  plt.xlabel('Step')
  plt.ylabel('Reward')
  plt.show()


if __name__ == '__main__':
  Run()
