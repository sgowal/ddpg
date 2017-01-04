from __future__ import print_function

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

# HACK to be able to hide binary in a separate folder.
import os
import sys
parent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_directory)

import ddpg.privacy

flags = tf.app.flags
flags.DEFINE_string('trajectories', None, 'File containing the privacy trajectories.')
flags.DEFINE_integer('ignore_after', 40, 'Ignore trajectory points beyond this many timesteps.')
flags.DEFINE_bool('disable_plot', False, 'Disable plots.')
flags.DEFINE_string('save_filename_prefix', None, 'Saves plots with a given prefix.')
FLAGS = flags.FLAGS


def Run():
  analyzer = ddpg.privacy.TrajectoryAnalyzer(cutoff=FLAGS.ignore_after)
  analyzer.Load(FLAGS.trajectories)

  # Plot trajectories.
  analyzer.PlotTrajectories()
  if FLAGS.save_filename_prefix:
    plt.savefig(FLAGS.save_filename_prefix + 'trajectories.png', format='png')
    plt.savefig(FLAGS.save_filename_prefix + 'trajectories.eps', format='eps')
    print('Saved trajectories: %s' % (FLAGS.save_filename_prefix + 'trajectories.[png|eps]'))

  # Plot distributions.
  analyzer.PlotDistributions()
  if FLAGS.save_filename_prefix:
    plt.savefig(FLAGS.save_filename_prefix + 'distributions.png', format='png')
    plt.savefig(FLAGS.save_filename_prefix + 'distributions.eps', format='eps')
    print('Saved distributions: %s' % (FLAGS.save_filename_prefix + 'distributions.[png|eps]'))

  # Compute 2D KS-distance.
  metrics = analyzer.ComputeMetrics()
  for k, v in metrics.iteritems():
    print('%s: %.3f' % (k, v))
  if FLAGS.save_filename_prefix:
    with open(FLAGS.save_filename_prefix + 'info.txt', 'w') as fp:
      for k, v in metrics.iteritems():
        fp.write('%s: %.3f\n' % (k, v))
      print('Saved information: %s' % (FLAGS.save_filename_prefix + 'info.txt'))

  if not FLAGS.disable_plot:
    plt.show()


if __name__ == '__main__':
  Run()
