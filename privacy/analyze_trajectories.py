from __future__ import print_function

import collections
import glob
import matplotlib.pylab as plt
import numpy as np
import os
import tensorflow as tf

# HACK to be able to hide binary in a separate folder.
import sys
parent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_directory)

import ddpg.privacy

flags = tf.app.flags
flags.DEFINE_string('trajectories', None, 'File containing the privacy trajectories (can be a glob).')
flags.DEFINE_string('switching_points', '35-45',
                    'Compute privacy on trajectory points before this many timesteps. '
                    'Compute performance on trajectory points after this many timesteps.')
flags.DEFINE_bool('disable_plot', False, 'Disable plots.')
flags.DEFINE_bool('boxplot', False, 'When there are multiple trajectory files, uses boxplots instead of time series.')
flags.DEFINE_string('save_filename_prefix', None, 'Saves plots with a given prefix.')
FLAGS = flags.FLAGS


def RunSingle(filename):
  analyzer = ddpg.privacy.TrajectoryAnalyzer(cutoffs=[int(p) for p in FLAGS.switching_points.split('-')])
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

  # Compute metrics.
  analyzer.ComputeMetrics(plot_ks_distance=True)
  if FLAGS.save_filename_prefix:
    plt.savefig(FLAGS.save_filename_prefix + 'ks_distance.png', format='png')
    plt.savefig(FLAGS.save_filename_prefix + 'ks_distance.eps', format='eps')
    print('Saved distributions: %s' % (FLAGS.save_filename_prefix + 'ks_distance.[png|eps]'))
  metrics = analyzer.ComputeMetrics(plot_leakage=True)
  if FLAGS.save_filename_prefix:
    plt.savefig(FLAGS.save_filename_prefix + 'leakage.png', format='png')
    plt.savefig(FLAGS.save_filename_prefix + 'leakage.eps', format='eps')
    print('Saved distributions: %s' % (FLAGS.save_filename_prefix + 'leakage.[png|eps]'))
  for k, v in metrics.iteritems():
    print('%s: %.3f' % (k, v))
  if FLAGS.save_filename_prefix:
    with open(FLAGS.save_filename_prefix + 'info.txt', 'w') as fp:
      for k, v in metrics.iteritems():
        fp.write('%s: %.3f\n' % (k, v))
      print('Saved information: %s' % (FLAGS.save_filename_prefix + 'info.txt'))


def RunMultiple(filenames):
  # Common start.
  common_start = os.path.commonprefix(filenames)
  if common_start and common_start[-1] != '/':
    common_start = os.path.dirname(common_start) + '/'
  # Common ending.
  reversed_filenames = [os.path.dirname(f[len(common_start):])[::-1] for f in filenames]
  common_end = os.path.commonprefix(reversed_filenames)
  if common_end and common_end[-1] != '/':
    common_end = os.path.dirname(common_end) + '/'
  common_end = common_end[::-1]
  # Group files.
  groups = collections.defaultdict(lambda: [])
  for filename in filenames:
    canonical_name = os.path.dirname(filename[len(common_start):])[:-len(common_end)]
    groups[canonical_name].append(filename)
  # Analyze each group.
  values = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
  for name, filenames in groups.iteritems():
    print('Analyzing', name)
    analyzer = ddpg.privacy.TrajectoryAnalyzer(cutoffs=[int(p) for p in FLAGS.switching_points.split('-')])
    for filename in filenames:
      analyzer.Load(filename)
      timestep = int(os.path.splitext(filename)[0].rsplit('_', 1)[1])
      metrics = analyzer.ComputeMetrics()
      for k, v in metrics.iteritems():
        values[k][name].append((timestep, v))
  # Plot all metrics.
  max_timestep = 0
  for i, (metric_name, v_dict) in enumerate(values.iteritems()):
    plt.figure()
    if FLAGS.boxplot:
      data = []
      labels = []
      for k, v in sorted(v_dict.iteritems()):
        _, vs = zip(*sorted(v))
        data.append(vs[2:])  # Ignore first 2 steps.
        labels.append(k)
      plt.boxplot(data)
      plt.xticks(range(1, len(data) + 1), labels, rotation='vertical')
    else:
      colors = _GetColors(len(v_dict))
      for (k, v), color in zip(v_dict.iteritems(), colors):
        timesteps, vs = zip(*sorted(v))
        plt.plot(timesteps, vs, color=color, lw=2, label=k)
        max_timestep = max(max_timestep, np.max(timesteps))
      plt.legend(loc='lower right')
      plt.xlim((0, max_timestep))
      plt.grid('on')
      plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.tight_layout()
    if FLAGS.save_filename_prefix:
      plt.savefig(FLAGS.save_filename_prefix + 'metrics_%d.png' % i, format='png')
      plt.savefig(FLAGS.save_filename_prefix + 'metrics_%d.eps' % i, format='eps')
      print('Saved metrics: %s' % (FLAGS.save_filename_prefix + 'metrics_*.[png|eps]'))


def Run():
  trajectory_files = sorted(list(glob.iglob(FLAGS.trajectories)), key=os.path.getctime)
  assert trajectory_files, '%s does not match any files.' % FLAGS.trajectories
  if len(trajectory_files) == 1:
    RunSingle(trajectory_files[0])
  else:
    RunMultiple(trajectory_files)

  if not FLAGS.disable_plot:
    plt.show()


def _GetColors(n):
  cm = plt.get_cmap('gist_rainbow')
  return [cm(float(i) / float(n)) for i in range(n)]


if __name__ == '__main__':
  Run()
