from __future__ import print_function

import collections
import glob
import matplotlib.pylab as plt
import numpy as np
import os
import re
import tensorflow as tf
import tqdm

# HACK to be able to hide binary in a separate folder.
import sys
parent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, parent_directory)

import ddpg.privacy

flags = tf.app.flags
flags.DEFINE_string('trajectory_directory', None, 'Directory where privacy trajectories are stored.')
flags.DEFINE_string('trajectory_file', None, 'Specific trajectory file for deeper analysis.')
flags.DEFINE_string('show_only', None, 'Comma-separated list of regular expressions that are matched against all subdirectories to include in the report.')
flags.DEFINE_string('group_by', None, 'Comma-separated list of regular expressions that are used to average across multiple runs.')

flags.DEFINE_string('switching_points', '35-45',
                    'Compute privacy on trajectory points before this many timesteps. '
                    'Compute performance on trajectory points after this many timesteps.')
flags.DEFINE_bool('disable_plot', False, 'Disable plots.')
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
  cutoffs = [int(p) for p in FLAGS.switching_points.split('-')]
  values = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
  for name, filenames in groups.iteritems():
    print('Analyzing', name)
    analyzer = ddpg.privacy.TrajectoryAnalyzer(cutoffs=cutoffs)
    for filename in tqdm.tqdm(filenames):
      analyzer.Load(filename)
      timestep = int(os.path.splitext(filename)[0].rsplit('_', 1)[1])
      metrics = analyzer.ComputeMetrics()
      for k, v in metrics.iteritems():
        values[k][name].append((timestep, v))

  if FLAGS.group_by is not None:
    regexps = FLAGS.group_by.split(',')
    new_values_mean = {}
    new_values_std = {}
    # For all metrics.
    for metric_name, metric_values in values.iteritems():
      groups = collections.defaultdict(lambda: [])
      for i, (k, v) in enumerate(metric_values.iteritems()):
        valid_regexps = []
        for r in regexps:
          g = re.match(r, k)
          if not g:
            continue
          valid_regexps.append('/'.join(g.groups()))

        timesteps, mean = zip(*sorted(v))
        for r in valid_regexps:
          groups[r].append((np.array(timesteps), np.array(mean)))
      average = collections.defaultdict(lambda: [])
      stddev = collections.defaultdict(lambda: [])
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
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        for t, m, s in zip(timesteps, mean, std):
          average[k].append((t, m))
          stddev[k].append((t, s))
      new_values_mean[metric_name] = average
      new_values_std[metric_name] = stddev
  else:
    new_values_mean = values
    new_values_std = None

  # Plot all metrics.
  for j, (metric_name, metric_values) in enumerate(new_values_mean.iteritems()):
    plt.figure()
    colors = _GetColors(len(metric_values))
    for i, (k, v) in enumerate(metric_values.iteritems()):
      timesteps, mean = zip(*sorted(v))
      if new_values_std is not None:
        _, std = zip(*sorted(new_values_std[metric_name][k]))
      else:
        std = 0.
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
      if new_values_std is not None:
        plt.fill_between(timesteps, mean - std, mean + std, color=colors[i % len(colors)], alpha=.5)
    if len(metric_values) > 1:
      plt.legend(loc='lower right')
    plt.xlim((0, np.max(timesteps)))
    plt.grid('on')
    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.tight_layout()
    if FLAGS.save_filename_prefix:
      plt.savefig(FLAGS.save_filename_prefix + 'metrics_%d.png' % j, format='png')
      plt.savefig(FLAGS.save_filename_prefix + 'metrics_%d.eps' % j, format='eps')
      print('Saved metrics: %s' % (FLAGS.save_filename_prefix + 'metrics_*.[png|eps]'))


def Run():
  if FLAGS.trajectory_file is not None:
    RunSingle(FLAGS.trajectory_file)
    return
  file_glob = os.path.join(FLAGS.trajectory_directory, '**/trajectories_*.pickle')
  trajectory_files = sorted(list(glob.iglob(file_glob)), key=os.path.getctime)
  assert trajectory_files, '%s does not match any files.' % FLAGS.trajectory_directory
  RunMultiple(trajectory_files)
  if not FLAGS.disable_plot:
    plt.show()


def _GetColors(n):
  cm = plt.get_cmap('gist_rainbow')
  return [cm(float(i) / float(n)) for i in range(n)]


if __name__ == '__main__':
  Run()
