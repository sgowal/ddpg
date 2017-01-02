from __future__ import print_function

import matplotlib.pylab as plt
import numpy as np
import pickle
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('trajectories', None, 'File containing the privacy trajectories.')
flags.DEFINE_integer('ignore_after', 40, 'Ignore trajectory points beyond this many timesteps.')
flags.DEFINE_bool('disable_plot', False, 'Disable plots.')
flags.DEFINE_string('save_filename_prefix', None, 'Saves plots with a given prefix.')
FLAGS = flags.FLAGS


# Leakage smoother. Corresponds to the likelihood of observing something fully random.
_NU = 1e-4


def IntegralImage(x):
  y = np.zeros_like(x)
  y[0, 0] = x[0, 0]
  for i in xrange(1, x.shape[0]):
    y[i, 0] = y[i - 1, 0] + x[i, 0]
  for j in xrange(1, x.shape[1]):
    y[0, j] = y[0, j - 1] + x[0, j]
  for i in xrange(1, x.shape[0]):
    for j in xrange(1, x.shape[1]):
      y[i, j] = y[i - 1, j] + y[i, j - 1] - y[i - 1, j - 1] + x[i, j]
  return y


def Run():
  with open(FLAGS.trajectories, 'rb') as fp:
    pickler = pickle.Unpickler(fp)
    left_trajectories = pickler.load()
    right_trajectories = pickler.load()
  # Plot trajectories.
  plt.figure()
  for t in left_trajectories:
    plt.plot(t[:, 0], np.arange(t.shape[0]) * 0.02, color='lightcoral')
  for t in right_trajectories:
    plt.plot(t[:, 0], np.arange(t.shape[0]) * 0.02, color='lightgray')
  plt.plot([0.8, 0.8], [0, 1.6], 'k--')
  plt.plot([-0.8, -0.8], [0, 1.6], 'k--')
  plt.ylim([0, 1.6])
  plt.xlim([-1., 1.])
  plt.xlabel('position')
  plt.ylabel('time')
  if FLAGS.save_filename_prefix:
    plt.savefig(FLAGS.save_filename_prefix + 'trajectories.png', format='png')
    plt.savefig(FLAGS.save_filename_prefix + 'trajectories.eps', format='eps')
    print('Saved trajectories: %s' % (FLAGS.save_filename_prefix + 'trajectories.[png|eps]'))

  # Plot distributions.
  plt.figure()
  left_dataset = []
  for t in left_trajectories:
    left_dataset.append(t[1:FLAGS.ignore_after, :])
  left_dataset = np.vstack(left_dataset)
  right_dataset = []
  for t in right_trajectories:
    right_dataset.append(t[1:FLAGS.ignore_after, :])
  right_dataset = np.vstack(right_dataset)
  plt.scatter(left_dataset[:, 0], left_dataset[:, 1], color='lightcoral')
  plt.scatter(right_dataset[:, 0], right_dataset[:, 1], color='lightgray')
  plt.xlabel('position')
  plt.xlabel('speed')
  if FLAGS.save_filename_prefix:
    plt.savefig(FLAGS.save_filename_prefix + 'distributions.png', format='png')
    plt.savefig(FLAGS.save_filename_prefix + 'distributions.eps', format='eps')
    print('Saved distributions: %s' % (FLAGS.save_filename_prefix + 'distributions.[png|eps]'))

  # Compute 2D KS-distance.
  bins = 20
  xmin = min(np.min(left_dataset[:, 0]), np.min(right_dataset[:, 0]))
  xmax = max(np.max(left_dataset[:, 0]), np.max(right_dataset[:, 0]))
  ymin = min(np.min(left_dataset[:, 1]), np.min(right_dataset[:, 1]))
  ymax = max(np.max(left_dataset[:, 1]), np.max(right_dataset[:, 1]))
  pdf_left, x, y = np.histogram2d(left_dataset[:, 0], left_dataset[:, 1], bins=bins, range=[[xmin, xmax], [ymin, ymax]])
  pdf_right, _, _ = np.histogram2d(right_dataset[:, 0], right_dataset[:, 1], bins=bins, range=[[xmin, xmax], [ymin, ymax]])
  pdf_left = pdf_left.T
  pdf_right = pdf_right.T
  cdf_left = IntegralImage(pdf_left)
  cdf_right = IntegralImage(pdf_right)
  pdf_left /= float(left_dataset.shape[0])
  pdf_right /= float(right_dataset.shape[0])
  cdf_left /= float(left_dataset.shape[0])
  cdf_right /= float(right_dataset.shape[0])
  ks_distance = np.max(np.abs(cdf_left - cdf_right))
  pdf_distance = np.max(np.abs(pdf_left - pdf_right))
  uniform_pdf = np.ones_like(pdf_left) / float(pdf_left.shape[0] * pdf_left.shape[1])
  smooth_leakage = np.max(np.abs(np.log(pdf_left * (1. - _NU) + uniform_pdf * _NU) - np.log(pdf_right * (1. - _NU) + uniform_pdf * _NU)))
  print('Empirical KS-distance: %.3f' % ks_distance)
  print('Empirical PDF-distance: %.3f' % pdf_distance)
  print('Empirical leakage: %.3f' % smooth_leakage)
  if FLAGS.save_filename_prefix:
    with open(FLAGS.save_filename_prefix + 'info.txt', 'w') as fp:
      fp.write('Empirical KS-distance: %.3f\n' % ks_distance)
      fp.write('Empirical KS-distance: %.3f\n' % pdf_distance)
      fp.write('Empirical leakage: %.3f\n' % smooth_leakage)
      print('Saved information: %s' % (FLAGS.save_filename_prefix + 'info.txt'))

  if not FLAGS.disable_plot:
    plt.show()


if __name__ == '__main__':
  Run()
