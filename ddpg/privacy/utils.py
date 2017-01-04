# WARNING: Code is specific to Private-Cart-v0 at this point.

import matplotlib.pylab as plt
import numpy as np
import pickle

# Leakage smoother. Corresponds to the likelihood of observing something fully random.
_NU = 5e-1


class TrajectoryAnalyzer(object):

  def __init__(self, trajectories=None, labels=None, cutoff=40):
    self.trajectories = trajectories
    self.labels = labels
    self.cutoff = cutoff

  # Loading and saving are a bit convoluted due to backward compatility.
  def Save(self, filename):
    with open(filename, 'wb') as fp:
      pickler = pickle.Pickler(fp, -1)
      for t in self.trajectories:
        pickler.dump(t)
      for l in self.labels:
        pickler.dump(l)

  def Load(self, filename):
    with open(filename, 'rb') as fp:
      unpickler = pickle.Unpickler(fp)
      self.trajectories = []
      self.labels = []
      while True:
        try:
          t = unpickler.load()
          # Must be a label.
          if isinstance(t, str):
            self.labels.append(t)
          else:
            self.trajectories.append(t)
        except EOFError:
          break
    # Special backward compatible case. Assumes Private-Cart-v0 environment.
    if not self.labels and len(self.trajectories) == 2:
      self.labels.extend(['Left', 'Right'])
    assert len(self.labels) == len(self.trajectories), 'Wrong file format.'

  def PlotTrajectories(self):
    plt.figure()
    colors = _GetColors(len(self.trajectories))
    for trajectories, color in zip(self.trajectories, colors):
      for t in trajectories:
        plt.plot(t[:, 0], np.arange(t.shape[0]) * 0.02, color=color)
    plt.plot([0.8, 0.8], [0, 1.6], 'k--')
    plt.plot([-0.8, -0.8], [0, 1.6], 'k--')
    plt.ylim([0, 1.6])
    plt.xlim([-1., 1.])
    plt.xlabel('position')
    plt.ylabel('time')

  def PlotDistributions(self):
    plt.figure()
    datasets = self._BuildDistributions()
    colors = _GetColors(len(datasets))
    for dataset, color in zip(datasets, colors):
      plt.scatter(dataset[:, 0], dataset[:, 1], color=color)
    plt.xlabel('position')
    plt.ylabel('speed')

  def ComputeMetrics(self, nbins=20):
    datasets = self._BuildDistributions()
    left_dataset, right_dataset = datasets[0], datasets[1]
    xmin = min(np.min(left_dataset[:, 0]), np.min(right_dataset[:, 0]))
    xmax = max(np.max(left_dataset[:, 0]), np.max(right_dataset[:, 0]))
    ymin = min(np.min(left_dataset[:, 1]), np.min(right_dataset[:, 1]))
    ymax = max(np.max(left_dataset[:, 1]), np.max(right_dataset[:, 1]))
    pdf_left, x, y = np.histogram2d(left_dataset[:, 0], left_dataset[:, 1], bins=nbins, range=[[xmin, xmax], [ymin, ymax]])
    pdf_right, _, _ = np.histogram2d(right_dataset[:, 0], right_dataset[:, 1], bins=nbins, range=[[xmin, xmax], [ymin, ymax]])
    pdf_left = pdf_left.T
    pdf_right = pdf_right.T
    cdf_left = _IntegralImage(pdf_left)
    cdf_right = _IntegralImage(pdf_right)
    pdf_left /= float(left_dataset.shape[0])
    pdf_right /= float(right_dataset.shape[0])
    cdf_left /= float(left_dataset.shape[0])
    cdf_right /= float(right_dataset.shape[0])
    ks_distance = np.max(np.abs(cdf_left - cdf_right))
    pdf_distance = np.max(np.abs(pdf_left - pdf_right))
    uniform_pdf = np.ones_like(pdf_left) / float(pdf_left.shape[0] * pdf_left.shape[1])
    smooth_leakage = np.max(np.abs(np.log(pdf_left * (1. - _NU) + uniform_pdf * _NU) - np.log(pdf_right * (1. - _NU) + uniform_pdf * _NU)))
    return {
        'Empirical KS-distance': ks_distance,
        'Empirical PDF-distance': pdf_distance,
        'Empirical leakage': smooth_leakage,
    }

  def _BuildDistributions(self):
    datasets = []
    for trajectories in self.trajectories:
      dataset = []
      for t in trajectories:
        dataset.append(t[1:self.cutoff, :])
      dataset = np.vstack(dataset)
      datasets.append(dataset)
    return datasets


def _IntegralImage(x):
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


def _GetColors(n):
  cm = plt.get_cmap('gist_rainbow')
  return [cm(float(i) / float(n)) for i in range(n)]
