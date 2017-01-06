# WARNING: Code is specific to Private-Cart-v0 at this point.

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import scipy
import scipy.spatial
import scipy.stats

_SIGMA_POSITION = 0.05
_SIGMA_SPEED = 0.1


class TrajectoryAnalyzer(object):

  def __init__(self, trajectories=None, labels=None, cutoffs=(35, 45)):
    self.trajectories = trajectories
    self.labels = labels
    self.cutoffs = cutoffs

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
    plt.fill_between([-1., 1., 1., -1.], [self.cutoffs[0] * 0.02, self.cutoffs[0] * 0.02, self.cutoffs[1] * 0.02, self.cutoffs[1] * 0.02], color='lightgray')
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

  def ComputeMetrics(self, nbins=50, plot_leakage=False, plot_ks_distance=False):
    # Privacy.
    datasets = self._BuildDistributions()
    colors = _GetColors(len(datasets) + 1)
    left_dataset, right_dataset = datasets[0], datasets[1]
    xmin = min(np.min(left_dataset[:, 0]), np.min(right_dataset[:, 0]))
    xmax = max(np.max(left_dataset[:, 0]), np.max(right_dataset[:, 0]))
    ymin = min(np.min(left_dataset[:, 1]), np.min(right_dataset[:, 1]))
    ymax = max(np.max(left_dataset[:, 1]), np.max(right_dataset[:, 1]))
    xmargin = (xmax - xmin) * 0.1
    ymargin = (ymax - ymin) * 0.1
    X, Y = np.meshgrid(np.linspace(xmin - xmargin, xmax + xmargin, nbins),
                       np.linspace(ymin - ymargin, ymax + ymargin, nbins))
    XY = np.stack([X, Y], axis=-1)
    pdf_left = np.zeros_like(X)
    for point in left_dataset:
      mu = point[:2]
      var = scipy.stats.multivariate_normal(mean=mu, cov=[[_SIGMA_POSITION, 0], [0, _SIGMA_SPEED]])
      pdf_left += var.pdf(XY)
    pdf_left /= float(left_dataset.shape[0])
    pdf_right = np.zeros_like(X)
    for point in right_dataset:
      mu = point[:2]
      var = scipy.stats.multivariate_normal(mean=mu, cov=[[_SIGMA_POSITION, 0], [0, _SIGMA_SPEED]])
      pdf_right += var.pdf(XY)
    pdf_right /= float(right_dataset.shape[0])
    L = np.abs(np.log(pdf_left) - np.log(pdf_right))
    cdf_left = _NormalizedIntegralImage(pdf_left)
    cdf_right = _NormalizedIntegralImage(pdf_right)
    KS = np.abs(cdf_left - cdf_right)
    # Only consider leakage within the convex hull of points.
    points = np.vstack([left_dataset[:, :2], right_dataset[:, :2]])
    hull = scipy.spatial.ConvexHull(points)
    trig = scipy.spatial.Delaunay(points[hull.vertices, :])
    outside = (trig.find_simplex(XY.reshape(nbins * nbins, 2)) < 0).reshape(nbins, nbins)
    L[outside] = 0.
    if plot_leakage:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot_surface(X, Y, pdf_left, color=colors[0], alpha=0.5, rstride=2, cstride=2)
      ax.plot_surface(X, Y, pdf_right, color=colors[1], alpha=0.5, rstride=2, cstride=2)
      ax.plot_surface(X, Y, L, color=colors[2], alpha=0.5, rstride=2, cstride=2)
      ax.plot(left_dataset[:, 0], left_dataset[:, 1], '+', markersize=.2, color=colors[0], zdir='z', zs=-0.3)
      ax.plot(right_dataset[:, 0], right_dataset[:, 1], '+', markersize=.2, color=colors[1], zdir='z', zs=-0.3)
      for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], '--', color=colors[2], lw=2, zdir='z', zs=-0.3)
      ax.set_xlabel('Position')
      ax.set_ylabel('Speed')
    if plot_ks_distance:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.plot_surface(X, Y, cdf_left, color=colors[0], alpha=0.5, rstride=2, cstride=2)
      ax.plot_surface(X, Y, cdf_right, color=colors[1], alpha=0.5, rstride=2, cstride=2)
      ax.plot_surface(X, Y, KS, color=colors[2], alpha=0.5, rstride=2, cstride=2)
      ax.plot(left_dataset[:, 0], left_dataset[:, 1], '+', markersize=.2, color=colors[0], zdir='z', zs=-0.3)
      ax.plot(right_dataset[:, 0], right_dataset[:, 1], '+', markersize=.2, color=colors[1], zdir='z', zs=-0.3)
      ax.set_xlabel('Position')
      ax.set_ylabel('Speed')
    pdf_distance = np.max(np.abs(pdf_left - pdf_right))
    ks_distance = np.max(KS)
    leakage = np.max(L)
    # Performance.
    datasets, sizes = self._BuildDistributions(before=False, return_size=True)
    left_dataset, right_dataset = datasets[0], datasets[1]
    left_size, right_size = sizes[0], sizes[1]
    left_dist = np.abs(left_dataset[:, 0] + 0.8)
    right_dist = np.abs(right_dataset[:, 0] - 0.8)
    performance = -(np.sum(left_dist) + np.sum(np.square(left_dataset[:, -1]))) / float(left_size)
    performance -= (np.sum(right_dist) + np.sum(np.square(right_dataset[:, -1]))) / float(right_size)
    return {
        'Empirical KS-distance': ks_distance,
        'Empirical PDF-distance': pdf_distance,
        'Empirical leakage': leakage,
        'Performance reward': performance,
    }

  def _BuildDistributions(self, before=True, return_size=False):
    datasets = []
    sizes = []
    for trajectories in self.trajectories:
      dataset = []
      for t in trajectories:
        if before:
          dataset.append(t[1:self.cutoffs[0], :])
        else:
          dataset.append(t[self.cutoffs[1]:, :])
      dataset = np.vstack(dataset)
      datasets.append(dataset)
      sizes.append(len(trajectories))
    if return_size:
      return datasets, sizes
    return datasets


def _NormalizedIntegralImage(x):
  y = np.zeros_like(x)
  y[0, 0] = x[0, 0]
  for i in xrange(1, x.shape[0]):
    y[i, 0] = y[i - 1, 0] + x[i, 0]
  for j in xrange(1, x.shape[1]):
    y[0, j] = y[0, j - 1] + x[0, j]
  for i in xrange(1, x.shape[0]):
    for j in xrange(1, x.shape[1]):
      y[i, j] = y[i - 1, j] + y[i, j - 1] - y[i - 1, j - 1] + x[i, j]
  return y / y[-1, -1]


def _GetColors(n):
  cm = plt.get_cmap('gist_rainbow')
  return [cm(float(i) / float(n)) for i in range(n)]
