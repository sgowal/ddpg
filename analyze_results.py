from __future__ import print_function

import glob
import matplotlib.pylab as plt
import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('event_directory', None, 'Directory where TensorFlow results are stored.')
FLAGS = flags.FLAGS


def Run():
  event_files = list(glob.iglob(os.path.join(FLAGS.event_directory, '**/events.out.tfevents.*')))
  print(event_files)
  # for summary in tf.train.summary_iterator("/path/to/log/file"):
  #   print summary


if __name__ == '__main__':
  Run()
