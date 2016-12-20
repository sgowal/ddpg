"""Runs DDPG on a Gym environment."""

from __future__ import print_function

import logging
import google.protobuf.text_format
import gym
try:
  import gym_extras  # Personal Gym environments. Feel free to ignore.
except ImportError:
  pass
import os
import shutil
import sys
import re

# Unfortunate pbr bug :(
try:
  import tensorflow as tf
except:
  print('Try setting PBR_VERSION=1.10.0 in the terminal (export PBR_VERSION=1.10.0).')
  sys.exit(-1)

import ddpg


flags = tf.app.flags
flags.DEFINE_string('environment', None, 'Gym environment.')
flags.DEFINE_string('search', None, 'Lists all Gym environments that correspond to the input regular expression.')
flags.DEFINE_string('output_directory', None, 'Directory where results are stored.')
flags.DEFINE_bool('list', False, 'Shows list of Gym environments.')
flags.DEFINE_bool('force', False, 'Overwrite --output_directory if it already exists.')
flags.DEFINE_bool('restore', False, 'Restore from a previous Run.')
flags.DEFINE_integer('run_many', 1, 'The whole training pipeline can be repeated many times if needed (serially).')

# Flags for training options.
flags.DEFINE_string('options', None, 'ddpg.Options protocol buffer in ASCII format.')
FLAGS = flags.FLAGS


# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def CreateDirectory(directory, force=False):
  if force and os.path.exists(directory):
    LOG.info('Deleting previous directory %s.', directory)
    shutil.rmtree(directory)
  LOG.info('Preparing directory %s.', directory)
  try:
    os.makedirs(directory)
  except os.error:
    print('Cannot create directory %s. Make sure it does not exist (or use --force).' % directory)
    return False
  return True


def ListEnvironments(regexp):
  names = (e.id for e in gym.envs.registry.all())
  if regexp:
    names = (n for n in names if re.match(regexp, n))
  print('List of environments:')
  print('  ' + '\n  '.join(sorted(names)))


def Run():
  if FLAGS.list or FLAGS.search:
    ListEnvironments(FLAGS.search)
    return

  assert FLAGS.output_directory, '--output_directory must be specified.'
  assert FLAGS.environment, '--environment must be specified.'
  assert FLAGS.run_many >= 1, '--run_many must be greater than 1.'

  if not FLAGS.restore:
    if not CreateDirectory(FLAGS.output_directory, FLAGS.force):
      return
  else:
    assert FLAGS.run_many == 1, '--run_many must be equal to 1 when --restore is set.'

  for run_index in range(FLAGS.run_many):
    if FLAGS.run_many > 1:
      output_directory = os.path.join(FLAGS.output_directory, 'run_%03d' % run_index)
    else:
      output_directory = FLAGS.output_directory
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not FLAGS.restore and not CreateDirectory(checkpoint_directory):
      return

    # Read options.
    options = ddpg.Options()
    if FLAGS.options:
      google.protobuf.text_format.Merge(FLAGS.options, options)

    # Create environment.
    environment = gym.make(FLAGS.environment)
    # Create Agent that will interact with the environment.
    agent = ddpg.Agent(environment.action_space, environment.observation_space,
                       checkpoint_directory=checkpoint_directory,
                       options=options, restore=FLAGS.restore)
    # Start experiment.
    ddpg.Start(environment, agent, output_directory, options=options,
               restore=FLAGS.restore)
    # Clear TensorFlow.
    tf.reset_default_graph()
  if FLAGS.run_many > 1:
    LOG.info('To visualize results: tensorboard --logdir="%s"', FLAGS.output_directory)
    LOG.info('Or plot performance: python analyze_results.py --event_directory="%s" --group_by="run_\d\d\d/train,run_\d\d\d/test"',
             os.path.join(FLAGS.output_directory, '*'))


if __name__ == '__main__':
  Run()
