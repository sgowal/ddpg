"""Runs DDPG on a Gym environment."""

from __future__ import print_function

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

# Unfortunate rendering issue.
# Rendering before importing tensorflow solves the issue.
try:
  env = gym.make('CartPole-v0')
  env.render()  # :(
  del env
except:
  print('Cannot render environments. Make sure the disable_rendering is set to true.')

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

# Flags for training options.
flags.DEFINE_string('options', None, 'ddpg.Options protocol buffer in ASCII format.')
FLAGS = flags.FLAGS


def CreateDirectory(directory, force=False):
  if force and os.path.exists(directory):
    print('Deleting previous directory %s.' % directory)
    shutil.rmtree(directory)
  print('Preparing directory %s.' % directory)
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
  assert FLAGS.output_directory, '--output_directory must be specified'
  assert FLAGS.environment, '--environment must be specified'
  checkpoint_directory = os.path.join(FLAGS.output_directory, 'checkpoints')
  if not FLAGS.restore:
    if not CreateDirectory(FLAGS.output_directory, FLAGS.force):
      return
    if not CreateDirectory(checkpoint_directory):
      return

  # Read options.
  options = ddpg.Options()
  if FLAGS.options:
    google.protobuf.text_format.Merge(FLAGS.options, options)
    print(options)

  # Create environment.
  environment = gym.make(FLAGS.environment)
  # Create Agent that will interact with the environment.
  agent = ddpg.Agent(environment.action_space, environment.observation_space,
                     checkpoint_directory=checkpoint_directory,
                     options=options, restore=FLAGS.restore)
  # Start experiment.
  ddpg.Start(environment, agent, FLAGS.output_directory, options=options)


if __name__ == '__main__':
  Run()
