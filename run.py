"""Runs DDPG on a Gym environment."""

from __future__ import print_function

import gym
import gym_private
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
flags.DEFINE_string('device', '/cpu:0', 'Device on which to run TensorFlow.')
flags.DEFINE_bool('list', False, 'Shows list of Gym environments.')
flags.DEFINE_bool('force', False, 'Overwrite --output_directory if it already exists.')

# Flags for general options.
flags.DEFINE_integer('max_timesteps_per_episode', 10000, 'Maximum number of timesteps per episode.')
flags.DEFINE_integer('evaluate_after_timesteps', 10000, 'Number of training timesteps between evaluations.')
flags.DEFINE_integer('max_episodes', 1000, 'Maximum number of training episodes.')
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
  if not CreateDirectory(FLAGS.output_directory, FLAGS.force):
    return
  checkpoint_directory = os.path.join(FLAGS.output_directory, 'checkpoints')
  if not CreateDirectory(checkpoint_directory):
    return

  # Create environment.
  environment = gym.make(FLAGS.environment)
  # Create Agent that will interact with the environment.
  agent = ddpg.Agent(environment.action_space, environment.observation_space,
                     checkpoint_directory=checkpoint_directory,
                     device=FLAGS.device)
  # Start experiment.
  options = ddpg.ParseFlags(FLAGS)
  ddpg.Start(environment, agent, options)


if __name__ == '__main__':
  Run()
