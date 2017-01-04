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
flags.DEFINE_integer('run_many', 1, 'The whole training pipeline can be repeated many times if needed (sequentially).')
flags.DEFINE_string('options', None, 'ddpg.Options protocol buffer in ASCII format.')
flags.DEFINE_string('option_variants', None,
                    'ddpg.OptionVariants protocol buffer in ASCII format. Runs the pipeline '
                    'for each option variant sequentially. The --options field is used as default values.')
flags.DEFINE_string('option_variants_filename', None,
                    'Same as --option_variants but reads the options from a file.')
FLAGS = flags.FLAGS


# Logging.
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

_SINGLE_VARIANT = '__single_variant__'
_OPTIONS_FILENAME = 'options.pbtxt'


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

  if FLAGS.option_variants_filename:
    assert FLAGS.option_variants is None, 'Cannot specify both --option_variants and --option_variants_filename.'
    with open(FLAGS.option_variants_filename) as fp:
      FLAGS.option_variants = fp.read()

  if not FLAGS.restore:
    if not CreateDirectory(FLAGS.output_directory, FLAGS.force):
      return
  else:
    assert FLAGS.run_many == 1, '--run_many must be equal to 1 when --restore is set.'
    assert FLAGS.option_variants is None, '--option_variants cannot be used with --restore.'

  # Read options.
  options = ddpg.Options()
  if FLAGS.options:
    google.protobuf.text_format.Merge(FLAGS.options, options)
  option_variants = {}
  if FLAGS.option_variants:
    variants = ddpg.OptionVariants()
    google.protobuf.text_format.Merge(FLAGS.option_variants, variants)
    for v in variants.variant:
      assert re.match(r'^[A-Za-z0-9_]+$', v.name), 'Variant name cannot be "%s", it must match [A-Za-z0-9_]+' % v.name
      o = ddpg.Options()
      o.CopyFrom(options)
      o.MergeFrom(v.options)
      option_variants[v.name] = o
  else:
    option_variants[_SINGLE_VARIANT] = options

  for variant_name, variant_options in option_variants.iteritems():
    variant_directory = (FLAGS.output_directory if variant_name == _SINGLE_VARIANT else
                         os.path.join(FLAGS.output_directory, variant_name))
    if variant_name != _SINGLE_VARIANT and not CreateDirectory(variant_directory):
      return

    # Store the options used in plain text. When restoring and if the options are different
    # warn the user. TODO: Compare the fields and not the ASCII format.
    options_filename = os.path.join(variant_directory, _OPTIONS_FILENAME)
    new_options = str(variant_options)
    if FLAGS.restore:
      with open(options_filename) as fp:
        previous_options = fp.read()
      if new_options != previous_options:
        LOG.warn('New options used for --restore are different from the previously used options.')
    with open(options_filename, 'w') as fp:
      fp.write(str(new_options))

    for run_index in range(FLAGS.run_many):
      output_directory = (os.path.join(variant_directory, 'run_%03d' % run_index) if FLAGS.run_many > 1 else
                          variant_directory)
      checkpoint_directory = os.path.join(output_directory, 'checkpoints')
      if not FLAGS.restore and not CreateDirectory(checkpoint_directory):
        return

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

  # Print helper text for plotting results.
  num_levels = int(FLAGS.run_many > 1) + int(FLAGS.option_variants is not None)
  LOG.info('To visualize results: tensorboard --logdir="%s"', FLAGS.output_directory)
  if num_levels == 1:
    if FLAGS.run_many > 1:
      LOG.info('Or plot performance: python analyze_results.py --event_directory="%s" --group_by="run_\d\d\d/test"',
               os.path.join(FLAGS.output_directory, '*'))
    else:
      LOG.info('Or plot performance: python analyze_results.py --event_directory="%s" --group_by="([A-Za-z0-9_]+)/test"',
               os.path.join(FLAGS.output_directory, '*'))
  elif num_levels == 2:
    LOG.info('Or plot performance: python analyze_results.py --event_directory="%s" --group_by="([A-Za-z0-9_]+)/run_\d\d\d/test"',
             os.path.join(FLAGS.output_directory, '*/*'))
  else:
    LOG.info('Or plot performance: python analyze_results.py --event_directory="%s"', FLAGS.output_directory)


if __name__ == '__main__':
  Run()
