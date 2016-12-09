import collections

# Options.
Options = collections.namedtuple(
    'Options',
    [
        'max_timesteps_per_episode',
        'evaluate_after_timesteps',
        'max_timesteps',
        'output_directory',
    ])


def ParseFlags(flags):
  return Options(
      flags.max_timesteps_per_episode,
      flags.evaluate_after_timesteps,
      flags.max_timesteps,
      flags.output_directory)
