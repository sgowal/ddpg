import manager
import privacy


def Start(environment, agent, output_directory, options, restore=False):
  try:
    environment.IsPrivate()
    m = privacy.EnvironmentPrivacyManager(environment, agent, output_directory, options, restore=restore)
  except AttributeError:
    m = manager.Manager(environment, agent, output_directory, options, restore=restore)
  m.Run()
