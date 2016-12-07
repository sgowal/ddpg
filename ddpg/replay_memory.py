import numpy as np


class ReplayMemory(object):

  def __init__(self, max_capacity, action_shape, observation_shape):
    self.size = 0
    self.max_capacity = max_capacity
    self.buffer_actions = np.empty((max_capacity,) + tuple(action_shape), dtype=np.float32)
    self.buffer_observations = np.empty((max_capacity,) + tuple(observation_shape), dtype=np.float32)
    self.buffer_next_observations = np.empty((max_capacity,) + tuple(observation_shape), dtype=np.float32)
    self.buffer_rewards = np.empty((max_capacity,), dtype=np.float32)
    self.buffer_done = np.empty((max_capacity,), dtype=np.bool)
    self.current_index = 0

  def Add(self, action, observation, reward, done, next_observation):
    i = self.current_index
    self.buffer_actions[i, :] = action
    self.buffer_observations[i, :] = observation
    self.buffer_next_observations[i, :] = next_observation
    self.buffer_rewards[i] = reward
    self.buffer_done[i] = done
    self.current_index = int((i + 1) % self.max_capacity)
    self.size = int(max(i + 1, self.size))  # Maxes out at max_capacity.

  def __len__(self):
    return self.size

  def Sample(self, n):
    assert n <= self.size, 'Replay memory contains less than %d elements.' % n
    indices = np.random.choice(self.size, n, replace=False)
    return (self.buffer_actions[indices, :],
            self.buffer_observations[indices, :],
            self.buffer_rewards[indices],
            self.buffer_done[indices],
            self.buffer_next_observations[indices, :])
