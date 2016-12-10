import numpy as np
import sorter


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
    self.buffer_actions[i, ...] = action
    self.buffer_observations[i, ...] = observation
    self.buffer_next_observations[i, ...] = next_observation
    self.buffer_rewards[i] = reward
    self.buffer_done[i] = done
    self.current_index = int((i + 1) % self.max_capacity)
    self.size = int(max(i + 1, self.size))  # Maxes out at max_capacity.

  def __len__(self):
    return self.size

  def Sample(self, n):
    assert n <= self.size, 'Replay memory contains less than %d elements.' % n
    self.indices, weights = self.SampleIndicesAndWeights(n)
    return (self.buffer_actions[self.indices, ...],
            self.buffer_observations[self.indices, ...],
            self.buffer_rewards[self.indices],
            self.buffer_done[self.indices],
            self.buffer_next_observations[self.indices, ...],
            weights)

  def Update(self, new_priorities):
    pass


class Uniform(ReplayMemory):
  def __init__(self, max_capacity, action_shape, observation_shape):
    super(Uniform, self).__init__(max_capacity, action_shape, observation_shape)
    self.uniform_weights = None

  def SampleIndicesAndWeights(self, n):
    if self.uniform_weights is None:
      self.uniform_weights = np.ones(n) / n
    return np.random.choice(self.size, n, replace=False), self.uniform_weights


_ALPHA = 0.7       # Priority exponent (0 == uniform, 1 == priority weighted).
_BETA_START = 0.5  # Importance-sampling (begin of experiment).
_BETA_END = 1.0    # Importance-sampling (end of experiment).
_STEPS_TO_REACH_MAX_BETA = 100000  # Beta saturates after this amount of steps.
_EPSILON = 1e-10
# When using rank-based priority, it might be good to reduce the learning rate (e.g., by 4).


class RankBased(ReplayMemory):

  def __init__(self, max_capacity, action_shape, observation_shape):
    super(RankBased, self).__init__(max_capacity, action_shape, observation_shape)
    self.sorter = sorter.PseudoSorter(max_capacity, balancing_interval=max_capacity)
    self.start_step = None
    self.current_step = 0
    self.computed_density_for = None

  def Add(self, *args):
    # The new elements is placed at index self.current_index in the sliding window.
    inserted_index = self.current_index
    super(RankBased, self).Add(*args)  # Store in rolling window.
    _, max_priority = self.sorter.Max()
    max_priority = max_priority or 0.
    self.sorter.Update(inserted_index, max_priority + _EPSILON)  # Add to priority queue (at the top).
    self.current_step += 1

  def SampleIndicesAndWeights(self, n):
    # Training has started.
    if self.start_step is None:
      self.start_step = self.current_step
    # Sampling according to weight P_i = p_i^alpha / Sum_j p_j^alpha where p_i = 1 / rank(i).
    self._BuildCumulativeDensity(self.size, n)  # Lazily built.
    ranks = np.empty(n, dtype=np.int32)
    for i, (segment_start, segment_end) in enumerate(self.distribution_segments):
      ranks[i] = np.random.randint(segment_start, segment_end)
    indices = self.sorter.GetByRank(ranks)
    # Weight for the batch update are w_i = (1 / N * 1 / P_i)^beta.
    beta = _BETA_START + (float(self.current_step - self.start_step) /
                          float(_STEPS_TO_REACH_MAX_BETA - self.start_step) * (_BETA_END - _BETA_START))
    beta = min(beta, _BETA_END)
    # Since we normalize anyways afterwards there is no need to normalize the priorities.
    priorities = self.unnormalized_precomputed_priorities[ranks]
    weights = np.power(priorities * self.size, -beta)
    weights /= np.max(weights)
    return indices, weights

  def _BuildCumulativeDensity(self, n, k):
    # Memorize the previous size for which it was computed.
    if self.computed_density_for == (n, k):
      return
    self.computed_density_for = (n, k)
    rank = np.arange(1, n + 1)
    self.unnormalized_precomputed_priorities = np.power(rank, -_ALPHA)
    # Build k segments (usually k == batch size).
    cdf = np.cumsum(self.unnormalized_precomputed_priorities)
    cdf /= cdf[-1]
    # Cut cdf in k segments.
    next_limit = 1. / k
    current_index = 0
    previous_segment_end = 0
    segments = []
    for current_segment in range(k):
      while current_index < len(cdf) and cdf[current_index] < next_limit:
        current_index += 1
      assert previous_segment_end <= current_index
      current_index += 1
      segments.append((previous_segment_end, current_index))
      previous_segment_end = current_index
      next_limit += 1. / k
      # Make sure there are enough indices left.
      segments_left = k - current_segment - 1
      indices_left = len(cdf) - current_index
      assert segments_left <= indices_left
      if segments_left == indices_left:
        # Assign all remaining segments.
        for _ in range(segments_left):
          segments.append((current_index, current_index + 1))
          current_index += 1
        break
    self.distribution_segments = segments
