import numpy as np


# Stores dense zero-based keys with values corresponding to priorities.
# Uses a heap internally. Overall, the heap order to almost sorted.
class PseudoSorter(object):

  def __init__(self, max_capacity, balancing_interval=None):
    assert max_capacity > 0
    self.size = 0
    self.heap = np.empty(max_capacity, dtype=np.int32)
    self.key_indices = np.empty(max_capacity, dtype=np.int32)
    self.key_priorities = np.empty(max_capacity, dtype=np.float32)
    self.max_capacity = max_capacity
    assert balancing_interval is None or balancing_interval > 0
    self.balancing_interval = balancing_interval or max_capacity
    self.updates_since_last_balancing = 0

  def Update(self, key, priority):
    assert key < self.max_capacity, 'Cannot go beyond capacity.'
    assert key <= self.size, 'Elements must be inserted in order.'
    self.updates_since_last_balancing += 1
    if self.updates_since_last_balancing >= self.balancing_interval:
      self._Balance()
      self.updates_since_last_balancing = 0
    if key == self.size:
      self.key_priorities[key] = priority
      self.key_indices[key] = self.size
      self.heap[self.size] = key
      self.size += 1
      self._Up(self.size - 1)
      return
    # Updating value.
    self.key_priorities[key] = priority
    self._UpOrDown(self.key_indices[key])

  def GetByRank(self, rank_heap):
    return self.heap[rank_heap]

  def Max(self):
    if self.size == 0:
      return None, None
    return self.heap[0], self.key_priorities[self.heap[0]]

  def Save(self, pickler):
    pickler.dump(self.size)
    pickler.dump(self.heap)
    pickler.dump(self.key_indices)
    pickler.dump(self.key_priorities)
    pickler.dump(self.max_capacity)
    pickler.dump(self.updates_since_last_balancing)

  def Load(self, unpickler):
    self.size = unpickler.load()
    self.heap = unpickler.load()
    self.key_indices = unpickler.load()
    self.key_priorities = unpickler.load()
    self.max_capacity = unpickler.load()
    self.updates_since_last_balancing = unpickler.load()

  def _Balance(self):
    # Resort according to priorities.
    sorted_indices = self.key_priorities[:self.size].argsort()[::-1]
    self.heap[:self.size] = sorted_indices
    self.key_indices[self.heap[:self.size]] = np.arange(self.size)

  def _UpOrDown(self, i):
    # Try to move up.
    if not self._Up(i):
      self._Down(i)

  def _Up(self, i):
    current_index = i
    parent_index = _Parent(current_index)
    ret = False
    while current_index > 0:
      if self._GetPriority(current_index) > self._GetPriority(parent_index):
        self._Swap(current_index, parent_index)
        current_index = parent_index
        parent_index = _Parent(current_index)
        ret = True
      else:
        break
    return ret

  def _Down(self, i):
    current_index = i
    left_index = _LeftChild(current_index)
    right_index = _RightChild(current_index)
    ret = False
    while self._Exists(left_index):
      if self._GetPriority(current_index) < self._GetPriority(left_index):
        swap_with = left_index
      elif self._Exists(right_index) and self._GetPriority(current_index) < self._GetPriority(right_index):
        swap_with = right_index
      else:
        break
      ret = True
      self._Swap(current_index, swap_with)
      current_index = swap_with
      left_index = _LeftChild(current_index)
      right_index = _RightChild(current_index)
    return ret

  def _Exists(self, i):
    return i < self.size

  def _GetPriority(self, i):
    return self.key_priorities[self.heap[i]]

  def _Swap(self, i, j):
    t = self.heap[i]
    self.heap[i] = self.heap[j]
    self.heap[j] = t
    t = self.key_indices[self.heap[i]]
    self.key_indices[self.heap[i]] = self.key_indices[self.heap[j]]
    self.key_indices[self.heap[j]] = t


def _Parent(i):
  return (i - 1) >> 1


def _LeftChild(i):
  return (i << 1) + 1


def _RightChild(i):
  return (i << 1) + 2
