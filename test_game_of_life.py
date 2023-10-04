import pytest
import numpy as np
import random

def test_game_of_life():
  from game_of_life import game_of_life_next_step
  SEED = 21
  random.seed(SEED)
  np.random.seed(SEED)
  step_array = np.random.randint(0, 2, (25, 25))
  for i in range(100):
      step_array = game_of_life_next_step(step_array)
  assert len(step_array[step_array == 1]) == 24
  assert np.where(step_array.reshape(-1) != 0)[0][-7] == 539
