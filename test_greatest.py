import pytest
import numpy as np
import random

def test_greatest():
  from greatest_task import find_largest_element
  SEED = 21
  random.seed(SEED)
  np.random.seed(SEED)
  random_array = np.arange(0, 1000000, np.random.randint(5, 17))
  np.random.shuffle(random_array)
  assert (find_largest_element(random_array) == np.array([999908, 999922, 999936, 999950, 999964, 999978, 999992])).all()
  assert (find_largest_element(random_array, n=9) == np.array([999880, 999894, 999908, 999922, 999936, 999950, 999964, 999978, 999992])).all()
  assert (find_largest_element(random_array, n=3) == np.array([999964, 999978, 999992])).all()
