import pytest
import numpy as np
import random

def test_einsum():
  from einsum_task import task_00, task_01, task_02, task_03
  SEED = 21
  random.seed(SEED)
  np.random.seed(SEED)
  A = np.random.uniform(0, 1, 27)
  B = np.random.uniform(0, 1, 27)
  assert task_00(A) == 13.11339235591917
  assert task_01(A, B)[-5] == 0.24026149540094477
  assert task_02(A, B) == 8.195894577699882
  assert task_03(A, B)[-7][-2] == 0.12503313510400518
