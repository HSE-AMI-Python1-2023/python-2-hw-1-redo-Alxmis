import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_2():
    from diff_evolution import differential_evolution
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)

    def rastrigin(array, A=10):
        return A * 2 + (array[0] ** 2 - A * np.cos(2 * np.pi * array[0])) + (array[1] ** 2 - A * np.cos(2 * np.pi * array[1]))
  
    def griewank(array):
        term_1 = (array[0] ** 2 + array[1] ** 2) / 2
        term_2 = np.cos(array[0]/ np.sqrt(2)) * np.cos(array[1]/ np.sqrt(2))
        return 1 + term_1 - term_2
  
    def rosenbrock(array):
        return (1 - array[0]) ** 2 + 100 * (array[1] - array[0] ** 2) ** 2

    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.0007965913468088421
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.003549742176257003
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.238660013196448e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  4.1099679882772574e-08
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.942063862893974e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0012017547047786792
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  8.881784197001252e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  6.572520305780927e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.7769787297747826e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  5.508860034808549e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0

