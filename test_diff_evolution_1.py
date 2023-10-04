import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_1():
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
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.0030976463647417354
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.0020390331159686175
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.00041926842306164364
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.942063862893974e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  7.703976443451666e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.5745836474678754e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  7.500436325358351e-07
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.7769787297747826e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  5.88462611972318e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.0628165014736624e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  7.244205235679146e-13
