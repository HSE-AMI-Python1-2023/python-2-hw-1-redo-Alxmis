import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_3():
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
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  9.08373646547208e-06
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.1283725015340844
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  0.006731723073572758

    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.942063862893974e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  8.881784197001252e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  3.2196574295539904e-09

    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.7769787297747826e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  1.4921397450962104e-13



