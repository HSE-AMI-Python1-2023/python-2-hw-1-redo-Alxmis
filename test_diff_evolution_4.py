import pytest
import numpy as np
import random
from scipy.stats import qmc
import math

def test_diff_evolution_part_4():
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


    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.7769787297747826e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  1.4921397450962104e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  5.508860034808549e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  2.19824158875781e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  5.775269151797602e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  5.88462611972318e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  1.1604051053382136e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  5.001028924311868e-09
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  1.5987211554602254e-14
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.5216827797814858e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.0628165014736624e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  5.626610288800293e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  2.047652714054493e-09
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  6.661338147750939e-16
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.0405909467436913e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  7.244205235679146e-13
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  1.2047030040207574e-12
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  6.923435158512348e-10
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  9.259260025373806e-14
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  6.930822582518203e-11
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(griewank, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0

    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  3.942063862893974e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  8.881784197001252e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  3.2196574295539904e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0012017547047786792
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  6.572520305780927e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  1.9841905896100798e-12
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  5.5621619701184954e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  8.881784197001252e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  6.572520305780927e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  7.703976443451666e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  3.304023721284466e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  6.765331406199948e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0004689764267364893
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  1.6484591469634324e-12
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  1.0698997243707709e-11
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.1549894676221584e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.7763568394002505e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  7.105427357601002e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.9949590570932898
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  1.5745836474678754e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  3.446132268436486e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  3.4560283523887847e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  9.685884661791988e-05
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  1.3731238368563936e-12
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  1.1802114840975264e-11
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  6.547322684014034e-08
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] ==  5.1514348342607263e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  9.592326932761353e-14
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  7.500436325358351e-07
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  1.865174681370263e-13
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  1.8123103018297115e-10
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0006148627983701971
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  5.9452496259382315e-09
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  1.9928596284302103e-06
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] ==  7.105427357601002e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  3.552713678800501e-15
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  0.0
    assert list(differential_evolution(rastrigin, [[-20, 20], [-20, 20]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  0.0

    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.0007965913468088421
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  9.08373646547208e-06
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.1283725015340844
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  0.006731723073572758
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.003549742176257003
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.009596339249381337
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  0.013582716390675553
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  0.0003723484847915667
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.238660013196448e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  5.278776674092618e-13
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  6.915497673257111e-11
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  9.675488345170799e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  4.1099679882772574e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.2979829845261009
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  1.4649690230402043e-10
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='random', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  4.793694517591804e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.0030976463647417354
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  0.026307578551453683
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  3.5045035617174896e-06
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  0.05351428972854671
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0059809187022676034
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  0.004133269703846163
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  5.3886789643112635e-06
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  0.0006010164092425452
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='current'))[-1][1] ==  9.437746368783641e-07
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  1.2571473289969765e-11
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  5.514577362443495e-10
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  3.1631144909250575e-10
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  2.8361158775337283e-07
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  0.088361658864934
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  4.067168994301852e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='LatinHypercube', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  1.0332978152483879e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.0020390331159686175
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  1.0311266719848964e-07
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  0.014103699456838781
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  0.00027427301944102237
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.0024995022609416977
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  7.6219896520782e-06
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  2.3228295757175225e-05
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  5.1496387314570494e-05
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='current'))[-1][1] ==  1.0061809753469817e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  8.881024669398345e-14
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  3.407757461245466e-12
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  5.272256099694732e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  9.316146160955149e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  1.2296254156000707e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  4.121470800721504e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Halton', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  1.6894087371579557e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='current'))[-1][1] ==  0.00041926842306164364
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='worst'))[-1][1] ==  1.3077840204745985e-07
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_among_worst'))[-1][1] ==  4.707423847491145e-05
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand1', selection_setting='random_selection'))[-1][1] ==  6.301343414031285e-05
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='current'))[-1][1] ==  0.00474051345988458
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='worst'))[-1][1] ==  4.853176249439067e-05
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_among_worst'))[-1][1] ==  2.4148403561452993e-05
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand2', selection_setting='random_selection'))[-1][1] ==  0.000415685416674239
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='current'))[-1][1] ==  7.821862824529268e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='worst'))[-1][1] ==  4.4806052747136294e-13
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_among_worst'))[-1][1] ==  5.325667092988643e-11
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='best1', selection_setting='random_selection'))[-1][1] ==  1.2137889673058568e-09
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='current'))[-1][1] ==  6.773232831273338e-08
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='worst'))[-1][1] ==  3.871460274708174e-11
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_among_worst'))[-1][1] ==  9.813134698732711e-10
    assert list(differential_evolution(rosenbrock, [[-2, 2], [-2, 2]], init_setting='Sobol', mutation_setting='rand_to_p_best1', selection_setting='random_selection'))[-1][1] ==  4.399765302816069e-09