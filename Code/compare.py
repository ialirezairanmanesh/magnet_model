#!/usr/bin/python3

import numpy as np
import time
from matplotlib import pyplot as plt
from functions import function_factory
from algorithms.GWO import GWO
from algorithms.DE import DE
from algorithms.PSO import PSO
from algorithms.GA import GA
from algorithms.GSA import GSA

functions = ['f' + str(i) for i in range(1,  24)]

# Fixed dimension functions with positive optimum
#  functions = ['f14', 'f15', 'f17', 'f18']

# List of all availables
algorithms = {'GA': GA, 'PSO': PSO, 'GWO': GWO, 'DE': DE, 'GSA': GSA}

# Our paper is on these ones
algorithms = {'GA': GA, 'PSO': PSO, 'GSA': GSA}

# When we run experiments on one algorithm
#  algorithms = {'GWO': GWO}

for func in functions:
    for alg_name, algorithm in algorithms.items():

        print(alg_name, 'on:', func)
        test = function_factory(func)

        dim = 30
        if test.dim != np.inf:
            dim = test.dim

        pop_size = 50
        max_iter = 500
        num_run = 30

        results = []
        bsf_list = []
        avg_list = []
        for _ in range(num_run):
            try:
                bsf, avg = algorithm(test.func, test.low, test.high, dim, pop_size, max_iter)
                results.append(bsf[-1])
                bsf_list.append(bsf)
                avg_list.append(avg)
            except:
                print('Faild')

        results = np.array(results)
        bsf = np.mean(np.array(bsf_list), axis=0)
        avg = np.mean(np.array(avg_list), axis=0)

        np.savetxt('./csv/' + alg_name + '_' + func + '_bsf_.csv', bsf, delimiter=',')
        np.savetxt('./csv/' + alg_name + '_' + func + '_avg_.csv', avg, delimiter=',')

        print('Mean in', num_run, 'times run is:', np.mean(results))
        print('STD is:', np.std(results))
        print('MED is:', np.median(results))
        print('-'*30)
