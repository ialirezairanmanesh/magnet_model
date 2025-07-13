#!/usr/bin/python3

from numpy import genfromtxt
import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt

# Dimensino [n, fix]
d = 'fix'
#  d = 'n'

if d == 'n':
    # N-Dimensions
    functions = ['f' + str(i) for i in range(1, 14)]
    DIR = '2-d'
else:
    # Fixed dimensions
    functions = ['f' + str(i) for i in range(14, 24)]
    #  functions = ['f14', 'f15', 'f17', 'f18']
    DIR = 'fix-d'

# List of algorithms results
algorithms = ['Pirates']#, 'PSO', 'GA', 'GSA']#, 'GWO']

ITER = 500

def plot_results(TYPE='BSF'):
    plt.figure()
    ax = plt.gca()

    if TYPE == 'BSF':
        title = 'Best so far'
    elif TYPE == 'AVG':
        title = 'Average'
    elif TYPE == 'TRA':
        title = 'Trajectory'

    title = ''

    plt.title(title + ' ' + function.upper())

    linecycler = cycle(['-', ':', '--', '-.'])
    #  colors = {'BSF': 'gold', 'AVG': 'coral', 'TRA': 'lightgreen'}

    for algorithm in algorithms:
        if TYPE != 'TRA' and not any(data[algorithm][TYPE] < 0):
            ax.set_yscale('log')

        plt.plot(data[algorithm][TYPE][:ITER], label=algorithm, linestyle=next(linecycler))#, c=colors[TYPE])

    plt.xlabel('Iterations')
    plt.ylabel('Position')
    #  plt.legend()
    plt.show()

for function in functions:

    # Read and store results
    data = {}
    for algorithm in algorithms:
        data[algorithm] = {}
        #  data[algorithm]['BSF'] = genfromtxt('./csv/' + DIR + '/' + algorithm + '_' + function + '_bsf_.csv', delimiter=',')
        #  data[algorithm]['AVG'] = genfromtxt('./csv/' + DIR + '/' + algorithm + '_' + function + '_avg_.csv', delimiter=',')
        data[algorithm]['TRA'] = genfromtxt('./csv/' + DIR + '/' + algorithm + '_' + function + '_tra_.csv', delimiter=',')

    #  plot_results('BSF')
    #  plot_results('AVG')
    plot_results('TRA')
