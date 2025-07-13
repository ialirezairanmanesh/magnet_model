#!/usr/bin/python3

import numpy as np
import helpers
import time
from pprint import pprint
from pirates import Pirates
from functions import function_factory

"""
Author: Milad Abolhassani

Main file of Pirates optimization algorithm

"""

def main():
    """
    Runs the base pirate clsas with pre defined arguments

    """

    functions = ['f' + str(i) for i in range(14, 24)]
    functions = ['f1']

    # Weights of leader, private map, map, top ships
    c = { 'leader' : 2.0, 'private_map' : 1.5, 'map' : 1.5, 'top_ships' : 1.5}

    #  Set more dynamic parameters here:
    d = 2
    max_r = 1
    max_wind = 1.0
    top_ships = 2
    num_ships = 50
    map_size = 1
    p_hr = 0.001

    num_run = 1
    max_iter = 200

    log = False
    animate = False
    dynamics = False
    iteration_plots = False
    normalize_results = False
    plot_curves_in_multi_run = True

    #  Quiet disables: results, plots, logs, animation
    #  Usage: while num_run > 1
    #  To disable logs, use log = 0
    quiet = True if (num_run > 1 or len(functions) > 1) else False

    # Change the behavior based on quiet
    animate = False if quiet else animate

    for function in functions:

        test = function_factory(function)

        # Review dimension
        if test.dim != np.inf:
            d = test.dim

        print('-'*50)
        print('Configuration')
        print('-'*50)
        print('Function:', function, ' --- Name:', test.name)
        print()
        print('x opt is:', test.xopt)
        print('f opt is:', test.fopt)
        print('Domain is', test.low, test.high)
        print('-'*50)
        print('Quiet:', quiet)
        print('Animate:', animate)
        print('-'*50)
        print('C', c)
        print('wind', max_wind)
        print('d', d)
        print('r', max_r)
        print('num_ships', num_ships)
        print('top_ships', top_ships)
        print('map_size', map_size)
        print('p_hr', p_hr)
        print('maxIter', max_iter)
        print('num_run', num_run)
        print('-'*50)
        print()

        results = []
        results_v = []
        results_w = []
        results_a = []
        results_f = []
        results_bsf = []
        results_avg = []
        results_trajectory = []

        for _ in range(num_run):

            optz = Pirates(test, dimensions=d, num_ships=num_ships, top_ships=top_ships,
                           fmax=((test.high),)*d, fmin=((test.low),)*d,
                           max_wind=max_wind, max_r=max_r, hr=p_hr, ms=map_size, max_iter=max_iter,
                           dynamic_sails=dynamics, log=log, iteration_plots=iteration_plots,
                           animate=animate, quiet=quiet, c=c)

            # Process is quiet, we also need averages!
            if num_run > 1:
                # Best so far's cost at this run
                results.append(optz.bsf_list[-1])

                if plot_curves_in_multi_run:
                    # Add bsf_list of this run
                    results_bsf.append(optz.bsf_list)

                    # Add avg of this run
                    results_avg.append(optz.avg)

                    # Add trajectory
                    results_trajectory.append(optz.trajectory)

                    # Add velocity and wind
                    results_v.append(optz.v_history)
                    results_w.append(optz.w_history)
                    results_a.append(optz.a_history)
                    results_f.append(optz.f_history)

        # When optimizing multiple functions at same time, pirates class
        # works at quiet mode, so print best result here
        if num_run == 1 and len(functions) > 1:
            print(optz.bsf_list[-1], '\n'*3)

        # Print results of multiple run
        if num_run > 1:

            # Min Max Normalization between [0, 1]
            if normalize_results:
                # When all results are zero
                if  np.sum(results) == 0:
                    results = [0] * 30
                # When results are greater than zero but all are the same
                elif len(np.unique(results)) == 1:
                    pass
                # Normal situation - results are vary fro each other
                else:
                    results = (results - np.min(results)) / (np.max(results) - np.min(results))

            print("Avg: {:>40}".format(np.mean(results)))
            print("STD: {:>40}".format(np.std(results)))
            print("MED: {:>40}".format(np.median(results)))
            print("BST: {:>40}".format(np.amin(results)))
            print("WRS: {:>40}".format(np.amax(results)))
            print()
            print()
            results.sort()
            print('Best result on each run: (sorted)\n', results)
            print()
            print('-'*50)
            print()

            if plot_curves_in_multi_run:

                # Measure means on multiple runs
                results_v = np.mean(np.array(results_v), axis=0)
                results_w = np.mean(np.array(results_w), axis=0)
                results_a = np.mean(np.array(results_a), axis=0)
                results_f = np.mean(np.array(results_f), axis=0)
                results_bsf = np.mean(np.array(results_bsf), axis=0)
                results_avg = np.mean(np.array(results_avg), axis=0)
                results_trajectory = np.mean(np.array(results_trajectory), axis=0)

                data = [results_bsf, results_avg, results_trajectory, results_v, results_w, results_a, results_f]

                # Save data for next useages
                np.savetxt('./csv/Pirates_' + function + '_bsf_.csv', results_bsf, delimiter=',')
                np.savetxt('./csv/Pirates_' + function + '_avg_.csv', results_avg, delimiter=',')
                np.savetxt('./csv/Pirates_' + function + '_tra_.csv', results_trajectory, delimiter=',')

                helpers.plot_results_curves(data, title='AVG BFS Trajectory - Function: '
                                            + function.capitalize() + ' at ' + str(num_run) + ' run '
                                            + str(d) + ' dimensions.')

if __name__ == '__main__':
    main()
