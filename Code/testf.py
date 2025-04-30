#!/usr/bin/python3

import numpy as np
from functions import function_factory

def ver_test(func):
    x = test.xopt
    print('Optimum x is:', test.xopt)
    print('Optimum should be:', test.fopt)
    print('Optimum is:', test.func(test.xopt))
    print('-'*20)

for i in range(1, 24):
    test = function_factory('f' + str(i))
    if test.func(test.xopt) == test.fopt:
        pass
        #  print('F'+str(i), 'test pass.')
    else:
        print('F'+str(i), 'test failed.')
        ver_test(test)
        print()

#  x = np.array([1.0117801114065029383e-19, 5.4731674881338802705e-20], dtype=np.longdouble)
#  x = [1, 2, -1, -1, -1]
#  x = [-1, -1]
#  test = function_factory('f12')
#  print(test.func(x))
