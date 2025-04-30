#!/usr/bin/python3

import numpy as np

from matplotlib import cm
from matplotlib import pyplot as plt
from functions import function_factory
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter, LogFormatterSciNotation, LogitFormatter
from numpy import arange, abs, cos, exp, mean, pi, prod, sin, sqrt, sum


def contour(function):
    func = function_factory(function)
    f = func.func

    low = func.low
    high = func.high

    for r in [0.05, 0.1, 0.2, 0.5, 0.7, 1, 1.5, 2]:
        #  high = 1
        #  low = -1

        x = arange(low, high, r)
        y = arange(low, high, r)

        # Grid of points
        X, Y = np.meshgrid(x, y)

        Z = X.copy()

        if (len(x) * len(y)) > 800000:
            continue
        else:
            break

    for i in range(len(x)):
        for j in range(len(y)):
            Z[i][j] = f(np.array((x[i], y[j])))

    #  Z = func.func(X, Y)

    fig, ax=plt.subplots(1,1)


    #  Contour Plot
    
    #  Filled
    #  cp = plt.contourf(X, Y, Z, cmap='RdBu_r')
    #  plt.clabel(cp, inline=False, fmt='%1.0f', colors='white', fontsize=10)

    #  Filled using imshow
    ax.imshow(Z, cmap='RdBu_r', extent=[func.low, func.high, func.low, func.high], aspect=1)
    cp = plt.contour(X, Y, Z, cmap='RdBu_r')
    plt.clabel(cp, inline=True, fmt='%1.0f', fontsize=10)

    opt_x, opt_y = func.xopt
    plt.plot([opt_x], [opt_y], c='white', marker='*')

    #  Adding the colobar on the right
    plt.colorbar(cp)

    #  #  Latex fashion title
    #  #  title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')

    # --- Function surface ---
    plt.figure()
    ax = plt.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap='RdBu_r',linewidth=0, antialiased=True)


    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.title(function.capitalize())
    plt.colorbar(surf, shrink=0.3, aspect=5)

    plt.show()

functions = ['f' + str(i) for i in range(17, 24)]

for f in functions:
    test = function_factory(f)
    if test.dim != np.inf and test.dim != 2:
        print('Skiping', f,' dim > 2')
        continue

    contour(f)

