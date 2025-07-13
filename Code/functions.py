#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
#  This file contains standard test (benchmark) functions
#
#  Author: Milad Abolhassani
#  Mail: milad[/a/]eng.uk.ac.ir
#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import math
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum

class function_factory():
    def __init__(self, function_name):


        self.data = {
                        #  ---- Unimodal - N-D
                        'f1': { 'bounds':[-100, 100], 'name': 'Sphere', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f2': { 'bounds':[-10, 10], 'name': 'Schwefel2.22', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f3': { 'bounds':[-100, 100], 'name': 'Schwefel1.2', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f4': { 'bounds':[-100, 100], 'name': 'Schwefel2.21', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f5': { 'bounds':[-2.048, 2.048], 'name': 'Rosenbrock', 'fopt': 0, 'xopt': [1, 1], 'dim': np.inf},
                        'f6': { 'bounds':[-100, 100], 'name': 'Step', 'fopt': 0, 'xopt': [-0.5, -0.5], 'dim': np.inf},
                        'f7': { 'bounds':[-1.28, 1.28], 'name': 'Quadratic', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},

                        #  ---- Multimodal - N-D
                        'f8': { 'bounds':[-500, 500], 'name': 'Schwefel2.6', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f9': { 'bounds':[-5.12, 5.12], 'name': 'Rastrigin', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f10': { 'bounds':[-32, 32], 'name': 'Ackley', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f11': { 'bounds':[-600, 600], 'name': 'Griewank', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'f12': { 'bounds':[-5, 5], 'name': 'Generalized Penalized v1', 'fopt': 0, 'xopt': [-1, -1], 'dim': np.inf},
                        'f13': { 'bounds':[-50, 50], 'name': 'Generalized Penalized v2', 'fopt': 0, 'xopt': [1, 1], 'dim': np.inf},

                        #  ---- Multimodal - Fixed Dimensions
                        'f14': { 'bounds':[-65.536, 65.536], 'name': 'Shekel Foxholes', 'fopt': 0.9980038388186224, 'xopt': [-32, 32], 'dim': 2},
                        'f15': { 'bounds':[-5, 5], 'name': 'kowalik', 'fopt': 0.0003074859886558728,
                                'xopt': [0.192833, 0.190836, 0.123117, 0.135766], 'dim': 4},
                        'f16': { 'bounds':[-5, 5], 'name': 'Six-Hump Camel-Back', 'fopt': -1.0316227165016698, 'xopt': [0.089, -0.712], 'dim': 2},
                        'f17': { 'bounds':[-5, 5], 'name': 'branian', 'fopt': 0.3979009112832941, 'xopt': [-3.14, 12.27], 'dim': 2},
                        'f18': { 'bounds':[-2, 2], 'name': 'goldestein price', 'fopt': 3, 'xopt': [0, -1], 'dim': 2},
                        'f19': { 'bounds':[0, 1], 'name': 'hartman no1', 'fopt': -3.86278214782076, 'xopt':
                                [0.1,0.55592003,0.85218259], 'dim': 3},
                        'f20': { 'bounds':[0, 1], 'name': 'hartman no2', 'fopt': -3.32236801141551, 'xopt':
                                [0.20168952,0.15001069,0.47687398,0.27533243,0.31165162,0.65730054], 'dim': 6},

                        # --- under construct ---
                        'f21': { 'bounds':[0, 10], 'name': 'shekel no1 [fixed 4]', 'fopt': -10, 'xopt': [4, 4, 4, 4], 'dim': 4}, # 5 optimum
                        'f22': { 'bounds':[0, 10], 'name': 'shekel no2 [fixed 4]', 'fopt': -10, 'xopt': [4, 4, 4, 4], 'dim': 4}, # 7 optimum
                        'f23': { 'bounds':[0, 10], 'name': 'hartman no3 [fixed 4]', 'fopt': -10, 'xopt': [4, 4, 4, 4], 'dim': 4},# 10 optimum

                        # --- More ---
                        'dixonprice': { 'bounds':[-10, 10], 'name': 'dixonprice', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'levy': { 'bounds':[-10, 10], 'name': 'levy', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'powell': { 'bounds':[-4, 5], 'name': 'powell', 'fopt': 0, 'xopt': [0, -0], 'dim': np.inf},
                        'zakharov': { 'bounds':[-5, 10], 'name': 'zakharov', 'fopt': 0, 'xopt': [0, 0], 'dim': np.inf},
                        'michalewicz': { 'bounds':[0, pi], 'name': 'michalewicz', 'fopt': -1.801140718473825, 'xopt': [2.2, 1.57], 'dim': np.inf}
        }

        self.make(function_name)

    def make(self, function_name):
        self.func = getattr(self, function_name)
        self.low = self.data[function_name]['bounds'][0]
        self.high = self.data[function_name]['bounds'][1]
        self.name = self.data[function_name]['name']
        self.fopt = self.data[function_name]['fopt']
        self.xopt = np.array(self.data[function_name]['xopt'])
        self.dim = self.data[function_name]['dim']



    #  Sphere Function
    #
    #  Global minimum: f(x*)=0 at x*=(0, 0, ...)
    #  xi ∈ [-5.12, 5.12], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/spheref.html
    @staticmethod
    def f1(x):
        x = np.asarray_chkfinite(x)
        return sum( x**2 )


    # Schwefel's 2.22
    @staticmethod
    def f2(x):
        x = np.asarray_chkfinite(x)
        return np.sum(np.abs(x)) + np.prod(np.abs(x))


    # Schwefel's 1.2
    #
    # https://goker.dev/iom/benchmarks/schwefel-1.2
    @staticmethod
    def f3(x):
        x = np.asarray_chkfinite(x)
        return np.sum([np.sum(x[:i]) ** 2
                       for i in range(len(x))])


    # Schwefel's 2.21
    #
    # https://goker.dev/iom/benchmarks/schwefel-2.21
    @staticmethod
    def f4(x):
        x = np.asarray_chkfinite(x)
        return np.max(np.abs(x))


    #  Rosenbrock
    #
    #  Global Minimum: f(x*)=0 at x* = (1, 1, ...)
    #  xi ∈ [-5, 10], for all i = 1, …, d,
    #  xi ∈ [-2.048, 2.048], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/rosen.html
    @staticmethod
    def f5(x):
        x = np.asarray_chkfinite(x)
        x0 = x[:-1]
        x1 = x[1:]
        return (sum( (1 - x0) **2 )
            + 100 * sum( (x1 - x0**2) **2 ))


    # Step function
    #
    @staticmethod
    def f6(x):
        x = np.asarray_chkfinite(x)
        return np.sum( (x + 0.5)**2 )


    # Quadratic (Noise)
    #
    @staticmethod
    def f7(x):
        x = np.asarray_chkfinite(x)
        s = 0
        for i in range(len(x)):
            s += i * (x[i]**4)
        s + np.random.random_sample()
        return s


    # Schwefel 2.6
    #
    @staticmethod
    def f8(x):
        x = np.asarray_chkfinite(x)
        return np.sum( -x * np.sin(np.sqrt(np.abs(x))) )


    #  Rastrigin
    #
    #  Global Minimum: f(x*)=0 at x* = (0, 0, ...)
    #  xi ∈ [-5.12, 5.12], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/rastr.html
    @staticmethod
    def f9(x):
        x = np.asarray_chkfinite(x)
        n = len(x)
        return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))


    #  Ackley function
    #
    #  Global minimum: f(x*)=0 at x* = (0, 0, ...)
    #  xi ∈ [-32.768, 32.768], for all i = 1, …, d,
    #  although it may also be restricted to a smaller domain.
    #
    #  https://www.sfu.ca/~ssurjano/ackley.html
    # TODO: Percision problem
    # TODO: High percision data returns nagitive results
    @staticmethod
    def f10(x, a=20, b=0.2, c=2*pi):
        dim=len(x)
        return -20 * np.exp(-.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim) + 20 + np.exp(1)

        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        n = len(x)
        s1 = sum( x**2 )
        s2 = sum( cos( c * x ))
        r = -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)
        #  python decimal percision problem
        if r == 4.440892098500626e-16:
            return 0
        return r

    #  Griewank function
    #
    #  Global minimum: f(x*)=0 at x*(0,0,0)
    #  xi ∈ [-600, 600], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/griewank.html
    @staticmethod
    def f11(x, fr=4000):
        x = np.asarray_chkfinite(x)
        n = len(x)
        j = np.arange( 1., n+1 )
        s = sum( x**2 )
        p = prod( cos( x / sqrt(j) ))
        return s/fr - p + 1

    #  Generalized Penalized no.1
    #
    #  Global minimum: f(x*)=0 at x*(-1,-1)
    #  xi ∈ [-50, 50], for all i = 1, …, d.
    #
    # https://al-roomi.org/benchmarks/unconstrained/n-dimensions/172-generalized-penalized-function-no-1
    @staticmethod
    def f12(x, a=10, k=100, m=4):
        x = np.asarray_chkfinite(x)
        dim = len(x)
        def Ufun(x,a,k,m):
            y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
            return y
        return (math.pi/dim)*(10*((np.sin(math.pi*(1+(x[0]+1)/4)))**2)+\
                              np.sum((((x[1:dim-1]+1)/4)**2)*(1+10*((np.sin(math.pi*(1+(x[1:dim-1]+1)/4))))**2))+((x[dim-1]+1)/4)**2)+np.sum(Ufun(x,10,100,4));


    #  Generalized Penalized no.2
    #
    #  Global minimum: f(x*)=0 at x*(1,1)
    #  xi ∈ [-50, 50], for all i = 1, …, d.
    #
    # https://al-roomi.org/benchmarks/unconstrained/n-dimensions/172-generalized-penalized-function-no-2
    @staticmethod
    def f13(x, a=5, k=100, m=4):
        x = np.asarray_chkfinite(x)
        dim = len(x)
        def Ufun(x,a,k,m):
            y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
            return y
        return .1*((np.sin(3*math.pi*x[1]))**2+sum((x[0:dim-2]-1)**2*(1+(np.sin(3*math.pi*x[1:dim-1]))**2))+
            ((x[dim-1]-1)**2)*(1+(np.sin(2*math.pi*x[dim-1]))**2))+np.sum(Ufun(x,5,100,4))


     #  Shekel's Foxholes Function
     #  https://www.al-roomi.org/benchmarks/unconstrained/2-dimensions/7-shekel-s-foxholes-function
    @staticmethod
    def f14(x):
        x = np.asarray_chkfinite(x)
        a = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
        s = 0
        for j in range(0, 25):

            r = 1
            for i in range(0, 2):
                r += (x[i] - a[i][j])**6
            s += 1/r
        return (1/500 + s )**-1


    # Kowalik function
    @staticmethod
    def f15(x):
        x = np.asarray_chkfinite(x)
        #  a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.627, 0.0456, 0.0342, 0.323, 0.235, 0.0246])
        a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        b = np.array([4, 2, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16])

        s = 0
        for i in range(0, 11):
            s += (a[i] - ( (x[0] * (b[i]**2 + b[i] * x[1]) ) / (b[i]**2 + b[i] * x[2] + x[3])))**2

        return s


    @staticmethod
    def f16(x):
        x = np.asarray_chkfinite(x)
        x1 = x[0]
        x2 = x[1]

        return 4 * x1**2 - 2.1 * x1**4 + (1/3) * x1**6 + x1 * x2 - 4 * x2**2 + 4 * x2**4


    @staticmethod
    def f17(x):
        x = np.asarray_chkfinite(x)
        x1 = x[0]
        x2 = x[1]

        return (x2 - (5.1/(4*np.pi**2)) * x1**2 + (5/np.pi) * x1 - 6)**2 + 10 * (1 - (1/(8*np.pi))) * np.cos(x1) + 10


    @staticmethod
    def f18(x):
        x = np.asarray_chkfinite(x)
        x1 = x[0]
        x2 = x[1]

        return ( 1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2) )  * \
            ( 30 + (2 * x1 - 3 * x2 )**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2))


    @staticmethod
    def f19(x):
        L=x
        aH=[[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]];
        aH=np.asarray(aH);
        cH=[1,1.2,3,3.2];
        cH=np.asarray(cH);
        pH=[[.3689,.117,.2673],[.4699,.4387,.747],[.1091,.8732,.5547],[.03815,.5743,.8828]];
        pH=np.asarray(pH);
        o=0;
        for i in range(0,4):
         o=o-cH[i]*np.exp(-(np.sum(aH[i,:]*((L-pH[i,:])**2))))
        return o


    @staticmethod
    def f20(x):
        L = x
        aH=[[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],[3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]]
        aH=np.asarray(aH)
        cH=[1,1.2,3,3.2]
        cH=np.asarray(cH)
        pH=[[.1312,.1696,.5569,.0124,.8283,.5886],[.2329,.4135,.8307,.3736,.1004,.9991],
            [.2348,.1415,.3522,.2883,.3047,.6650],[.4047,.8828,.8732,.5743,.1091,.0381]]
        pH=np.asarray(pH)
        o=0
        for i in range(0,4):
            o=o-cH[i]*np.exp(-(np.sum(aH[i,:]*((L-pH[i,:])**2))))
        return o


    @staticmethod
    def f21(L):
        aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]];
        cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5];
        aSH=np.asarray(aSH);
        cSH=np.asarray(cSH);
        fit=0;
        for i in range(0,4):
          v=np.matrix(L-aSH[i,:])
          fit=fit-((v)*(v.T)+cSH[i])**(-1);
        o=fit.item(0);
        return o


    @staticmethod
    def f22(L):
        aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]];
        cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5];
        aSH=np.asarray(aSH);
        cSH=np.asarray(cSH);
        fit=0;
        for i in range(0,6):
          v=np.matrix(L-aSH[i,:])
          fit=fit-((v)*(v.T)+cSH[i])**(-1);
        o=fit.item(0);
        return o


    @staticmethod
    def f23(L):
        aSH=[[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]];
        cSH=[.1,.2,.2,.4,.4,.6,.3,.7,.5,.5];
        aSH=np.asarray(aSH);
        cSH=np.asarray(cSH);
        fit=0;
        for i in range(0,9):
          v=np.matrix(L-aSH[i,:])
          fit=fit-((v)*(v.T)+cSH[i])**(-1);
        o=fit.item(0);
        return o


    #  Michalewicz Function
    #
    #  Global minimum:
    #  d=2  f(x*)=-1.8013, x*=(2.20, 1.57)
    #  d=5  f(x*)=-4.687658
    #  d=10 f(x*)=-9.66015
    #
    #  xi ∈ [0, π], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/michal.html
    @staticmethod
    def michalewicz(x):
        michalewicz_m = 10 #.5  # orig 10: ^20 => underflow
        x = np.asarray_chkfinite(x)
        n = len(x)
        j = np.arange( 1., n+1 )
        return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )


    #  Levy
    #
    #  Global minimum: f(x*)=0 at x*(1,1,1)
    #  xi ∈ [-10, 10], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/levy.html
    @staticmethod
    def levy(x):
        x = np.asarray_chkfinite(x)
        n = len(x)
        z = 1 + (x - 1) / 4
        return (sin( pi * z[0] )**2
            + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
            +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))


    #  Dixon Price
    #
    #  Global minimum: f(x*)=0 at xi= 2^-{(2^i-2)/2^i}
    #  xi ∈ [-10, 10], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/dixonpr.html
    @staticmethod
    def dixonprice(x):
        x = np.asarray_chkfinite(x)
        n = len(x)
        j = np.arange( 2, n+1 )
        x2 = 2 * x**2
        return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2


    #  Powell Function
    #
    #  Global minimum: f(x*)=0 at x*=(0, 0, ...)
    #  xi ∈ [-4, 5], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/powell.html
    @staticmethod
    def powell(x):
        x = np.asarray_chkfinite(x)
        n = len(x)
        n4 = ((n + 3) // 4) * 4
        if n < n4:
            x = np.append( x, np.zeros( n4 - n ))
        x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
        f = np.empty_like( x )
        f[0] = x[0] + 10 * x[1]
        f[1] = sqrt(5) * (x[2] - x[3])
        f[2] = (x[1] - 2 * x[2]) **2
        f[3] = sqrt(10) * (x[0] - x[3]) **2
        return sum( f**2 )


    #  Zakharov Function
    #
    #  Global minimum: f(x*)=0 at x*=(0, 0, ...)
    #  xi ∈ [-5, 10], for all i = 1, …, d.
    #
    #  https://www.sfu.ca/~ssurjano/zakharov.html
    @staticmethod
    def zakharov(x):
        x = np.asarray_chkfinite(x)
        n = len(x)
        j = np.arange( 1., n+1 )
        s2 = sum( j * x ) / 2
        return sum( x**2 ) + s2**2 + s2**4


    @staticmethod
    def booth(x):
        x = np.asarray_chkfinite(x)
        x1 = x[:1]
        x2 = x[1:]
        return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2
