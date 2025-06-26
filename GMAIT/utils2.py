import math
import random
from utils import *
import cmath
MAX_VARS = 2
letters = ['x', 'y', 'z', 'w', 't', '+']
open_paranthesis = [[" "], [" "], [" "], ["("], [" "], [" "], [" "]]
clsd_paranthesis = [[" "], [" "], [" "], [")"], [" "], [" "], [" "]]

def prevpprify(ind):
    arr = [[" "] for i in range(7)]
    arr[3] = [letters[ind]][:]
    return arr[:]


def Ndiff(function, dx=0.001):
    return lambda x : (function(x+dx)-function(x))/dx

def pNdiff(function, var, dx=0.001):
    if var == 0:
        return lambda y : Ndiff(lambda x : function([x, y]), dx=dx)
    elif var == 1:
        return lambda x : Ndiff(lambda y : function([x, y]), dx=dx)
    
def NdoubleIntRect(function, rect_region, dx=0.001, dy=0.001):
    '''
    rect_region = [[a, b], [c, d]]-> int a->b (int c->d f(x, y) dy) dx 
    '''
    s = 0
    x = rect_region[0][0]
    
    while x < rect_region[0][1]:
        y = rect_region[1][0]
        while y < rect_region[1][1]:
            s += dx * dy * function([x, y])
            y += dy
        
        x += dx
    
    return s

def NdoubleIntReg(function, bounds, dx=0.001, dy=0.001):
    '''
    bounds = [[a, b], [f(x), g(x)]]-> int a->b (int f(x)->g(x) h(x, y) dy) dx 
    '''
    s = 0
    x = bounds[0][0]
    
    while x < bounds[0][1]:
        y = bounds[1][0](x)
        while y < bounds[1][1](x):
            s += dx * dy * function([x, y])
            y += dy
        
        x += dx
    
    return s

def NSurfIntRect(function, surface, bounds, dx=0.001, dy=0.001):
    new_f = lambda x, y: function([x, y]) * math.sqrt(pNdiff(function, 0) ** 2 + pNdiff(function, 1) ** 2 + 1)
    
class multVar:
    def __init__(self, operation, array):
        #array = [(f1(x), 0), (f2(y), 1), (f3(z), 2), ... ]
        self.operation = operation
        self.array = array
    
    def __call__(self, input_array):
        array = []
        for func, inp_ind in self.array:
            if isinstance(func, multVar):
                array.append(func(input_array[:]))
            else:
                if callable(func):
                    array.append(func(input_array[inp_ind]))
                else:
                    array.append(func)
        z = self.operation(array[:])
        return z(0) if callable(z) else z
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        npprs = []
        for func, ind in self.array:
            if hasattr(func, 'npprint'):
                npprs.append(func.npprint(prev_ppr=prevpprify(ind)))
            else:
                npprs.append(poly([func]).npprint(prev_ppr=prevpprify(ind)))
        if isinstance(self.operation([]), Sum):
            ppr = [[], [], [], [], [], [], []]
            new_ppr = []
            for i in npprs[:-1]:
                new_ppr[:] = connect(i[:], prevpprify(-1))[:]
                ppr = connect(ppr[:], new_ppr[:])
            return connect(ppr, npprs[-1])
        
        elif isinstance(self.operation([]), Prod):
            ppr = [[], [], [], [], [], [], []]
            for i in npprs:
                ppr[:] = connect(ppr[:], connect(open_paranthesis[:], connect(i[:], clsd_paranthesis[:]))[:])[:]
            return ppr
    
    def __add__(self, other):
        if isinstance(other, multVar):
            return multVar(Sum, [(self, [i for i in range(MAX_VARS)]), (other, [i for i in range(MAX_VARS)])])
        else:
            raise Exception(TypeError, 'both types should be multVar when adding with multVar')
    
    def __mul__(self, other):
        if isinstance(other, multVar):
            return multVar(Prod, [(self, [i for i in range(MAX_VARS)]), (other, [i for i in range(MAX_VARS)])])
        else:
            raise Exception(TypeError, 'both types should be multVar when multiplying with multVar')
    def __neg__(self):
        return multVar(Prod, [(self, [i for i in range(MAX_VARS)]), (-1, 0)])
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        if isinstance(other, multVar):
            return multVar(Div, [(self, [i for i in range(MAX_VARS)]), (other, [i for i in range(MAX_VARS)])])
        else:
            raise Exception(TypeError, 'both types should be multVar when dividing with multVar')

