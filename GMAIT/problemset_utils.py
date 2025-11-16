from utils import *
import random
import cmath
import math

def rand_poly(deg=2, nranges=[-10, 10]):
    return lambda : poly.rand(deg, coeff_range=nranges[:])

def rand_int(ndigits):
    return lambda : random.randint(10 ** (ndigits - 1), 10 ** ndigits - 1)

def rand_complex(ndigits):
    return lambda : complex(random.randint(10 ** (ndigits - 1), 10 ** ndigits - 1), random.randint(10 ** (ndigits - 1), 10 ** ndigits - 1))

