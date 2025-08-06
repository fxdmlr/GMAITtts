'''
A pde is stored as a two dimensoinal array EQ
where EQ[i][j] is D^i_x D^j_y u(x, y) e.g.
if EQ[1][2] = 3 then there exists a 3u_xyy term in
our equation.

The L.H.S is always supposed to be 0.

The solution is derived via multiple laplace
transforms with respect to each variable.

'''


from utils import *
import math
import random
import cmath

zero_pol = [[[0 for j in range(2)] for i in range(2)]for k in range(2)]

def conv_to_lap(eq):
    new_arr = [[[0 for k in range(len(eq))]for j in i]for i in eq]
    for i in range(len(eq)):
        for j in range(len(eq[i])):
            new_arr[i][j][0] = eq[i][j]
    return polymvar(new_arr)

def y_roots(p):
    pass