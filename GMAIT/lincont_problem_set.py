from utils import *
import utils_integration as uint
from lincontutils import *
import random
import cmath
import sfggame as sfg
import turtle


def rand_poly():
    return poly.rand(3)

def rand_poly_hdeg():

    deg = random.randint(2, 6)
    arr = [0, 0, 1]
    hurwitz_stable = arr[random.randint(0, len(arr) - 1)]
    if hurwitz_stable:
        p = rand_poly_nice_roots([-10, -1], deg, all_real=False)
        p.variable = 's'
        return p
    else:
        new_poly = rand_poly_nice_roots([-10, 10], deg, all_real=False).coeffs[:]
        p = poly([abs(i) for i in new_poly])
        while hurwitz_stable_num(p):
            new_poly = rand_poly_nice_roots([-10, 10], deg, all_real=False).coeffs[:]
            p = poly([abs(i) for i in new_poly])
        p.variable = 's'
        return p 

        
def rand_poly_max():
    p = poly.rand(2)
    p.coeffs[-1] = -abs(p.coeffs[-1])
    return p

def rand_poly_min():
    p = poly.rand(2)
    p.coeffs[-1] = abs(p.coeffs[-1])
    return p

def rand_poly_mat():
    return matrix.randpoly()

def rand_poly_3():
    return poly.rand(3)

def rand_num():
    return random.randint(0, 10)

def npprify(string):
    new_arr = [[], [], [], [], [], [], []]
    for i in string:
        new_arr = connect(new_arr[:], [[" "], [" "], [" "], [i], [" "], [" "], [" "]])
    return new_arr[:]


def rand_alg_func():
    '''
    the generated function is of the form : (p(x)sqrt(q(x))/(r(x)sqrt(s(x))))
    '''
    p, q, r, s = poly.rand(2), poly.rand(2), poly.rand(2), poly.rand(2)
    

def rand_point():
    return (random.randint(0, 10) * (-1) ** random.randint(0, 1), random.randint(0, 10) * (-1) ** random.randint(0, 1))

def rand_matrix():
    return matrix.rand(dims=[3, 3], nrange=[1, 100])

def rand_lap_mat():
    m = matrix.randpoly(dims=[3, 3], max_deg=1, coeff_range=[-10, 10])
    for k in m.array:
        for p in k:
            p.variable = 's'
    
    return m

def rand_vect_lap():
    m = matrix.randpoly(dims=[3, 1], max_deg=1, coeff_range=[-10, 10])
    for k in m.array:
        for p in k:
            p.variable = 's'
    
    return m

def lap_mat_inv(yn):
    d = yn.det()
    n = len(yn.array[:])
    new_arr = [[det(minor(yn.array[:], [i, j])) * (-1)**(i + j) for i in range(n)] for j in range(n)]
    return matrix(new_arr[:]), d


def solve_lap_mat(objects):
    mat = objects[0]
    vect = objects[1]
    i, d = lap_mat_inv(mat)
    res = (i * vect).array[0][0]
    return sym_inv_lap_rat(res, d)(objects[2]())

def rnd_lap_rat():
    p = poly.rand(2, coeff_range=[-10, 10])
    q = rand_poly_nice_roots([-20, 20], 3, all_real=False)
    p.variable = 's'
    q.variable = 's'
    return Div([p, q])

def rnd_lap_rat_2():
    p = poly.rand(random.randint(1, 3), coeff_range=[-10, 10])
    q = rand_poly_nice_roots([-20, 20], 4, all_real=False)
    p.variable = 's'
    q.variable = 's'
    return Div([p, q])

def nice_poly(sym='s'):
    p = rand_poly_nice_roots([-100, 100], 2, all_real=random.randint(0, 1))
    p.variable = sym[:]
    return p

def nice_poly_2(sym='s'):
    p = rand_poly_nice_roots([-100, 100], 3, all_real=False)
    p.variable = sym[:]
    return p

def rnd_pol(sym='k'):
    p = poly.rand(random.randint(1, 2), coeff_range=[1, 10])
    p.variable = sym[:]
    return p

def rand_fraction_num():
    q = random.randint(2, 100)
    p = random.randint(-10, 10)
    return rational([p, q]).simplify()

def rand_pol_coeff_pol():
    deg = random.randint(2, 4)
    arr = []
    for i in range(deg + 1):
       if random.randint(0, 1):
           obj = rnd_pol()
       else:
           obj = random.randint(1, 100)
       arr.append(obj)
    return poly(arr[:])

def check_pol_co_pol_hurwitz(objects):
    def sol(kp):
        k = float(kp)
        p = objects[0]
        arr = []
        for i in p.coeffs[:]:
            if callable(i):
                arr.append(i(k))
            else:
                arr.append(i)
        return hurwitz_stable_num(poly(arr[:]))
    
    return sol

def check_mat_hurwitz(objects):
    def sol(x):
        k1, k2, k3 = [float(i) for i in x.split()]
        v = objects[1] * matrix([[-k1, -k2, -k3]]) + objects[0]
        char_p = char_poly_state_eq(v)
        return hurwitz_stable_num(char_p)
    
    return sol

def check_mat_pol_hurwitz(objects):
    def sol(x):
        k1, k2, k3 = [float(i) for i in x.split()]
        p1, p2, p3 = objects[2].array[0]
        nmat = matrix([[-p1(k1), -p2(k2), -p3(k3)]])
        v = objects[1] * nmat + objects[0]
        char_p = char_poly_state_eq(v)
        return hurwitz_stable_num(char_p)
    
    return sol
def rand_mat():
    return matrix.rand()

def rand_vect():
    return matrix.rand(dims=[3, 1])

def rand_row_pol(syms=['p', 'q', 'r']):
    p1 = poly.rand(1, coeff_range=[1, 10])
    p2 = poly.rand(1, coeff_range=[1, 10])
    p3 = poly.rand(1, coeff_range=[1, 10])
    p1.variable = syms[0]
    p2.variable = syms[1]
    p3.variable = syms[2]
    return matrix([[p1, p2, p3]])
    

class sfg_graph:
    def __init__(self, arr):
        self.graph = arr[:]
    
    def __str__(self):
        tr = turtle.Turtle()
        sfg.draw_sfg(self.graph, 0, len(self.graph) - 1, tr)
        return ''

def rand_sfg():
    return sfg_graph(sfg.rand_sfg(5))
    
def sfg_game(objects):
    a, b = sfg.sfg_tf(objects[0].graph[:], 0, len(objects[0].graph[:]) - 1)
    r = (a/b).simplification()
    print(r)
    p, q = r.p1, r.p2
    return sym_inv_lap_rat(p, q)(objects[1])

def rand_row():
    return matrix([[random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)]])

def ss_tf_solve(objects):
    r = sstf(objects[0], objects[1], objects[2], objects[3])
    f = sym_inv_lap_rat(r.p1, r.p2)(objects[4])
    return f


'''
0 -> real number
1 -> polynomial
2 -> point
3 -> function
4 -> matrix
5 -> question_method
'''




    


number_generators = [
    ["the solution to the initial value problem ($)y=0 and y(0) = $, y'(0) = $ evaluated at x = $", lambda objects : solve_diffeq_sym(objects[0].coeffs[:], [0, 1], objects[1:-1])(objects[-1]()), [lambda : nice_poly(sym='D'), rand_num, rand_num, rand_fraction_num]],
    ['the inverse laplace transform of F(s) = $ evaluated at t = $', lambda objects : sym_inv_lap_rat(objects[0].arr[0], objects[0].arr[1])(objects[1]()), [rnd_lap_rat, rand_fraction_num]],
    ['the inverse laplace transform of F(s) = $ evaluated at t = $', lambda objects : sym_inv_lap_rat(objects[0].arr[0], objects[0].arr[1])(objects[1]()), [rnd_lap_rat_2, rand_fraction_num]],
    ["the solution to the initial value problem ($)y=0 and y(0) = $, y'(0) = $ , y''(0) = $ evaluated at x = $", lambda objects : solve_diffeq_sym(objects[0].coeffs[:], [0, 1], objects[1:-1])(objects[-1]()), [lambda : nice_poly_2(sym='D'), rand_num, rand_num, rand_num,  rand_fraction_num]],
    ['Is the system with the characateristic polynomial $ , hurwitz-routh stable ? (1-Yes 0-No) ', lambda objects : 1 if hurwitz_stable_num(objects[0]) else 0, [rand_poly_hdeg]],
    ['find a value for k so that p(x) = $ is hurwitz-routh stable. ', check_pol_co_pol_hurwitz, [rand_pol_coeff_pol]],
    ['find values for k1, k2, k3 so that the system defined by the state equations dX/dt = $X + $u(t) where u(t)= -<k1, k2, k3>X is hurwitz-routh stable. ',check_mat_hurwitz, [rand_mat, rand_vect]],
    ['find values for p, q, r so that the system defined by the state equations dX/dt = $X + $u(t) where u(t)= -$X is hurwitz-routh stable. ',check_mat_pol_hurwitz, [rand_mat, rand_vect, rand_row_pol]],
    ['find the impulse response for the figure $ at t = $ ', sfg_game, [rand_sfg, rand_num]],
    ['find the output for the system defined by the state equations dX/dt = $X + $u(t) where the output is given by y = $X + $ at t = $', ss_tf_solve, [rand_mat, rand_vect, rand_row, rand_num, lambda : round(random.random(), ndigits=2)]]
]




def single_number_gen():
    z = random.randint(0, len(number_generators) - 1)
    string, f, rand = number_generators[z]
    new_string = [[], [], [], [], [], [], []]
    k = 0
    inp_arr = []
    prev_ppr = []
    for j in string:
        if j != '$':
            new_string = connect(new_string[:], npprify(j)[:])
        else:
            m = rand[k]()
            inp_arr.append(m)
            #if z == 0:
            #    prev_ppr=npprify('D')
            #if z in [1, 2]:
            #    prev_ppr=npprify('s')
            
            nns =  m.npprint()[:] if hasattr(m, 'npprint') else npprify(str(m))[:]
            new_string = connect(new_string, nns[:])[:]
            k += 1
    
    return new_string, f(inp_arr[:])

a, b = single_number_gen()
print(strpprint(a))
print(b)