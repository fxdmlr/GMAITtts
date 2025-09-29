from utils import *
import utils_integration as uint
import random
import cmath


def rand_poly():
    return poly.rand(3)

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
    q = rand_poly_nice_roots([-20, 20], 3, all_real=random.randint(0, 1))
    return Div([p, q])

def nice_poly():
    return rand_poly_nice_roots([-100, 100], 2, all_real=random.randint(0, 1))

def rand_fraction_num():
    return rational.rand(nrange=[-100, 100]).simplify()
'''
0 -> real number
1 -> polynomial
2 -> point
3 -> function
4 -> matrix
5 -> question_method
'''




    


number_generators = [
    ["the solution to the initial value problem p(D)y = 0 where p(x) = $ and y(0) = $, y'(0) = $ evaluated at x = $", lambda objects : solve_diffeq_sym(objects[0].coeffs[:], [0, 1], objects[1:-1])(objects[-1]()), [nice_poly, rand_num, rand_num, rand_fraction_num], [1, 0, 0, 0]],
    ['the inverse laplace transform of F(s) = $ evaluated at t = $', lambda objects : sym_inv_lap_rat(objects[0].arr[0], objects[0].arr[1])(objects[1]()), [rnd_lap_rat, rand_fraction_num], [3, 0]],
    ['if $F(s) = $ then evaluate f_0(t) at t = $ (zero state response)', solve_lap_mat, [rand_lap_mat, rand_vect_lap, rand_fraction_num], [4, 4, 0]]
    
    
]




def single_number_gen():
    string, f, rand, tp = number_generators[random.randint(0, len(number_generators) - 1)]
    new_string = [[], [], [], [], [], [], []]
    k = 0
    inp_arr = []
    for j in string:
        if j != '$':
            new_string = connect(new_string[:], npprify(j)[:])
        else:
            m = rand[k]()
            inp_arr.append(m)
            nns =  m.npprint()[:] if hasattr(m, 'npprint') else npprify(str(m))[:]
            new_string = connect(new_string, nns[:])[:]
            k += 1
    
    return new_string, f(inp_arr[:])

'''
a, b = gen_rand_prob()
print(strpprint(a))
print(b)
'''
a, b = single_number_gen()
print(strpprint(a))
print(b)
