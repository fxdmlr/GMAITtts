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

def rand_line():
    return poly.rand(1)

def rand_poly_3():
    return poly.rand(3)

def rand_num():
    return random.randint(0, 10)

def npprify(string):
    new_arr = [[], [], [], [], [], [], []]
    for i in string:
        new_arr = connect(new_arr[:], [[" "], [" "], [" "], [i], [" "], [" "], [" "]])
    return new_arr[:]

def lap_gen():
    a = random.randint(0, 1)
    p = poly.rand(a)
    q = poly.rand(random.randint(a, 2))
    p.variable = 's'
    q.variable = 's'
    return Div([p, q])

def rand_alg_func():
    '''
    the generated function is of the form : (p(x)sqrt(q(x))/(r(x)sqrt(s(x))))
    '''
    p, q, r, s = poly.rand(2), poly.rand(2), poly.rand(2), poly.rand(2)
    

def rand_point():
    return (random.randint(0, 10) * (-1) ** random.randint(0, 1), random.randint(0, 10) * (-1) ** random.randint(0, 1))

def rand_matrix():
    return matrix.rand(dims=[3, 3], nrange=[1, 100])

'''
0 -> real number
1 -> polynomial
2 -> point
3 -> function
4 -> matrix

'''

def distance(point, function):
    x = poly([0, 1])
    f = ((x - point[0])**2 + (function - point[1])**2).diff()
    p = f.roots()[0]
    np = function(p)
    return math.sqrt(abs((point[0] - p) ** 2 + (point[1] - np)**2))

def critical_point_poly(p):
    pol = p
    for i in range(p.deg - 1):
        pol = pol.diff()
    
    x = -pol.coeffs[0] / pol.coeffs[1]
    y = p(x)
    return [x, y]

def maximum_poly(p):
    pol = p.diff()
    xarr = pol.roots()
    yarr = [p(x) for x in xarr]
    sarr = [pol.diff()(x) < 0 for x in xarr]
    z_arr = []
    min_item = None
    for i in range(len(xarr)):
        if sarr[i]:
            if min_item is None:
                min_item = (xarr[i], yarr[i])
            else:
                if xarr[i] < min_item[0]:
                    min_item = (xarr[i], yarr[i])
    
    return min_item

def minimum_poly(p):
    pol = p.diff()
    xarr = pol.roots()
    yarr = [p(x) for x in xarr]
    sarr = [pol.diff()(x) > 0 for x in xarr]
    z_arr = []
    min_item = None
    for i in range(len(xarr)):
        if sarr[i]:
            if min_item is None:
                min_item = (xarr[i], yarr[i])
            else:
                if xarr[i] < min_item[0]:
                    min_item = (xarr[i], yarr[i])
    
    return min_item
    
def bisector(p1, p2):
    nx = (p1[0] + p2[0])/2
    ny = (p1[1] + p2[1])/2
    return poly([(p2[0] - p1[0])*nx/(p2[1] - p1[1])+ny, -(p2[0] - p1[0])/(p2[1] - p1[1])]) 

def center_circle_tg_line(point, line, x_tang):
    y = line(x_tang)
    pl = bisector(point, [x_tang, y])
    x = poly([0, 1])
    z = (pl - line) ** 2 * (1 / (1 + line.diff()(0)**2)) - (x - point[0])**2 - (pl - point[1])**2
    nx = z.roots()[0]
    return [nx, pl(nx)]    

def solve_poly_trig(p, n, a, b, t=0):
    r, s = uint.solve_poly_trig(p, n, t=t)
    f = lambda x : r(x) * cmath.exp(x * n * complex(0, 1)) + s(x) * cmath.exp(-x * n * complex(0, 1))
    return f(b) - f(a)

def solve_poly_exp(p, n, a, b, t=0):
    r = uint.solve_poly_exp(p, n)
    f = lambda x : r(x) * cmath.exp(x * n)
    return f(b) - f(a)

def rand_rat_expr():
    p = poly.rand(random.randint(0, 2))
    q = poly.rand(random.randint(p.deg, 4))
    return Div([p, q])
    

number_generators = [
    ['sum of the roots of $ ', lambda objects : sum(objects[0].roots()), [rand_poly], [1]],
    ['sum of the squares of the roots of $ ', lambda objects : sum([i**2 for i in objects[0].roots()]), [rand_poly], [1]],
    ['$ evaluated at x = $ ', lambda objects: objects[0](objects[1]), [rand_poly, lambda : random.randint(0, 10)], [3, 0]],
    ['intersection of $ and $ ', lambda objects : (objects[0] - objects[1]).roots()[0], [rand_poly, rand_poly], [1, 1]],
    ['distance between $ and $ ', lambda objects : abs(complex(objects[0][0], objects[0][1]) - complex(objects[1][0], objects[1][1])), [rand_point, rand_point], [2, 2]],
    ['distance between $ and $ ', lambda objects : distance(objects[0], objects[1]), [rand_point, rand_line], [2, 1]],
    ['det $', lambda objects : objects[0].det(), [rand_matrix], [4]],
    ['the integral of f(x) = $ from $ to $', lambda objects : uint.integrate_ratexp(objects[0].arr[0], objects[0].arr[1])(objects[2]) - uint.integrate_ratexp(objects[0].arr[0], objects[0].arr[1])(objects[1]), [rand_rat_expr, rand_num, rand_num], [5, 0, 0]],
    ['the integral of f(x) = ($)exp($x) from $ to $ ', lambda objects : solve_poly_exp(objects[0], objects[1], objects[2], objects[3]), [rand_poly, rand_num, rand_num, rand_num], [1, 0, 0, 0]],
    ['the integral of f(x) = ($)cos($x) from $ to $ ', lambda objects : (solve_poly_trig(objects[0], objects[1], objects[2], objects[3], t=1)).real, [rand_poly, rand_num, rand_num, rand_num], [1, 0, 0, 0]],
    ['the integral of f(x) = ($)sin($x) from $ to $ ', lambda objects : (solve_poly_trig(objects[0], objects[1], objects[2], objects[3])).real, [rand_poly, rand_num, rand_num, rand_num], [1, 0, 0, 0]],
    ['the coefficient of cos($wx) in the fourier series of $ with a period of $ ', lambda objects : uint.real_fourier_series_poly(objects[1], objects[2]/2)[0](objects[0]), [rand_num, rand_poly, rand_num], [0, 1, 0]],
    ['the coefficient of sin($wx) in the fourier series of $ with a period of $ ', lambda objects : uint.real_fourier_series_poly(objects[1], objects[2]/2)[1](objects[0]), [rand_num, rand_poly, rand_num], [0, 1, 0]],
    

    
    
]

function_generators = [
    ['d($)/dx ', lambda objects : objects[0].diff(), [rand_poly], [1]],
    ['the line tangent to $ at x = $', lambda objects : poly([objects[0](objects[1]) - objects[1] * objects[0].diff()(objects[1]), objects[0].diff()(objects[1])]), [rand_poly, lambda : random.randint(0, 10)], [1, 0]],
    ['the inverse laplace transfrom of $', lambda objects : sym_inv_lap_rat(objects[0].arr[0], objects[0].arr[1]), [lap_gen], [3]],
    ["the solution to the initial value problem p(D)y = 0 where p(x) = $ and y(0) = $, y'(0) = $ ", lambda objects : solve_diffeq_sym(objects[0].coeffs[:], [0, 1], objects[1:]), [rand_poly_min, rand_num, rand_num], [1, 0, 0]],
    ['the integral of f(x) = ($)/($) ', lambda objects : uint.integrate_ratexp(objects[0], objects[1]), [rand_poly, rand_poly], [1, 1]]
    
]

poly_generators = [
    ['d($)/dx ', lambda objects : objects[0].diff(), [rand_poly], [1]],
    ['the line tangent to $ at x = $', lambda objects : poly([objects[0](objects[1]) - objects[1] * objects[0].diff()(objects[1]), objects[0].diff()(objects[1])]), [rand_poly, lambda : random.randint(0, 10)], [1, 0]],
    ['the bisector of the line connecting $ and $', lambda objects : bisector(objects[0], objects[1]), [rand_point, rand_point], [2, 2]],
    ['det $', lambda objects: objects[0].det(), [rand_poly_mat], [4]],
    ['characteristic polynomial of $', lambda objects: objects[0].charpoly(), [rand_matrix], [4]]
    
]

point_generators = [
    ['the maximum of $', lambda objects : maximum_poly(objects[0]), [rand_poly_max], [1]],
    ['the minimum of $', lambda objects : minimum_poly(objects[0]), [rand_poly_min], [1]],
    ['center of the circle passing through $ and tangent to $ at x = $', lambda objects : center_circle_tg_line(objects[0], objects[1], objects[2]), [rand_point, rand_line, rand_num], [2, 1, 0]]
]

matrix_generators = [
    ['the inverse matrix of $', lambda objects: objects[0].inverse(), [rand_matrix], [4]],
    ['the result of $$', lambda objects : objects[0]*objects[1], [rand_matrix, rand_matrix], [4]],
    ['the result of f(A) where f(x) is  $ and A = $', lambda objects: objects[0](objects[1]), [rand_poly, rand_matrix], [1, 4]]
]

def create_problem(base_prob):
    string, func, rand_method, inp_types = base_prob[:]
    mod_inp_arr = []
    k = 0
    new_string = [[], [], [], [], [], [], []]
    for j in string:
        if j != '$':
            new_string = connect(new_string[:], npprify(j)[:])
        else:
            m = inp_types[k]
            if m == 0:
                ns, nf, ni, nr = number_generators[random.randint(0, len(number_generators) - 1)]
            elif m == 1:
                ns, nf, ni, nr = poly_generators[random.randint(0, len(poly_generators) - 1)]
            
            elif m == 2:
                ns, nf, ni, nr = point_generators[random.randint(0, len(point_generators) - 1)]
            
            elif m == 3:
                s = random.randint(0, 1)
                if s == 0:
                    ns, nf, ni, nr = poly_generators[random.randint(0, len(poly_generators) - 1)]
                else:
                    ns, nf, ni, nr = function_generators[random.randint(0, len(function_generators) - 1)]
            
            elif m == 4:
                ns, nf, ni, nr = matrix_generators[random.randint(0, len(matrix_generators) - 1)]
            
            
            k += 1
            inp_arr = [i() for i in ni]
            mod_inp_arr.append(nf(inp_arr[:]))
            t = 0
            nns = [[], [], [], [], [], [], []]
            for l in ns:
                if l != '$':
                    nns = connect(nns[:], npprify(l))[:]
                else:
                    nns = connect(nns[:], inp_arr[t].npprint() if hasattr(inp_arr[t], 'npprint') else npprify(str(inp_arr[t])))[:]
                    t += 1
            
            new_string = connect(new_string, nns[:])[:]
    
    return new_string, func(mod_inp_arr[:])

def gen_rand_prob():
    base = number_generators[random.randint(0, len(number_generators) - 1)]
    return create_problem(base)

def fragmented_problems(n=2):
    base = number_generators[random.randint(0, len(number_generators) - 1)]
    string, func, rand_method, inp_types = base[:]
    mod_inp_arr = []
    k = 0
    new_string = [[], [], [], [], [], [], []]
    for j in string:
        if j != '$':
            new_string = connect(new_string[:], npprify(j)[:])
        else:
            m = inp_types[k]
            if m == 0:
                b = number_generators[random.randint(0, len(number_generators) - 1)]
                string, result = create_problem(b)
            elif m == 1:
                b = poly_generators[random.randint(0, len(poly_generators) - 1)]
                string, result = create_problem(b)
            
            elif m == 2:
                b = point_generators[random.randint(0, len(point_generators) - 1)]
                string, result = create_problem(b)
            
            elif m == 3:
                s = random.randint(0, 1)
                if s == 0:
                    b = poly_generators[random.randint(0, len(poly_generators) - 1)]
                    string, result = create_problem(b)
                else:
                    b = function_generators[random.randint(0, len(function_generators) - 1)]
                    string, result = create_problem(b)
            
            elif m == 4:
                b = matrix_generators[random.randint(0, len(matrix_generators) - 1)]
                string, result = create_problem(b)
            
            
            k += 1
            mod_inp_arr.append(result)
            new_string = connect(new_string[:], string[:])[:]
    
    return new_string, func(mod_inp_arr[:])

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

a, b = single_number_gen()
print(strpprint(a))
print(b)
'''