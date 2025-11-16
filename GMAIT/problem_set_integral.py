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
5 -> question_method
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
    p = poly.rand(random.randint(0, 3))
    q = rand_poly_nice_roots([-30, 30], random.randint(p.deg, 4), all_real = 0)#poly.rand(random.randint(p.deg, 4))
    return Div([p, q])

class question_method:
    def __init__(self, res, nppr):
        self.res = res
        self.npr = nppr
    
    def __call__(self):
        return self.res
    
    def npprint(self):
        return self.npr

def rand_int_sq():
    r, s, lb, hb = generate_integral_problem_iii_nppr(nranges=[-100, 100], boundary_ranges=[0, 3], n=4, max_deg=1, fweights=[0, 0, 0, 0, 0, 1, 0, 0, 1], wweights=[1, 0, 2, 1])
    while lb == hb:
        r, s, lb, hb = generate_integral_problem_iii_nppr(nranges=[-100, 100], boundary_ranges=[0, 3], n=4, max_deg=1, fweights=[0, 0, 0, 0, 0, 1, 0, 0, 1], wweights=[1, 0, 2, 1])
    return  question_method(r, s) 

def rand_int_trig_h():
    r, s, lb, hb = generate_integral_problem_iii_nppr(nranges=[-100, 100], boundary_ranges=[0, 3], n=4, max_deg=1, fweights=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1], wweights=[1, 0, 2, 0])
    while lb == hb:
        r, s, lb, hb = generate_integral_problem_iii_nppr(nranges=[-100, 100], boundary_ranges=[0, 3], n=4, max_deg=1, fweights=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1], wweights=[1, 0, 2, 0])
    return  question_method(r, s) 

def rand_fourier():
    f = rand_func_iii(nranges=[-10, 10], max_deg=random.randint(2, 4), n=1, fweights=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], wweights=[1, 0, 0, 0])
    return f

def fser(objects):
    return fourier_series(objects[4], objects[5])[0](objects[0]) + fourier_series(objects[4], objects[5])[1](objects[1]) + fourier_series(objects[4], objects[5])[0](objects[2]) + fourier_series(objects[4], objects[5])[1](objects[3])

def cfseries(objects):
    return sum([complex_fourier_series(objects[4], objects[7], objects[5], objects[6])(i) for i in objects[:4]])

def cfseries_poly(objects):
    f = uint.complex_fourier_series_poly(objects[4], objects[5], objects[6], objects[7])
    return sum([f(i) for i in objects[:4]])

def line_point(p1, p2):
    x = poly([0, 1])
    y = ((p2[1] - p1[1]) / (p2[0] - p1[0]) ) * x - ((p2[1] - p1[1]) / (p2[0] - p1[0]) ) * p1[0] + p1[1]
    return y

def rand_func_linear(line_num, a, b):
    points = []
    for i in range(line_num + 1):
        if len(points) == 0:
            points.append([a, random.randint(-10, 10)])
        elif len(points) == line_num:
            point = [b, random.randint(-10, 10)]
            points.append(point[:])
        
        else:
            points.append([random.randint(points[-1] + 1, b), random.randint(-10, 10)])
    
    p_array = []
    range_arr = []
    for i in range(len(points[:-1])):
        p_array.append(line_point(points[i], points[i+1]))
        range_arr.append([points[i], points[i+1]])
    
    return p_array, range_arr, points[:]

def connect_points(points):
    p_array = []
    range_arr = []
    for i in range(len(points[:-1])):
        p_array.append(line_point(points[i], points[i+1]))
        range_arr.append([points[i][0], points[i+1][0]])
    
    return p_array[:], range_arr[:]

def fourier_linear_cm_s(objects):
    points = [[objects[4], objects[5]], [objects[6], objects[7]], [objects[8], objects[9]], [objects[10], objects[11]]]
    p, r = connect_points(points[:])
    t = points[-1][0] - points[0][0]
    f = uint.mult_cfs(p[:], r[:], t)
    return f(objects[0]) + f(objects[1]) + f(objects[2]) + f(objects[2])

def fourier_linear_cm_t(objects):
    points = [[objects[0], objects[1]], [objects[2], objects[3]], [objects[4], objects[5]], [objects[6], objects[7]]]
    p, r = connect_points(points[:])
    f = uint.mult_cft(p[:], r[:])
    
    return f(objects[8])

def fourier_linear_re_s(objects):
    points = [[objects[4], objects[5]], [objects[6], objects[7]], [objects[8], objects[9]], [objects[10], objects[11]]]
    p, r = connect_points(points[:])
    t = points[-1][0] - points[0][0]
    f = uint.mult_rfs(p[:], r[:], t)
    return f(objects[0])[0] + f(objects[1])[1] + f(objects[2])[0] + f(objects[2])[1]

def gen_int_type1():
    n = random.randint(2, 5)
    p = poly([0, 1]) ** n
    q = Prod([poly([random.randint(1, 10)**2, 0, 1]) for k in range(n)])
    return Div([p, q])

def rand_euler():
    p = polymvar.rand_2_var()
    q = polymvar.rand_2_var()
    r = poly.rand(2, coeff_range=[-10, 10])
    while r.coeffs[1]**2 == 4 * r.coeffs[0] * r.coeffs[2]:
        r = poly.rand(2, coeff_range=[-10, 10])
    f = Comp([r, sqrt()])
    a = 0
    for i in range(len(p.array[:])):
        for j in range(len(p.array[i])):
            if p.array[i][j][0] != 0:
                alpha = poly([0, 1])**i
                beta = f ** j if j != 0 else 1
                if j % 2 == 0:
                    beta = r ** int(j/2)
                a += Prod([p.array[i][j][0], alpha , beta])
    
    b = 0
    for i in range(len(q.array[:])):
        for j in range(len(q.array[i])):
            if q.array[i][j][0] != 0:
                alpha = poly([0, 1])**i
                beta = f ** j if j != 0 else 1
                if j % 2 == 0:
                    beta = r ** int(j/2)
                b += Prod([q.array[i][j][0], alpha , beta])
    
    return question([p, q, r], Div([a, b]).npprint())

def generate_quad_n():
    p = poly.rand(random.randint(0, 4), coeff_range=[-10, 10])
    z = complex(random.randint(-10, 10), random.randint(-10, 10))
    r = random.randint(1, 10) * poly([round(abs(z)**2), -2*z.real, 1])
    n = random.randint(3, 11)
    while n % 2 == 0:
        n = random.randint(3, 7)
    
    return question([p, r, n], Div([p, Comp([Comp([r, sqrt()]), poly([0, 1])**n])]).npprint())

def rand_trig_rat():
    p = polymvar.rand_2_var(d=3, nrange=[1, 10], terms=4)
    q = polymvar.rand_2_var(d=3, nrange=[1, 10], terms=4)

    a = 0
    for i in range(len(p.array[:])):
        for j in range(len(p.array[i])):
            if p.array[i][j][0] != 0:
                alpha = Comp([sin(), poly([0, 1])**i]) if i != 0 else 1
                beta = Comp([cos(), poly([0, 1])**j]) if j != 0 else 1
                a += Prod([p.array[i][j][0], alpha , beta])
    
    b = 0
    for i in range(len(q.array[:])):
        for j in range(len(q.array[i])):
            if q.array[i][j][0] != 0:
                alpha = Comp([sin(), poly([0, 1])**i]) if i != 0 else 1
                beta = Comp([cos(), poly([0, 1])**j]) if j != 0 else 1
                b += Prod([q.array[i][j][0], alpha , beta])
    
    return question([p, q], Div([a, b]).npprint())
    

def int_type1(objects):
    p = 1
    for i in objects[0].arr[1].arr:
        p *= i

    f = uint.ratexp_integration_real(objects[0].arr[0], p)
    return f(objects[2]) - f(objects[1])

class question:
    def __init__(self, arg_arr, nppr):
        self.arr = arg_arr[:]
        self.nppr = nppr[:]
    
    def npprint(self, prev_pprint=[]):
        return self.nppr[:]

def int_euler(objects):
    p, q, r = objects[0].arr
    f = uint.integrate_euler(p, q, r)
    return f(objects[2]) - f(objects[1])

def int_quad_p(objects):
    p, r, n = objects[0].arr 
    f = uint.integrate_q_p_n(p, r, n)
    return f(objects[2]) - f(objects[1])

def int_trig_rat(objects):
    p, q = objects[0].arr 
    f = uint.integrate_trig_rat(p, q)
    return f(objects[2]) - f(objects[1])

arr_cmplx =  [rand_num, rand_num, rand_num, rand_num, rand_fourier, lambda : random.randint(0, 3), lambda : random.randint(4, 7), lambda : random.randint(7, 15)]
arr_cmplx_2 =  [rand_num, rand_num, rand_num, rand_num, rand_poly, lambda : random.randint(0, 3), lambda : random.randint(4, 7), lambda : random.randint(7, 15)]
arr_cfls =  [rand_num, rand_num, rand_num, rand_num, lambda : random.randint(0, 3), lambda : random.randint(-10, 10), lambda : random.randint(4, 7), lambda : random.randint(-10, 10), lambda : random.randint(8, 11), lambda : random.randint(-10, 10), lambda : random.randint(12, 15), lambda : random.randint(-10, 10)]
arr_cflt =  [lambda : random.randint(0, 3), lambda : random.randint(-10, 10), lambda : random.randint(4, 7), lambda : random.randint(-10, 10), lambda : random.randint(8, 11), lambda : random.randint(-10, 10), lambda : random.randint(12, 15), lambda : random.randint(-10, 10), rand_num]


number_generators = [
    
    ['the integral of f(x) = $ from $ to $', lambda objects : uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[2]) - uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[1]), [rand_rat_expr, rand_num, rand_num], [5, 0, 0]],
    ['the integral of f(x) = ($)exp($x) from $ to $ ', lambda objects : solve_poly_exp(objects[0], objects[1], objects[2], objects[3]), [rand_poly, rand_num, rand_num, rand_num], [1, 0, 0, 0]],
    ['the integral of f(x) = ($)cos($x) from $ to $ ', lambda objects : (solve_poly_trig(objects[0], objects[1], objects[2], objects[3], t=1)).real, [rand_poly, rand_num, rand_num, rand_num], [1, 0, 0, 0]],
    ['the integral of f(x) = ($)sin($x) from $ to $ ', lambda objects : (solve_poly_trig(objects[0], objects[1], objects[2], objects[3])).real, [rand_poly, rand_num, rand_num, rand_num], [1, 0, 0, 0]],
    ['the result of $', lambda objects : objects[0].res, [rand_int_sq], [5]],
    ['the result of $', lambda objects : objects[0].res, [rand_int_trig_h], [5]],
    ['the integral of f(x) = $ from $ to $ ', int_type1, [gen_int_type1, lambda : random.randint(0, 3), lambda : random.randint(4, 7)], [3, 0, 0]],
    ['the integral of f(x) = $ from $ to $ ', int_type1, [gen_int_type1, lambda : random.randint(0, 3), lambda : random.randint(4, 7)], [3, 0, 0]],
    ['the integral of f(x) = $ from $ to $ ', int_euler, [rand_euler, lambda : random.randint(0, 2), lambda : random.randint(3, 10)], [7, 0, 0]],
    ['the integral of f(x) = $ from $ to $ ', int_quad_p, [generate_quad_n, lambda : random.randint(0, 2), lambda : random.randint(3, 10)], [7, 0, 0]],
    ['the integral of f(x) = $ from $ to $ ', int_trig_rat, [rand_trig_rat, lambda : random.randint(0, 2), lambda : random.randint(3, 10)], [7, 0, 0]],
    ['the integral of f(x) = $ from $ to $', lambda objects : uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[2]) - uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[1]), [rand_rat_expr, rand_num, rand_num], [5, 0, 0]],
    ['the integral of f(x) = $ from $ to $', lambda objects : uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[2]) - uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[1]), [rand_rat_expr, rand_num, rand_num], [5, 0, 0]],
    ['the integral of f(x) = $ from $ to $', lambda objects : uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[2]) - uint.ratexp_integration_real(objects[0].arr[0], objects[0].arr[1])(objects[1]), [rand_rat_expr, rand_num, rand_num], [5, 0, 0]],


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

a, b = single_number_gen()
print(strpprint(a))
print(b)
'''
