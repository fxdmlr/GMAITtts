from utils import *
import math


def add_ratexp(r1, r2):
    return [(r1[0] * r2[1] + r2[0]*r1[1]) , (r1[1] * r2[1])]

def lagrange_interpolation(inputs, outputs):
    def lj(j, inputs):
        p = 1
        q = 1
        for i in range(len(inputs)):
            if i != j:
                p *= poly([-inputs[i], 1])
                q *= inputs[j] - inputs[i]
        return p * (1 / q)
    
    s = 0
    for i in range(len(inputs)):
        s += outputs[i] * lj(i, inputs[:])
    
    return s


def simplify_ratexp(p, q):
    arr = []
    min_deg = min(p.deg, q.deg)
    for i in range(min_deg + 1):
        print("p(i) = ", p(i))
        print("q(i) = ", q(i))
        print("gcd = ", math.gcd(p(i), q(i)))
        arr.append(math.gcd(p(i), q(i)))
    pol = lagrange_interpolation([i for i in range(min_deg + 1)], arr[:])
    print(pol)
    print(arr)
    return p / pol, q / pol

def simplify_ratexp2(p, q):
    n = p.deg + q.deg + 2
    array = []
    lhs = []
    for i in range(n):
        sub = [q(i) * (i**j) for j in range(p.deg + 1)]
        sub += [-p(i) * (i**j) for j in range(q.deg + 1)]
        array.append(sub[:])
        lhs.append([p(i) / q(i)])
    
    res = (matrix(array).inverse() * matrix(lhs)).transpose().array[0]
    np = poly(res[:p.deg + 2])
    nq = poly(res[p.deg+2:])
    
    return np, nq

def add_rat_poly(p, rat):
    # finds p(q/r) where rat = [q, r]
    arr = [0, 1]
    q, r = rat[:]
    for i in range(len(p.coeffs)):
        p1, q1 = p.coeffs[i] * (q**i), r**i
        
        arr = add_ratexp(arr[:], [p1, q1])
    
    return arr

def integrate_ratexp(np, nq):
    if np.deg >= nq.deg:
        p = np % nq
        q = nq
        z = (np / nq).integrate()
        
    else:
        p = np
        q = nq
        z = 0
        
    coeffs = partial_frac_decomp(p, q)
    roots = []
    co = []
    x = poly([0, 1])
    for c, r_m in coeffs:
        r= complex(round(r_m.real, ndigits=5), round(r_m.imag, ndigits=5))
        if r not in roots and r.conjugate() not in roots:
            roots.append(r)
            co.append([c])
        
        else:
            if r in roots:
                ind = roots.index(r)
            else:
                ind = roots.index(r.conjugate())
            co[ind].append(c)
    

    rat_arr = []
    for i in range(len(roots)):
        
        if len(co[i]) == 1:
            rat_arr.append([co[i][0][0], x - roots[i]])
        else:
            sub_rat = []
            for j in range(len(co[i][0])):
                arr = add_ratexp([co[i][0][j], poly([- roots[i], 1]) ** (j + 1)], [co[i][1][j], poly([- roots[i].conjugate(), 1]) ** (j + 1)])
                sub_rat.append(arr[:])
            
            rat_arr += sub_rat[:]
    
    mod_rat = []
    for a, b in rat_arr:
        if hasattr(a, 'coeffs'):
            new_a = poly([round(i.real, ndigits=5) for i in a.coeffs[:]])
        else:
            new_a = poly([round(a.real, ndigits=5)])
        
        if hasattr(b, 'coeffs'):
            new_b = poly([round(i.real, ndigits=5) for i in b.coeffs[:]])
        else:
            new_b = poly([round(b.real, ndigits=5)])
        
        mod_rat.append([new_a, new_b])
    
    s = 0
    for u, v in mod_rat:
        
        if v.deg == 1:
            s += Prod([u, Comp([v, log()])])
        else:
            sdiff = v.diff()
            y1, y2 = u / sdiff, u % sdiff
            s += Prod([y1, Comp([v, log()])])
            w1 = v.coeffs[-1]
            y2 = y2 * (1/w1)
            w2 = poly([i / w1 for i in v.coeffs[:]])
            a, b = poly([w2.coeffs[1] / 2, 1]), w2.coeffs[0] - (w2.coeffs[1] / 2)**2
            s += Prod([y2* (1 / math.sqrt(b)), Comp([a * (1/math.sqrt(b)), atan()])])
    
    return s + z

def transform_rat(p, q, bounds, n = 2):

    functions = [sin(), cos(), tan(), atan(), asin(), sqrt()]
    inv_funcs = [asin(), Sum([Prod([-1, asin()]), math.pi / 2]), atan(), tan(), sin(), poly([0, 0, 1])]
    init_p, init_q = poly(p.coeffs[:]), poly(q.coeffs[:])
    for i in range(n):
        ind = random.randint(0, len(functions) - 1)
        fun = functions[ind]

        init_p = Prod([fun.diff(), Comp([fun, init_p])])
        init_q = Comp([fun, init_q])
        bounds[0] = inv_funcs[ind](bounds[0])
        bounds[1] = inv_funcs[ind](bounds[1])
    
    return init_p, init_q, bounds[:]

def adjust_bounds(p1, q1, bounds):
    lower_roots = (p1 - bounds[0] * q1).roots()
    for r in lower_roots:
        if r.imag < 10**(-5) and abs(q1(r)) > 10 ** (-5):
            lb = r
            break
    else:
        return None
    higher_roots = (p1 - bounds[1] * q1).roots()
    for r in higher_roots:
        if r.imag < 10**(-5) and abs(q1(r)) > 10 ** (-5):
            hb = r
            break
    else:
        return None
    
    return [lb, hb]

def simplify_rat(p, q):
    a1 = math.gcd(*p.coeffs[:])
    a2 = math.gcd(*q.coeffs[:])
    c = math.gcd(a1, a2)
    return p * (1/c), q * (1/c) 

def transform_rat_p(p, q, bounds, n = 1):

    init_p, init_q = poly(p.coeffs[:]), poly(q.coeffs[:])
    for i in range(n):

        p1 = poly.rand(1, coeff_range=[1, 10])
        q1 = poly.rand(1, coeff_range=[1, 10])
        d = p1.diff()*q1 - p1*q1.diff()
        dp = q1**2
        
        arr = adjust_bounds(p1, q1, bounds[:])
        while arr is None:
            p1 = poly.rand(1, coeff_range=[1, 10])
            q1 = poly.rand(1, coeff_range=[1, 10])
            d = p1.diff()*q1 - p1*q1.diff()
            dp = q1**2
            arr = adjust_bounds(p1, q1, bounds[:])
        print('p1 = ', p1)
        print('q1 = ', q1)
        
        bounds[:] = arr[:]
        
        p11, q11 = add_rat_poly(init_p, [p1, q1])
        p12, q12 = add_rat_poly(init_q, [p1, q1])
        
        new_p = p11 * q12 * d
        new_q = q11 * p12 * dp
        
        print('p1 = ', new_p)
        print('q1 = ', new_q)
        
        init_p, init_q = simplify_rat(new_p, new_q)
        
        
    
    return init_p, init_q, bounds[:]

def solve_mono_exp(n, a):
    # finds the integral of x^n e^{ax}
    # if the result is p(x)e^{ax} returns p(x)
    if a == 0:
        x = poly([0, 1]) ** n
        return x.integrate()
    s = poly([0, 1])**n
    np = 0
    i = 0
    while s.deg != 0:
        np += s * (1 / a**(i+1)) * (-1)**i
        s = s.diff()
        i += 1
    np += s * (1/a**(i+1)) * (-1)**i
    return np


def solve_poly_exp(p, a):
    # finds the integral of p(x)e^{ax}
    
    s = 0
    for i in range(len(p.coeffs[:])):
        
        s += p.coeffs[i] * solve_mono_exp(i, a)
    
    return s

def solve_poly_trig(p, a, t=0):
    # if t = 0 then solves the integral of p(x) sin ax otherwise cos
    if t == 0:
        res = [solve_poly_exp(p, a*complex(0, 1)) * (1 / complex(0, 2)), solve_poly_exp(p, a*complex(0, -1)) * (-1 / complex(0, 2))]
    
    else:
        res = [solve_poly_exp(p, a * complex(0, 1)) * (1 / 2), solve_poly_exp(p, a*complex(0, -1)) * (1 / 2)]
    
    return res

def real_fourier_series_poly(p, l):
    def a_n (n):
        if n == 0:
            f = p.integrate()
            return (f(l) - f(-l)) / (2*l)
        f1, f2 = solve_poly_trig(p, n * math.pi / l, t = 1)
        func = lambda x : f1(x) * cmath.exp(n*cmath.pi*complex(0, 1)*x/l ) + f2(x) * cmath.exp(-n*cmath.pi*complex(0, 1)*x/l) 
        return (func(l) - func(-l)) / l
    
    def b_n (n):
        if n == 0: return 0
        f1, f2 = solve_poly_trig(p, n * math.pi / l)
        func = lambda x : f1(x) * cmath.exp(n*cmath.pi*complex(0, 1)*x/l ) + f2(x) * cmath.exp(-n*cmath.pi*complex(0, 1)*x/l)
        return (func(l) - func(-l)) / l
    
    return a_n, b_n

def complex_fourier_series_poly(p, a, b, T):
    res = lambda n : solve_poly_exp(p, complex(0, -n*2*math.pi/T)) / T
    return res

def complex_fourier_transform(p, a, b):
    #finds the fourier transform of f = p(x) a<x<b 0 O.W.
    def f(w):
        g = solve_poly_exp(p, complex(0, -w))
        return g(b)*cmath.exp(complex(0, -w*b)) - g(a)*cmath.exp(complex(0, -w*a))
    return f
    
def inverse_fourier_cmplx(p, a, b):
    #finds the fourier transform of f = p(x) a<x<b 0 O.W.
    def f(w):
        g = solve_poly_exp(p, complex(0, w))
        return (g(b)*cmath.exp(complex(0, w*b)) - g(a)*cmath.exp(complex(0, w*a))) / (2*math.pi)
    return f
