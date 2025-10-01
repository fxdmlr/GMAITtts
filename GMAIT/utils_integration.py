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

    print(coeffs[:])

    s = []
    for c, r in coeffs:
        pol = poly([-r, 1])
        for k in range(len(c)):
            if c[k] != 0:
                if k == 0:
                    s.append(c[k] * Comp([pol, log()]))
                else:
                    s.append(Div([-c[k], k * pol ** (k)]))
    return Sum(s[:])


def special_case_sq_pretty(a, n):
    # solves the integral of 1 / (x^2+a)^n
    p = Comp([poly([a, 0]), sqrt()])
    if n == 1:
        return Prod([Div([1, p]), Comp([Div([poly([0, 1]), p]), atan()])])

    m = Prod([Div([1, 2*(n-1)*a]), Div([poly([0, 1]), poly([a, 0, 1]) ** (n-1)])])
    return Sum([m, Prod([Div([2*n-3, (2*n-2)*a]), special_case_sq(a, n - 1)])])
def special_case_sq(a, n):
    # solves the integral of 1 / (x^2+a)^n

    p = cmath.sqrt(a)
    if n == 1:
        return Comp([Div([poly([0, 1]), p]), atan()]) * (1/p)

    m = (1 / (2*(n-1)*a)) * Div([poly([0, 1]), poly([a, 0, 1]) ** (n-1)])


    return m + ((2*n-3) / ((2*n-2)*a) ) * special_case_sq(a, n - 1)

def integrate_cos_n(n):
    #integrates cos^n x
    if n == 0:
        return poly([0, 1])
    elif n == 1:
        return sin()
    else:
        m = Prod([Comp([cos(), poly([0, 1]) ** (n-1)]), sin(), 1/n])
        return Sum([m, Prod([(n - 1 )/n, integrate_cos_n(n - 2)])])

def integrate_sin_n_cos_m(n, m):
    #integrates sin^n x cos ^m x
    if n % 2 :
        p = -(poly([1, 0, -1])**int((n-1)/2) * poly([0, 1])**m).integrate()
        return Comp([cos(), p])
    else:
        p = poly([1, 0, -1])**int(n/2) * poly([0, 1])**m
        s = []
        for i in range(len(p.coeffs)):
            s.append(p.coeffs[i] * integrate_cos_n(i))

        return Sum(s[:])

def special_case_sq_2(m, a, n):
    # solves the integral of x^m / (x^2+a)^n
    # let sqrt(a)tan u = x then the integral is transformed to
    # a^(m/2)tan(u)^m a^(1/2 - n) cos(u)^(2n - 2) = a^(m/2-n+1/2) sin(u)^m cos(u)^(2n - m - 2)
    if m == 0:
        return special_case_sq(a, n)
    f = integrate_sin_n_cos_m(m, 2*n-m-2) * a ** (m/2 - n +0.5)
    return Comp([Comp([poly([0, 1/cmath.sqrt(a)]), atan()]), f])

def integrate_m_p_n(m, p, n):
    # integrates x^m / (p(x)^n)
    # where p(x) = x^2+bx+c where b^2 < 4*c
    a = (p - (p.diff() * 0.5) ** 2)(0)
    ns = poly([-p.diff()(0) * 0.5, 1])**m
    s = []
    for i in range(len(ns.coeffs)):
        f = ns.coeffs[i] * special_case_sq_2(i, a, n)
        s.append(Comp([p.diff()*0.5, f]))

    return Sum(s[:])

def integrate_q_p_n(q, p, n):
    #integrates q(x) / (p(x)^n) where p(x) is the same quadratic as above
    return Sum([q.coeffs[i] * integrate_m_p_n(i, p, n) for i in range(len(q.coeffs))])

def integrate_quad_n(p, m):
    #integrates 1/(p(x)^m) where p(x)=ax^2+bx+c
    n = m - 1/2
    a, b, c = p.coeffs[2], p.coeffs[1], p.coeffs[0]
    if m == 1:
        if b ** 2 - 4*a*c < 0:
            t = c - b**2/(4*a)
            return Div([Comp([poly([b/(2*a), 1]) * (1/cmath.sqrt(t)), atan()]), a * cmath.sqrt(t)])
        elif b ** 2 - 4*a*c == 0:
            return Div([-1, poly([b/2, a])])
        else:
            r1, r2 = p.roots()
            alpha = 1/(a*(r1 - r2))
            return Sum([alpha * Comp([poly([-r1, 1]), log()]), -alpha*Comp([poly([-r2, 1]), log()])])

    q = Div([2 * p.diff(), (2*n-1) * (4*a*c-b**2) * (p ** (m - 1))])
    return Sum([q, 8*a*(n-1) * integrate_quad_n(p, m-1) * (1/ ((2*n-1) * (4*a*c-b**2)))])

def integrate_mono_quad(m, p, n):
    # integrates x^m / (p(x)^n) p(x) is the same from above
    if n == 0:
        return (poly([0, 1]) ** m).integrate()
    if m == 0:
        return integrate_quad_n(p, n)
    elif m == 1:
        a, b = p.coeffs[2], p.coeffs[1]
        f1 = (-b/(2*a)) * integrate_quad_n(p, n)
        if n == 1:
            f2 = (1/(2*a)) * Comp([p, log()])
        else:
            f2 = Div([1, (2*a*(1-n)) * p ** (n-1)])
        return Sum([f1, f2])

    else:
        a, b, c = p.coeffs[2], p.coeffs[1], p.coeffs[0]
        q0 = Div([-poly([0, 1])**(m - 1), a * (2*n - m - 1) * p ** (n - 1)])
        q1 = (b * (m - n) / (a * (2*n - m - 1))) * integrate_mono_quad(m-1, p, n)
        q2 = (c * (m - 1) / (a * (2*n - m - 1))) * integrate_mono_quad(m - 2, p, n)
        return Sum([q0, q1, q2])

def integrate_poly_quad_n(q, p, n):
    #integrates q(x) / (p(x) ** n)
    return Sum([integrate_mono_quad(i, p, n) * q.coeffs[i] for i in range(len(q.coeffs))])

def ratexp_integration_real(np, nq):
    if np.deg >= nq.deg:
        p = np % nq
        q = nq
        z = (np / nq).integrate()

    else:
        p = np
        q = nq
        z = 0

    coeffs = partial_frac_decomp(p, q)
    print(coeffs)
    all_roots = coeffs[:]
    real_roots = []
    imag_roots = []
    for j in range(len(all_roots)):
        c, r = all_roots[j]
        nr = complex(round(r.real, ndigits=5), round(r.imag, ndigits=5))
        if nr.imag == 0:
            real_roots.append([c[:], nr])
        else:
            cond = False
            for k in range(len(imag_roots)):
                if (abs(imag_roots[k][1] - nr) < 0.00001 or abs(imag_roots[k][1] - nr.conjugate()) < 0.00001) and k != j:
                    cond = True
                    break

            if not cond:
                sub_arr = []
                for k in range(len(all_roots)):
                    cp, rp = all_roots[k]
                    if abs(nr.conjugate() - rp) < 0.00001 and k != j:
                        sub_arr = cp[:]
                        break
                else:
                    print('Warning! denominator is not a real polynomial!')
                polys = []
                for i in range(len(c)):
                    polys.append(c[i] * poly([-nr.conjugate(), 1])**(i+1) + sub_arr[i] * poly([-nr, 1])**(i+1))
                imag_roots.append([polys[:], nr])

    s = []
    for c, r in real_roots:
        pol = poly([-r, 1])
        for k in range(len(c)):
            nck = round(c[k].real if isinstance(c[k], complex) else c[k], ndigits=5)
            if nck != 0 :
                if k == 0:
                    s.append(nck * Comp([pol, log()]))
                else:
                    s.append(Div([-nck, k * pol ** (k)]))


    for parr, r in imag_roots:
        pol = poly([abs(r) ** 2, -2*r.real, 1])
        for k in range(len(parr)):
            nparrk = poly([i.real for i in parr[k].coeffs[:]])

            s.append(integrate_poly_quad_n(nparrk, pol, k + 1))

    f = 0
    for k in s:
        f += k
    
    answer = f+z
    if all([answer(i) == 0 for i in range(1, 10)]):
        answer = Sum(s)

    return answer









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


def real_fourier_series_poly_2(p, a, b, T):
    l = T / 2
    def a_n (n):
        if n == 0:
            f = p.integrate()
            return (f(b) - f(a)) / (2*l)
        f1, f2 = solve_poly_trig(p, n * math.pi / l, t = 1)
        func = lambda x : f1(x) * cmath.exp(n*cmath.pi*complex(0, 1)*x/l ) + f2(x) * cmath.exp(-n*cmath.pi*complex(0, 1)*x/l) 
        return (func(b) - func(a)) / l
    
    def b_n (n):
        if n == 0: return 0
        f1, f2 = solve_poly_trig(p, n * math.pi / l)
        func = lambda x : f1(x) * cmath.exp(n*cmath.pi*complex(0, 1)*x/l ) + f2(x) * cmath.exp(-n*cmath.pi*complex(0, 1)*x/l)
        return (func(b) - func(a)) / l
    
    return a_n, b_n

def complex_fourier_series_poly(p, a, b, T):
    def res(n):
        f = lambda x : solve_poly_exp(p, complex(0, -n*2*math.pi/T))(x) * cmath.exp(complex(0, -n*2*math.pi*x/T))
        return (f(b) - f(a)) / T
    return res

def complex_fourier_transform(p, a, b):
    #finds the fourier transform of f = p(x) a<x<b 0 O.W.
    def f(w):
        g = solve_poly_exp(p, complex(0, -w))
        return g(b)*cmath.exp(complex(0, -w*b)) - g(a)*cmath.exp(complex(0, -w*a))
    return f

def mult_cft(p_array, range_array):
    '''
    if p_array = [p1, p2, p3, ...]
    and 
    range_array = [(a00, a01), (a10, a11), ...]
    
    computes the fourier transform of f(x) = p1(x)(u(x-a00) - u(x-a01)) + p2(x)(u(x-a10)-u(x-a11)) + ...
    where u(x) is the heaviside step function.
    '''
    return lambda w : sum([complex_fourier_transform(p_array[i], range_array[i][0], range_array[i][1])(w) for i in range(len(p_array))])

def mult_cfs(p_array, range_array, period):
    '''
    if p_array = [p1, p2, p3, ...]
    and 
    range_array = [(a00, a01), (a10, a11), ...]
    
    computes the fourier transform of f(x) = p1(x)(u(x-a00) - u(x-a01)) + p2(x)(u(x-a10)-u(x-a11)) + ...
    where u(x) is the heaviside step function.
    '''
    return lambda n : sum([complex_fourier_series_poly(p_array[i], range_array[i][0], range_array[i][1], period)(n) for i in range(len(p_array))])

def mult_rfs(p_array, range_array, period):
    '''
    if p_array = [p1, p2, p3, ...]
    and 
    range_array = [(a00, a01), (a10, a11), ...]
    
    computes the fourier transform of f(x) = p1(x)(u(x-a00) - u(x-a01)) + p2(x)(u(x-a10)-u(x-a11)) + ...
    where u(x) is the heaviside step function.
    '''
    return lambda w : [sum([real_fourier_series_poly_2(p_array[i], range_array[i][0], range_array[i][1], period)[0](w) for i in range(len(p_array))]), sum([real_fourier_series_poly_2(p_array[i], range_array[i][0], range_array[i][1], period)[1](w) for i in range(len(p_array))])]
    
def inverse_fourier_cmplx(p, a, b):
    #finds the fourier transform of f = p(x) a<x<b 0 O.W.
    def f(w):
        g = solve_poly_exp(p, complex(0, w))
        return (g(b)*cmath.exp(complex(0, w*b)) - g(a)*cmath.exp(complex(0, w*a))) / (2*math.pi)
    return f

def integrate_trig_rat(p, q):
    '''
    p and q are polymvar with two variables x and y
    where x represents sin(x) and y, cos(x).
    finds an anti-derivative for p(sinx, cosx) / q(sinx, cosx)
    using weirestrauss sub.
    '''

    #finding the maximum degree of (1+t^2)
    max_p = 0
    for i in range(len(p.array)):
        for j in range(len(p.array[i])):
            if p.array[i][j][0] != 0 and i + j > max_p:
                max_p = i + j

    max_q = 0
    for i in range(len(q.array)):
        for j in range(len(q.array[i])):
            if q.array[i][j][0] != 0 and i + j > max_q:
                    max_q = i + j

    new_p, new_q = 0, 0
    for i in range(len(p.array)):
        for j in range(len(p.array[i])):
            if p.array[i][j][0] != 0:
                new_p += (poly([0, 2]) ** i) * (poly([1, 0, -1]) ** j) * (poly([1, 0, 1]) ** (max_p - (i + j)))

    for i in range(len(q.array)):
        for j in range(len(q.array[i])):
            if q.array[i][j][0] != 0:
                new_q += (poly([0, 2]) ** i) * (poly([1, 0, -1]) ** j) * (poly([1, 0, 1]) ** (max_q - (i + j)))
    d = max_q - max_p - 1

    if d > 0:
        new_p *= 2 * poly([1, 0, 1]) ** d
    else:
        new_p *= 2
        new_q *= poly([1, 0, 1]) ** (-d)

    f = ratexp_integration_real(new_p, new_q)

    return Comp([Comp([poly([0, 0.5]), tan()]), f])

def integrate_euler(p, q, r):
    # integrates the integral of p(x, sqrt(ax^2 + bx + c)) / q(x, sqrt(ax^2 + bx + c))
    # p and q are polymvar and x = x, y = sqrt(ax^2 + bx + c) and r is a poly and is ax^2 + bx + c
    a, b, c = r.coeffs[2], r.coeffs[1], r.coeffs[0]
    delta = b**2 - 4*a*c
    if delta > 0:
        
        r1, r2 = [i.real for i in r.roots()]
        # sqrt(ax^2 + bx + c) = (x-r1)t -> a(x-r2)/(x-r1) = t^2 ->  = (r1*t^2-a*r2)/(t^2 - a) = x -> sqrt(...) = a*(r1-r2)t/(t^2 - a)
        #dx = -2at(r1-r2) / (t^2-a)^2 dt
        max_p = 0
        for i in range(len(p.array)):
            for j in range(len(p.array[i])):
                if p.array[i][j][0] != 0 and i + j > max_p:
                    max_p = i + j

        max_q = 0
        for i in range(len(q.array)):
            for j in range(len(q.array[i])):
                if q.array[i][j][0] != 0 and i + j > max_q:
                        max_q = i + j

        new_p, new_q = 0, 0
        p_sq = poly([0, a*(r1 - r2)])
        p_x = poly([-a*r2, 0, r1])
        p_denom = poly([-a, 0, 1])
        for i in range(len(p.array)):
            for j in range(len(p.array[i])):
                if p.array[i][j][0] != 0:
                    new_p += (p_x ** i) * (p_sq ** j) * (p_denom ** (max_p - (i + j)))

        for i in range(len(q.array)):
            for j in range(len(q.array[i])):
                if q.array[i][j][0] != 0:
                    new_q += (p_x ** i) * (p_sq ** j) * (p_denom ** (max_q - (i + j)))
        d = max_q - max_p - 2

        if d > 0:
            new_p *= -2 * poly([0, a*(r1-r2)]) * p_denom ** d
        else:
            new_p *= -2 * poly([0, a*(r1-r2)])
            new_q *= p_denom ** (-d)


        f = ratexp_integration_real(new_p, new_q)

        return Comp([Div([Comp([r, sqrt()]), poly([-r1, 1])]), f])
    
    elif delta < 0:
        if a > 0:
            # letting sqrt(ax^2 + bx + c) = t - sqrt(a)x -> sqrt(ax^2 + bx + c) = t - (t^2 - c)/(b/sqrt(a) + 2t) = (t^2 + bt/sqrt(a) + c) / (b/sqrt(a) + 2t)
            # -> x = (t^2 - c)/(b + 2t*sqrt(a)) -> dx = (2sqrt(a)t^2+2bt + 2c sqrt(a))/(4at^2 + 2bsqrt(a)+b^2)
            t_var = Comp([r, sqrt()]) + poly([0, cmath.sqrt(a)])
            max_p = 0
            for i in range(len(p.array)):
                for j in range(len(p.array[i])):
                    if p.array[i][j][0] != 0 and i + j > max_p:
                        max_p = i + j

            max_q = 0
            for i in range(len(q.array)):
                for j in range(len(q.array[i])):
                    if q.array[i][j][0] != 0 and i + j > max_q:
                            max_q = i + j

            new_p, new_q = 0, 0
            p_sq = poly([c, b/cmath.sqrt(a), 1])
            p_x = poly([-c, 0, 1])
            p_denom = poly([b, 2*cmath.sqrt(a)])
            for i in range(len(p.array)):
                for j in range(len(p.array[i])):
                    if p.array[i][j][0] != 0:
                        new_p += (p_x ** i) * (p_sq ** j) * (p_denom ** (max_p - (i + j)))

            for i in range(len(q.array)):
                for j in range(len(q.array[i])):
                    if q.array[i][j][0] != 0:
                        new_q += (p_x ** i) * (p_sq ** j) * (p_denom ** (max_q - (i + j)))
            d = max_q - max_p - 2

            if d > 0:
                new_p *= poly([2*c*cmath.sqrt(a), 2*b, 2*cmath.sqrt(a)]) * p_denom ** d
            else:
                new_p *= poly([2*c*cmath.sqrt(a), 2*b, 2*cmath.sqrt(a)])
                new_q *= p_denom ** (-d)

            print(new_p)
            print("-----------")
            print(new_q)
            f = ratexp_integration_real(new_p, new_q)

            return Comp([t_var, f])
        elif c > 0:
            # letting sqrt(ax^2 + bx + c) = tx + sqrt(c) -> x = (2sqrt(c)t - b) / (-t^2+a) -> sqrt(ax^2 + bx + c) = (sqrt(c)t^2 - bt + a*sqrt(c))/(a - t^2)
            # dx = (2sqrt(c)t^2 - 2b t + 2sqrt(c)a) / (a-t^2)^2 
            t_var = Div([Comp([r, sqrt()]) - cmath.sqrt(c), poly([0, 1])])
            max_p = 0
            for i in range(len(p.array)):
                for j in range(len(p.array[i])):
                    if p.array[i][j][0] != 0 and i + j > max_p:
                        max_p = i + j

            max_q = 0
            for i in range(len(q.array)):
                for j in range(len(q.array[i])):
                    if q.array[i][j][0] != 0 and i + j > max_q:
                            max_q = i + j

            new_p, new_q = 0, 0
            p_sq = poly([a*cmath.sqrt(c), b, cmath.sqrt(c)])
            p_x = poly([-b, 2*cmath.sqrt(c)])
            p_denom = poly([a, 0, -1])
            for i in range(len(p.array)):
                for j in range(len(p.array[i])):
                    if p.array[i][j][0] != 0:
                        new_p += (p_x ** i) * (p_sq ** j) * (p_denom ** (max_p - (i + j)))

            for i in range(len(q.array)):
                for j in range(len(q.array[i])):
                    if q.array[i][j][0] != 0:
                        new_q += (p_x ** i) * (p_sq ** j) * (p_denom ** (max_q - (i + j)))
            d = max_q - max_p - 2

            if d > 0:
                new_p *= poly([2*a*cmath.sqrt(c), -2*b, 2*cmath.sqrt(c)]) * p_denom ** d
            else:
                new_p *= poly([2*a*cmath.sqrt(c), -2*b, 2*cmath.sqrt(c)])
                new_q *= p_denom ** (-d)

            f = ratexp_integration_real(new_p, new_q)

            return Comp([t_var, f])            

