import math
import random
import time
import utils
import evaluator as evl
import statistics as st
import circ_analysis as circ
import turtle
import problem_set_calc
import problem_set_integral, fourier_prob_set, laplace_problem_set, lincont_problem_set
#import circuit_analyze as circ

'''
The output of each function is as follows : 
[resulting_string, expected_result, conversion_functon]
'''

def poly_conv_function(wentry):
    entry = [int(i) for i in wentry.split(" ")]
    entry.reverse()
    return utils.poly(entry[:]) 

def integral_conv_func(inp, ndigits):
    return round(evl.evl(inp), ndigits=ndigits)

def string_sgn(x):
    if x > 0:
        return "+"
    elif x < 0:
        return "-"
    else:
        return "0"

def regMul(inpt_dict):
    nrange = inpt_dict["nranges"]
    float_mode = inpt_dict["float_mode"]
    ndigits = inpt_dict["ndigits"]
    n1 = random.randint(nrange[0], nrange[1]) + float_mode*round(random.random(), ndigits)
    n2 = random.randint(nrange[0], nrange[1]) + float_mode*round(random.random(), ndigits)
    return ["%d * %d = "%(n1, n2) if not float_mode else "%f * %f = "%(n1, n2), n1 * n2, lambda x : float(x)]

def polyMul(inpt_dict):
    nrange = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    p1 = utils.poly.rand(random.randint(1, max_deg), coeff_range=nrange[:]) 
    p2 = utils.poly.rand(random.randint(1, max_deg), coeff_range=nrange[:]) 
    q = utils.connect([[" "], ["("], [" "]], utils.connect(p1.pprint(), [[" "], [")"], [" "]]))
    r = utils.connect([[" "], ["("], [" "]], utils.connect(p2.pprint(), [[" "], [")"], [" "]]))
    s = utils.strpprint(utils.connect(q, utils.connect([[" "], [" "], [" "]], r))) + "\nRes = "
    
    return [s, p1 * p2, poly_conv_function]

def polyEval(inpt_dict):#(deg=3, coeffs_range=[1, 10], input_range=[1, 10]):
    deg = inpt_dict["deg"]
    coeffs_range = inpt_dict["nranges"]
    input_range = inpt_dict["inp_ranges"]
    p = utils.poly.rand(deg, coeff_range=coeffs_range)
    x = (-1) ** random.randint(1, 10) * random.randint(input_range[0], input_range[1])
    string = "%s\t at x = %d : \n"%(str(p), x)
    return [string, p(x), lambda x : float(x)]

def evalRoot(inpt_dict):#(root_range=[2, 5], ranges=[100, 1000], ndigits = 2):
    root_range = inpt_dict["root_ranges"]
    ranges = inpt_dict["nranges"]
    ndigits = inpt_dict["ndigits"]
    p = utils.AlgebraicReal.randpurer(1, nrange_surd=ranges[:], nrange_root=root_range[:])
    string = str(p) + "\nr = "
    return [string, round(p(), ndigits=ndigits), lambda x : float(x)]

def evalRootPoly(inpt_dict):#(deg, coeffs_range = [10, 100], root_range=[2, 5], ranges=[100, 1000], ndigits = 2):
    deg = inpt_dict["deg"]
    coeffs_range = inpt_dict["nranges"]
    root_range = inpt_dict["root_ranges"]
    ranges = inpt_dict["nranges"]
    ndigits = inpt_dict["ndigits"]
    r = utils.AlgebraicReal.randpurer(1, nrange_surd=ranges[:], nrange_root=root_range[:])
    p = utils.poly.rand(deg, coeff_range=coeffs_range)
    string = str(p) + "\n at x = \n" + str(r)
    res = round(p(r()), ndigits=ndigits)
    return [string, res, lambda x : float(x)]

def surd(inpt_dict):#(ranges=[100, 1000], ndigits = 2):
    ranges = inpt_dict["nranges"]
    ndigits = inpt_dict["ndigits"]
    n = utils.rational.rand(nrange=ranges[:])
    string = utils.strpprint(utils.connect(n.pprint(), [["   "], [" = "], ["   "]]))
    entry_string = "%s \n\t"%string
    res = round(n(), ndigits=ndigits)
    return [entry_string, res, lambda x : float(x)]

def polyDiv(inpt_dict):#( max_deg=5, nrange=[10, 100], ndigits = 2):
    max_deg = inpt_dict["deg"]
    nrange = inpt_dict["nranges"]
    ndigits = inpt_dict["ndigits"]
    p1 = utils.poly.rand(random.randint(1, max_deg), coeff_range=nrange[:]) 
    p2 = utils.poly.rand(random.randint(1, p1.deg), coeff_range=nrange[:]) 
    wentry_string = "%s \n%s \n > "%(str(p1), str(p2))
    return [wentry_string, round(p1 / p2, ndigits=ndigits), poly_conv_function]

def div(inpt_dicts):#(ranges=[100, 1000], ndigits=5):
    ranges = inpt_dicts["nranges"]
    ndigits = inpt_dicts["ndigits"]
    n = utils.rational.rand(nrange=ranges[:])
    string = utils.strpprint(utils.connect(n.pprint(), [["   "], [" = "], ["   "]])) + "\n\t"
    return [string, round(n(), ndigits=ndigits), lambda x : float(x)]

def polyroots(inpt_dict):#(root_range=[1, 10], deg=3):
    root_range = inpt_dict["nranges"]
    deg = inpt_dict["deg"]
    z = utils.poly([1])
    rarray = []
    for j in range(deg):
        q = (-1)**(random.randint(1, 10)) * random.randint(root_range[0], root_range[1])
        rarray.append(-q)
        z *= utils.poly([q, 1])
    string = str(z) + "Answer : "
    narr = list(sorted(rarray))
    res = sum([narr[i] ** (deg - i) for i in range(deg)])
    return string, res, lambda x : float(x)

def polydisc(inpt_dict):#(coeff_range=[1, 10], deg=3):
    coeff_range = inpt_dict["nranges"]
    deg = inpt_dict["deg"]
    z = utils.poly.rand(deg, coeff_range=coeff_range[:])
    string = str(z) + "\nDiscriminant : "
    res = z.disc()
    return string, res, lambda x: float(x)

def partialFraction(inpt_dict):#(max_deg=4, nrange=[1, 10]):
    max_deg = inpt_dict["deg"]
    nrange = inpt_dict["nranges"]
    z = []
    for j in range(max_deg):
        x = (-1)**random.randint(1, 2) * random.randint(nrange[0], nrange[1])
        while x in z:
            x = (-1)**random.randint(1, 2) * random.randint(nrange[0], nrange[1])
        z.append(x)
    z.sort()
    z.reverse()
    q = [utils.poly([j, 1]) for j in z]
    a = 1
    for j in q:
        a *= j
    v = [random.randint(nrange[0], nrange[1]) * ((-1)**random.randint(1,2)) for j in range(len(q))]
    p = utils.poly([0])
    for j in range(len(q)):
        pol_arr = q[:j] + q[j+1:] if j < len(q) - 1 else q[:j]
        r = utils.poly([1])
        for k in pol_arr:
            r *= k
        p += r*v[j]
    str1 = str(p)
    str2 = str(a)
    str1cpy = str1[:]
    str2cpy = str2[:]
    len_measure1 = len(str1cpy.split("\n")[0])
    len_measure2 = len(str2cpy.split("\n")[0])
    str3 = "".join(["-" for j in range(max(len_measure1, len_measure2))])
    new_str = str1 + "\n" + str3 + "\n" + str2 + "\n" + "res : "
    res =  " ".join([str(j) for j in v])
    return new_str, res, lambda x : x

def subIntGame(inpt_dict):#(mode=1, deg=2, nranges=[1, 10], boundranges=[1, 10], ndigits=2):
    mode = inpt_dict["mode"] if inpt_dict["mode"] != 5 else random.randint(1, 4)
    deg = inpt_dict["deg"]
    nranges = inpt_dict["nranges"]
    boundranges = inpt_dict["boundary_ranges"]
    ndigits = inpt_dict["ndigits"]
    if mode == 1:
            p, q, string = utils.generate_integrable_ratExpr(deg, nranges=nranges[:])
            a, b = random.randint(boundranges[0], boundranges[1]), random.randint(boundranges[0], boundranges[1])
            res = round(utils.numericIntegration(lambda x : p(x) / q(x), min(a, b), max(a, b)), ndigits=ndigits)
            string2 = "Evaluate the integral of the function below from %d to %d\n"%(min(a, b), max(a, b)) + string + "\nI = "
            
            
    elif mode == 2:
        f, string = utils.generate_eulersub_rand(deg, nranges=nranges[:])
        a, b = random.randint(boundranges[0], boundranges[1]), random.randint(boundranges[0], boundranges[1])
        res = round(utils.numericIntegration(f, min(a, b), max(a, b)), ndigits=ndigits)
        string2 = "Evaluate the integral of the function below from %d to %d\n"%(min(a, b), max(a, b)) + string + "\nI = "
        
        
    
    elif mode == 3:
        f, string = utils.generate_trig(nranges=nranges[:])
        a, b = random.randint(boundranges[0], boundranges[1]), random.randint(boundranges[0], boundranges[1])
        res = round(utils.numericIntegration(f, min(a, b), max(a, b)), ndigits=ndigits)
        string2 = "Evaluate the integral of the function below from %d to %d\n"%(min(a, b), max(a, b)) + string + "\nI = "
    
    elif mode == 4:
            p, q, string = utils.generate_ratExpr(deg, nranges=nranges[:])
            a, b = random.randint(boundranges[0], boundranges[1]), random.randint(boundranges[0], boundranges[1])
            res = round(utils.numericIntegration(lambda x : p(x) / q(x), min(a, b), max(a, b)), ndigits=ndigits)
            string2 = "Evaluate the integral of the function below from %d to %d\n"%(min(a, b), max(a, b)) + string + "\nI = "
            
    
    return [string2, res, lambda x : integral_conv_func(x, ndigits)]

def realIntGame(inpt_dict):
    nranges = inpt_dict["nranges"]
    branges = inpt_dict["branges"]
    n = inpt_dict["n"]
    k = inpt_dict["k"]
    comp = inpt_dict["comp"]
    sums = inpt_dict["sums"]
    prod = inpt_dict["prod"]
    max_d = inpt_dict["maxd"]
    moe = inpt_dict["moe"]
    res, string, lb, hb = utils.generate_integral_problem(nranges=nranges, boundary_ranges=branges, n=n, k=k, max_deg=max_d, comp=comp, sums=sums, prod=prod)
    cond = lambda x : (1-moe) * res <= abs(evl.evl(x)) <= (1+moe)*res or (1+moe) * res <= evl.evl(x) <= (1-moe)*res
    return [string, res, lambda x : res if cond(x) else res + 1000]
def realIntGameHARD(inpt_dict):
    nranges = inpt_dict["nranges"]
    branges = inpt_dict["branges"]
    n = inpt_dict["n"]
    fweights = [int(i) for i in inpt_dict["fweights"]]
    wweights = [int(i) for i in inpt_dict["wweights"]]
    moe = inpt_dict["moe"]
    res, string, lb, hb = utils.generate_integral_problem_iii(nranges=nranges, boundary_ranges=branges, n=n, fweights=fweights, wweights=wweights)
    cond = lambda x : abs(res - evl.evl(x)) <= abs(moe * res)
    return [string, res, lambda x : res if cond(x) else res + 1000] 
def regMulDig(inpt_dict):#(digits=5):
    digits = inpt_dict["ndigits"]
    n1 = random.randint(10 ** (digits - 1), 10 ** (digits) - 1) 
    n2 = random.randint(10 ** (digits - 1), 10 ** (digits) - 1) 
    n1s = ' '.join([i for i in str(n1)])
    n2s = ' '.join([i for i in str(n2)])
    string = "  %s\n* %s\n"%(n1s, n2s) + ''.join(['-' for i in range(max(len(n1s), len(n2s)) + 2)]) + '\n'
    return [string, n1 * n2, lambda x : int(x)]

def fourierSeries(inpt_dict):#(nranges=[1, 10], deg=2, p_range=[0, 2], exp_cond=False, u_cond=False, umvar_cond=False, moe=0.01):
    nranges = inpt_dict["nranges"]
    deg = inpt_dict["deg"]
    p_range = inpt_dict["period_ranges"]
    exp_cond = inpt_dict["exp_cond"]
    u_cond = inpt_dict["u_cond"]
    umvar_cond = inpt_dict["umvar_cond"]
    moe = inpt_dict["moe"]
    n_partite = inpt_dict["n_partite"]
    f, period, a_n, b_n, a_0, string, p1, c1 = utils.generate_fourier_s(nranges=nranges[:], n_partite=n_partite, deg=random.randint(0, deg), p_range=p_range, exp_cond=exp_cond, u_cond=u_cond, umvar_cond=umvar_cond)
    s = 'If f(x) = a0 + a1Cos(Lx) + b1Sin(Lx) + ... then find  a0 + a1 + b1 if f(x) = \n'
    s += string + "\n with a period of " + str(period) + "\n"
    res = a_0 + a_n(1) + b_n(1)
    cond = lambda x : (1-moe) * res <= evl.evl(x) <= (1+moe)*res or (1+moe) * res <= evl.evl(x) <= (1-moe)*res
    return [s, res, lambda x : res if cond(x) else res + 1000]
def fourierTransform(inpt_dict):#(nranges=[1, 10], deg=2, p_range=[0, 2], exp_cond=False, u_cond=False, umvar_cond=False, moe=0.01):
    nranges = inpt_dict["nranges"]
    deg = inpt_dict["deg"]
    p_range = inpt_dict["period_ranges"]
    exp_cond = inpt_dict["exp_cond"]
    u_cond = inpt_dict["u_cond"]
    umvar_cond = inpt_dict["umvar_cond"]
    moe = inpt_dict["moe"]
    n_partite = inpt_dict["n_partite"]
    coin = random.randint(0, 1)
    f, period, fct, string, p1, c1 = utils.generate_fourier_ct(nranges=nranges[:], n_partite=n_partite, deg=random.randint(0, deg), p_range=p_range, exp_cond=exp_cond, u_cond=u_cond, umvar_cond=umvar_cond) if not coin else utils.generate_fourier_st(nranges=nranges[:], n_partite=n_partite, deg=random.randint(0, deg), p_range=p_range, exp_cond=exp_cond, u_cond=u_cond, umvar_cond=umvar_cond)
    inp = random.randint(1, 10)
    s = string + "\nP = " + str(period) + "\n" + "F_%s{f}(%d) = "%("c" if not coin else "s", inp)
    res = fct(inp)
    cond = lambda x : (1-moe) * res <= evl.evl(x) <= (1+moe)*res or (1+moe) * res <= evl.evl(x) <= (1-moe)*res
    return [s, res, lambda x : res if cond(x) else res + 1000]

def lineq(inpt_dict):#coeff_abs_ranges=[1, 10], parameters=3, param_abs_ranges=[1, 10]):
    coeff_abs_ranges = inpt_dict["nranges"]
    parameters = inpt_dict["params"]
    param_abs_ranges = inpt_dict["param_ranges"]
    variable_names = ["x", "y", "z", "w", "n", "m", "p", "q", "r", "s", "t"]
    answers = [((-1)**random.randint(1, 2)) * random.randint(param_abs_ranges[0], param_abs_ranges[1]) for j in range(parameters)]
    equations = []
    rhs = []
    for j in range(parameters):
        c_eq = []
        for k in range(parameters):
            c_eq.append(((-1)**random.randint(1, 2)) *random.randint(coeff_abs_ranges[0], coeff_abs_ranges[1]))
        rhs.append(sum([answers[k] * c_eq[k] for k in range(parameters)]))
        equations.append(c_eq)
    
    eq_strings = []
    for j in range(parameters):
        c_str = []
        for k in range(parameters):
            if equations[j][k] != 0:
                if abs(equations[j][k]) != 1:
                    c_str.append(string_sgn(equations[j][k]))
                    c_str.append(str(abs(equations[j][k]))+variable_names[k])
                else:
                    c_str.append(string_sgn(equations[j][k]))
                    c_str.append(variable_names[k])
        
        if equations[j][0] > 0:
            c_str.pop(0)
                    
        eq_strings.append(" ".join(c_str) + " = " + str(rhs[j]))
    
    fin_str = "\n".join(eq_strings)
    fin_fin_str = "Find sum of the squares of the solutions to the system below. \n" + fin_str + "\n" 
    return fin_fin_str, sum([i**2 for i in answers]), lambda x : [int(i) for i in x.split(" ")]

def mean(inpt_dict):#(nrange=[1, 10], n=10, ndigits=2):
    nrange = inpt_dict["nranges"]
    n = inpt_dict["n"]
    ndigits = inpt_dict["ndigits"]
    array = [random.randint(nrange[0], nrange[1]) * ((-1)**random.randint(1,2)) for j in range(n)]
    m = round(st.mean(array[:]), ndigits=ndigits)
    s = ", ".join([str(j) for j in array]) + "\nS = "
    
    return [s, m, lambda x : float(x)]

def stdev(inpt_dict):#(nrange=[1, 10], n=10, ndigits=2):
    nrange = inpt_dict["nranges"]
    n = inpt_dict["n"]
    ndigits = inpt_dict["ndigits"]
    array = [random.randint(nrange[0], nrange[1]) * ((-1)**random.randint(1,2)) for j in range(n)]
    m = round(st.stdev(array[:]), ndigits=ndigits)
    string = ", ".join([str(i) for i in array]) + "\nS = "
    return [string, m, lambda x : float(x)]

def diffeq(inpt_dict):#(nranges=[1, 10], max_deg=2):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    order = inpt_dict["ord"]
    inp_ranges = inpt_dict['inprange']
    f, s, iv = utils.rand_diffeq_sym(nranges[:], order, max_deg)
    z = random.randint(inp_ranges[0], inp_ranges[1])
    nstr = "y(0) = " + str(iv[0]) + "\n" + "y'(0) = " + str(iv[1]) + "\n" + s + "\n" + "y(%d) = "%z 
    cond = lambda x : abs(evl.evl(x) - f(z)) < 0.1
    return [nstr, f(z), lambda x : z if cond(x) else z + 1]

def diffeq_mixed(inpt_dict):#(nranges=[1, 10], max_deg=2):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    order = inpt_dict["ord"]
    f, s, iv = utils.random_diff_eq_ord_mixed(order=order, nranges=nranges, n=random.randint(0, 1), max_deg=max_deg)
    z = round(random.random(), ndigits=2)
    nstr = "y(0) = " + str(iv[0]) + "\n" + "y'(0) = " + str(iv[1]) + "\n" + s + "\n" + "y(%f) = "%z 
    cond = lambda x : f(z) * 0.8 <round(evl.evl(x), ndigits=2)< f(z)*1.2
    return [nstr, f(z), lambda x : z if cond(x) else z + 1]

def pcurve(inpt_dict):#(nranges=[1, 10], max_deg=2, ndigits=2):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    ndigits = inpt_dict["ndigits"]
    pc = utils.pcurve.rand(max_deg=max_deg, nranges=nranges[:])
    inp = random.randint(nranges[0], nranges[1])
    k = pc.curvature()(inp)
    string = str(pc) + "\n" + "find the curvature at x = %d : "%inp
    res = round(k, ndigits=ndigits) 
    return string, res, lambda x : round(evl.evl(x), ndigits=ndigits)

def pcurveT(inpt_dict):#(nranges=[1, 10], max_deg=2, ndigits=2):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    ndigits = inpt_dict["ndigits"]
    pc = utils.pcurve.rand(max_deg=max_deg, nranges=nranges[:])
    inp = random.randint(nranges[0], nranges[1])
    k = pc.T()(inp)
    string = str(pc) + "\n" + "find the curvature at x = %d : "%inp
    res = round(k, ndigits=ndigits) 
    return string, res, lambda x : round(evl.evl(x), ndigits=ndigits)

def lineIntegral(inpt_dict):#(nranges=[1, 10], max_deg=2, moe=0.001):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    moe = inpt_dict["moe"]
    p = utils.pcurve.rand(max_deg=max_deg, nranges=nranges[:])
    v = utils.vectF.rand(max_deg=max_deg, nranges=nranges[:])
    init = random.randint(nranges[0], nranges[1])
    fin = random.randint(nranges[0], nranges[1])
    k = v.integrate(p)
    ans = k(fin) - k(init)
    s = "C : \n" +  str(p) + "\n" + "F = \n" + str(v) + "\nfind the line integral of F along C from %s to %s : "%(p(init), p(fin))
    cond = lambda x : ans-moe*ans <= x <= ans + moe*ans or ans+moe*ans <= x <= ans - moe*ans
    return [s, ans, lambda x : ans if cond(evl.evl(x)) else ans+1]

def divergence(inpt_dict):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    ndigits = inpt_dict["ndigits"]
    pc = utils.vectF.rand(max_deg=max_deg, nranges=nranges[:])
    inps = [random.randint(nranges[0], nranges[1]) for i in range(3)]
    k = pc.div()(inps[0], inps[1], inps[2])
    s = str(pc) + "\n" + "\nfind the divergance of F at %s: "% ("(" + ", ".join([str(y) for y in inps]) + ")")
    cond = lambda x : round(x, ndigits=ndigits) == round(k, ndigits=ndigits)
    return [s, round(k, ndigits=ndigits), lambda x : k if cond(evl.evl(x)) else k+1]

def lineIntegralScalar(inpt_dict):#(nranges=[1, 10], max_deg=2, moe=0.001):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    moe = inpt_dict["moe"]
    p = utils.pcurve.rand(max_deg=max_deg, nranges=nranges[:])
    v = utils.polymvar.rand(max_deg=max_deg, nrange=nranges[:])
    init = random.randint(nranges[0], nranges[1])
    fin = random.randint(nranges[0], nranges[1])
    ans = v.c_integrate(p, init, fin)
    s = "C : \n" + str(p) + "\n" + "F = \n" + str(v) + "\n" + "\nfind the line integral of F along C from %s to %s : "%(p(init), p(fin))
    cond = lambda x : ans-moe*ans <= x <= ans + moe*ans or ans+moe*ans <= x <= ans - moe*ans
    return [s, ans, lambda x : ans if cond(evl.evl(x)) else ans+1]

def regDet(inpt_dict):
    nrange = inpt_dict["nranges"]
    dims = inpt_dict["dim"]
    m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
    res = m.det()
    string = "What is the determinant of \n " + str(m) + "\n ?"
    return string, res, lambda x : float(x)

def eigenValue(inpt_dict):
    dims = inpt_dict["dim"]
    nrange = inpt_dict["nranges"]
    ndigits = inpt_dict["ndigits"]
    m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
    string = "Find the greatest integer less than the absolute value of the sum of the squares of the eigenvalues for the matrix below : \n"
    string += str(m) + "\n" 
    eigens = [int(i.real * 10 ** ndigits) / 10 ** ndigits for i in m.eigenvalue()]
    ans = math.floor(abs(sum([i ** 2 for i in eigens])))
    return string, ans, lambda x : float(x)

def polyDet(inpt_dict):
    dims = inpt_dict["dim"]
    nrange = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    m = utils.matrix.randpoly(dims=[dims, dims], max_deg=max_deg, coeff_range=nrange[:])
    res = min([abs(i) for i in m.det().roots()])
    res = int(100 * res) / 100
    string = str(m) + "\n" + "Find the root of the determinant with the least abs (accurate to 2 digits):  "
    return string, res, lambda x : float(x)


FUNCTIONS_ARRAY = [regMul, polyMul, polyEval, evalRoot, evalRootPoly, surd, polyDiv, div, polyroots, polydisc, partialFraction, subIntGame, 
                   regMulDig, fourierSeries, lineq, mean, stdev, diffeq, pcurve, pcurveT, divergence, lineIntegral, lineIntegralScalar, eigenValue,
                   polyDet]
CALC_ARRAY = [subIntGame, diffeq, lineIntegral, lineIntegralScalar, divergence, fourierSeries, pcurve, pcurveT]
ARITHMETIC_ARRAY = [regMul, regDet, evalRoot, div, regDet, regDet, regDet]
LINEAR_ARRAY = [regDet, eigenValue, lineq, polyDet]
FAVORITE_ARRAY = [regDet, polyDet]

def interpolationGame(inpt_dict):
    n = inpt_dict["n"]
    nrange_inps = inpt_dict["nranges-inps"]
    nranges_coeffs = inpt_dict["nranges"]
    p = utils.poly.rand(n - 1, coeff_range=nranges_coeffs[:])
    points = []
    X = []
    for i in range(n):
        q = random.randint(nrange_inps[0], nrange_inps[1])
        while q in X:
            q = random.randint(nrange_inps[0], nrange_inps[1])
        points.append((q, p(q)))
        X.append(q)
    
    n_inp = random.randint(nrange_inps[0], nrange_inps[1])
    res = p(n_inp)
    string = "\n".join(["(" + str(i) + ", " + str(j) + ")" for i, j in points]) + "\n" + "evaluate at x = " + str(n_inp) + "\n"
    return string, res, lambda x : int(x)

def diffeqPoly(inpt_dict):#(nranges=[1, 10], max_deg=2):
    nranges = inpt_dict["nranges"]
    max_deg_coeffs = inpt_dict["degc"]
    max_deg_rhs = inpt_dict["deg"]
    f, s, iv = utils.random_diff_eq_2_poly(nranges, mdeg_coeffs=max_deg_coeffs, max_deg=max_deg_rhs)
    z = round(random.random(), ndigits=2)
    nstr = "y(0) = " + str(iv[0]) + "\n" + "y'(0) = " + str(iv[1]) + "\n" + s + "\n" + "y(%f) = "%z 
    cond = lambda x : f(z) * 0.8 <round(evl.evl(x), ndigits=2)< f(z)*1.2
    return [nstr, f(z), lambda x : z if cond(x) else z + 1]

def PDEConst(inpt_dict):
    nranges = inpt_dict["nranges"]
    l_ranges = inpt_dict["l-ranges"]
    moe = inpt_dict["moe"]
    sep = inpt_dict["sep"]
    sol, string, l, z, s = utils.randomPDEconst(nranges, l_ranges, sep=sep)
    inp1, inp2 = random.randint(0, l-1) + round(random.random(), ndigits=1), random.randint(0, l-1) + round(random.random(), ndigits=1)
    
    res = sol(inp1, inp2)
    str1 = "u(x, 0) = \n" + s[0] + "\nu(x, %d) = \n"%l + s[1] + "\nu(0, x) = \n" + s[2] + "\nu(%d, x) = \n"%l + s[3] + "\n" + string + "\nevaluate at x = %f, y = %f\n"%(inp1, inp2)
    cond = lambda x : res-moe*res <= x <= res + moe*res or res+moe*res <= x <= res - moe*res
    return str1, res, lambda x : res if cond(evl.evl(x)) else res+1

def PDESpecial(inpt_dict):
    nranges = inpt_dict["nranges"]
    l_ranges = inpt_dict["l-ranges"]
    moe = inpt_dict["moe"]
    sol, string, l, z, s = utils.specialPDE(nranges, l_ranges)
    inp1, inp2 = random.randint(0, l-1) + round(random.random(), ndigits=1), random.randint(0, l-1) + round(random.random(), ndigits=1)
    
    res = sol(inp1, inp2)
    str1 = "u(x, 0) = \n" + s[0] + "\nu(x, %d) = \n"%l + s[1] + "\nu(0, x) = \n" + s[2] + "\nu(%d, x) = \n"%l + s[3] + "\n" + string + "\nevaluate at x = %f, y = %f\n"%(inp1, inp2)
    cond = lambda x : res-moe*res <= x <= res + moe*res or res+moe*res <= x <= res - moe*res
    return str1, res, lambda x : res if cond(evl.evl(x)) else res+1

def complex_integral(inpt_dict):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["mdeg"]
    n = inpt_dict["n"]
    moe = inpt_dict["moe"]
    clsd = inpt_dict["clsd"]
    repeat = inpt_dict["rep"]
    mrep = inpt_dict["mrep"]
    boundary_ranges = inpt_dict["branges"]
    if clsd:
        fstring, ppr_string, res = utils.random_f_c_integrate(nranges=nranges[:],max_deg=max_deg, n=n, mrep=mrep, repeat=repeat, clsd=True)
        string = fstring + "\n" + ppr_string+"\n"
        cond = lambda x : (abs(x - res) <= abs(moe * res)) or abs(x - res) <= 0.0005
        return string, res, lambda x : res if cond(complex(evl.evl(x.split(" ")[0]), evl.evl(x.split(" ")[1]))) else res + 1
    else:
        fstring, ppr_string, res, s, e = utils.random_f_c_integrate(nranges=nranges[:],max_deg=max_deg, n=n, mrep=0, repeat=repeat, clsd=False, boundary_ranges=boundary_ranges[:])
        string = fstring + "\n" + ppr_string+"\n" + "from t = " + str(s) + " to t = " + str(e) + "\n"
        cond = lambda x : (abs(x - res) <= abs(moe * res)) or abs(x - res) <= 0.0005
        return string, res, lambda x : res if cond(complex(evl.evl(x.split(" ")[0]), evl.evl(x.split(" ")[1]))) else res + 1
def integral_cmplx(inpt_dict):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["mdeg"]
    moe = inpt_dict["moe"]
    mode = inpt_dict["mode"]
    
    if not mode:
        f,  res, s = utils.generate_integral_cmplx_rat(nranges=nranges, max_deg=max_deg)
        string = "Evaluate the integral for f(x) = "+"\n"
        string += s + "\n" + "from -∞ to +∞ : "
        cond = lambda x : abs(x - res) <= abs(moe*res) or abs(x - res) <= 0.0005
        return string, res, lambda x : res if cond(evl.evl(x)) else res+1
    
    else:
        res, s, n = utils.generate_trig_cmplx(nranges)
        string = "Evaluate the integral for f(x) = "+"\n"
        string += s + "\n" + "from 0 to %s : " % ("2π/%d"%n if n != 1 else "2π")
        cond = lambda x : abs(x - res) <= abs(moe*res) or abs(x - res) <= 0.0005
        return string, res, lambda x : res if cond(evl.evl(x)) else res+1

def maclaurin_series(inpt_dict):
    nranges = inpt_dict["nranges"]
    max_deg = inpt_dict["mdeg"]
    moe = inpt_dict["moe"]
    
    s, r = utils.generate_rand_mseries(nranges, max_deg)
    res = r(0)**2 + r(1)**2 + r(2)**2 
    string = "Find the sum of the squares of the first three terms in the maclaurin series for f(x) = \n" + s + "\n"
    cond = lambda x : abs(x - res) <= abs(moe*res)
    return string, res, lambda x : res if cond(evl.evl(x)) else res+1
    
def funcMatDet(inpt_dict):
    ndigits = inpt_dict["ndigits"]
    dim = inpt_dict["dim"]
    dig = inpt_dict["dig"]
    
    res, string = utils.generate_matrix_item(ndigits=ndigits, dim=dim, calc_ndigits=dig)
    cond = lambda x : res - 0.5 * 10 ** (-dig) <= x < res + 0.5 * 10 ** (-dig)
    return "Find the determinant for the matrix below accurate to %d digits. \n"%dig + string+"\n", res, lambda x : res if cond(float(x)) else res+1

def funcEval(inpt_dict):
    ndigits = inpt_dict["ndigits"]
    dig = inpt_dict["dig"]
    n = inpt_dict["N"]
    
    newstr, res = utils.generate_function_item(ndigits=ndigits, calc_ndigits=dig, n=n)
    cond = lambda x : res - 0.5 * 10 ** (-dig) <= x < res + 0.5 * 10 ** (-dig)
    return "Find the value of \n"+newstr+"\n accurate to %d digits"%dig, res, lambda x : res if cond(float(x)) else res+1 

def shuffle(inpt_dict):
    return random.choice(FUNCTIONS_ARRAY)(inpt_dict)

def calc_suite(inpt_dict):
    return random.choice(CALC_ARRAY)(inpt_dict)

def arithmetic_suite(inpt_dict):
    return random.choice(ARITHMETIC_ARRAY)(inpt_dict)

def linear_suite(inpt_dict):
    return random.choice(LINEAR_ARRAY)(inpt_dict)

def pick1(inpt_dict):
    return random.choice(FAVORITE_ARRAY)(inpt_dict)

def PDE(inpt_dict):
    arr = [PDEConst, PDESpecial]
    return arr[random.randint(0, 1)](inpt_dict)

def solvableInt(inpt_dict):
    tp = inpt_dict["type"]
    nranges = inpt_dict["nranges"]
    branges = inpt_dict["branges"]
    moe = inpt_dict["moe"]
    if tp == 0: #rational expression
        arr = [utils.rand_pfd_prop, 
               utils.rand_rat_prop, 
               utils.rand_rat_tan, 
               utils.rand_rat_cos]
        max_deg = inpt_dict["mdeg"]
        n_range = inpt_dict["n_range"]
        m_range = inpt_dict["m_range"]
        seed = random.randint(0, 3)
        if seed in [0, 1]:
            r, s, lb, hb = utils.generate_general_int(arr[seed], [nranges[:], max_deg], branges[:])
        
        elif seed in [2, 3]:
            r, s, lb, hb = utils.generate_general_int(arr[seed], [nranges[:], n_range, m_range], branges[:])
    
    elif tp == 1:#integration by parts
        arr = [utils.rand_int_part_i, utils.rand_int_part_ii]
        max_deg = inpt_dict["mdeg"]
        n = inpt_dict["n"]
        mdeg_c = inpt_dict["mdeg_c"]
        seed = random.randint(0, 1)
        r, s, lb, hb = utils.generate_general_int(arr[seed], [nranges[:], max_deg, n, mdeg_c], branges[:])
    
    elif tp == 2: #sqrt
        arr = [utils.rand_sqrt_type_i,
               utils.rand_sqrt_type_ii,
               utils.rand_sqrt_type_iii,
               utils.rand_sqrt_iv,
               utils.rand_sqrt_v,
               utils.rand_sqrt_vi]
        max_deg = inpt_dict["mdeg"]
        deg_i = inpt_dict["degi"]
        n_range = inpt_dict["n_range"]
        seed = random.randint(0, 5)
        r, s, lb, hb = utils.generate_general_int(arr[seed], [nranges[:], max_deg, deg_i, n_range], branges[:])
    
    elif tp == 3:#rand trig
        r, s, lb, hb = utils.generate_general_int(utils.rand_trig_i, [nranges[:]], branges[:])
    
    elif tp == 4:#rand poly
        n_range = inpt_dict["n_range"]
        r, s, lb, hb = utils.generate_general_int(utils.rand_poly_i, [nranges[:], n_range[:]], branges[:])
    
    cond = lambda x : abs(x - r) <= abs(moe*r)
    return s, r, lambda x : r if cond(evl.evl(x)) else r+1

def inv_laplace_game(inpt_dict):
    nranges = inpt_dict["nranges"]
    mdeg = inpt_dict["mdeg"]
    trange = inpt_dict["tranges"]
    moe = inpt_dict["moe"]
    t = random.randint(trange[0], trange[1])
    diff_int = inpt_dict["diffint"]
    l, sf = utils.generate_invlaplace_transform_problem(nranges=nranges[:], max_deg=mdeg, diff_int=diff_int)
    inv_lp = l(t)

    cond = lambda x : abs(x - inv_lp) <= abs(moe*inv_lp)
    s = utils.strpprint(sf.npprint()) + "\nEvaluate inverse Laplace transform at t = "+str(t)+"\n>"
    return s, inv_lp, lambda x : inv_lp if cond(evl.evl(x)) else inv_lp+1

def root_game_integer(inpt_dict):
    nranges = inpt_dict["nranges"]
    root_ranges = inpt_dict["rrange"]
    n = random.randint(root_ranges[0], root_ranges[1])
    mod_nrange = [int(nranges[0]**(1/n)) + 1, int(nranges[1]**(1/n))]
    res = random.randint(mod_nrange[0], mod_nrange[1])
    s = res ** n
    string = str(utils.AlgebraicReal([utils.rational([0, 1]), (1, s, n)]))
    
    return string, res, lambda x: int(x)

def numerical_analysis(inpt_dict):
    num_ranges = inpt_dict["numranges"]
    rat_ranges = inpt_dict["ratranges"]
    fun_ranges = inpt_dict["funranges"]
    inp_ndigit = inpt_dict["inpndigit"]
    var_number = inpt_dict["varnumber"]
    pparts_num = inpt_dict["ppartsnum"]
    pure_arith = inpt_dict["purearith"]
    res_ndigit = inpt_dict["resndigit"]
    
    
    n, s = utils.generate_mult_arithm_item(num_ranges=num_ranges[:], rat_range=rat_ranges[:], number_of_parts=pparts_num, number_of_variables=var_number, pure_arithm=pure_arith, fun_ranges=fun_ranges[:], inp_ndigits=inp_ndigit, res_ndigits=res_ndigit)
    
    return "Find the value of \n" + s + "\naccurate to %d digits"%res_ndigit+"\n", n, lambda x : n if abs(float(x) - n) < 10 ** (-7) else n+1

def inv_lap_mat(inpt_dict):
    nranges = inpt_dict["nranges"]
    dim = inpt_dict["dim"]
    mdeg = inpt_dict["mdeg"]
    mdeg_rhs = inpt_dict["mdeg_rhs"]
    ndigits_t = inpt_dict["mdeg_rhs"]
    moe = inpt_dict["moe"]
    l, rhs, ans, ans_t = utils.generate_sys_lap_problem(nranges=nranges, dim=dim, mdeg=mdeg, mdeg_rhs=mdeg_rhs)
    t = random.randint(-2, 2) + round(random.random(), ndigits=ndigits_t)
    string = "If Av = B (where the entries are in laplacian frequency domain) then find the length of the vector v in the time domain at t = " + str(t) + "\n"
    string  += "A = \n" + str(l) + "\n" + "B = \n" + str(rhs) + "\n" 
    answer = math.sqrt(sum([abs(i[0]) ** 2 for i in ans_t(t).array[:]]))
    cond = lambda x : abs(x - answer) <= abs(moe*answer)
    return string, answer, lambda x : answer if cond(float(x)) else answer+1

def circuit_game(inpt_dict):

    nranges = inpt_dict['nranges']
    tranges = inpt_dict['tranges']
    nnode = inpt_dict['nnode']
    nmesh = inpt_dict['nmesh']
    moe = inpt_dict['moe']
    prob = inpt_dict['prob']
    source_arr = [0 for i in range(prob - 1)] + [1]
    t, answer, a = circ.generate_circuit_problem(nranges, tranges, nnode, nmesh, source_arr=source_arr[:])
    

    cond = lambda x : abs(x - answer) <= abs(moe*answer)
    return 'Voltage at t = %f : '%t, answer, lambda x : answer if cond(float(x)) else answer+1

def diffDet(inpt_dict):
    nranges = inpt_dict['ndig']
    dim = inpt_dict['dim']
    matdeg = inpt_dict['matdeg']
    mdeg = inpt_dict['mdeg']
    inp_range = inpt_dict['inprange']
    f, s, mat= utils.diff_det(nranges, dim, matdeg, mdeg)
    z = random.randint(inp_range[0], inp_range[1])
    nstr = 'If p(D) is equal to the determinant of the matrix below; Then solve the equation below (zero state solution):\n'
    st = [["" for i in range(8)], ["" for i in range(8)], ["" for i in range(8)], ["p(D)y = "], ["" for i in range(8)], ["" for i in range(8)], ["" for i in range(8)]]
    nstr = nstr + utils.strpprint(utils.connect(st, s.npprint()[:]))
    nstr = nstr + "\n" + utils.matrixpprint(mat.pprint(prev_ppr = [[" "], [" "], [" "], ["D"], [" "], [" "], [" "]]))
    
    nstr += '\ny(%d) = '%z
    
    
    cond = lambda x : abs(evl.evl(x) - f(z)) < 0.01 * abs(f(z))
    def check(x):
        print("your result was calculated to be : ", evl.evl(x))
        print('your deviation from the answer was : ', abs(evl.evl(x) - f(z)))
        return f(z) if cond(x) else f(z) + 1
    return [nstr, f(z), check]


def matrixPoly(inpt_dict):
    mnranges = inpt_dict['mnranges']
    pnranges = inpt_dict['pnranges']
    deg = inpt_dict['deg']
    dim = inpt_dict['dim']
    
    p = utils.poly.rand(deg, coeff_range=pnranges)
    m = utils.matrix.rand(dims = [dim, dim], nrange=mnranges)
    s = 'Find the determinant of \n'+utils.strpprint(p.npprint(prev_ppr=[[" "], [" "], [" "], ["A"], [" "], [" "], [" "]])) + "\nWhere A = \n" + str(m) + "\n> "
    
    return s, p(m).det(), lambda x : int(x)

def tangent_line(inpt_dict):
    nranges = inpt_dict['nranges']
    
    function = utils.rand_func_iii(nranges=nranges[:], max_deg=2, n=3, fweights=[0, 0, 0, 1, 1, 4, 0, 2, 4])
    x = random.randint(nranges[0], nranges[1])
    line = utils.tangent_line(function, x)
    p = utils.poly.rand(random.randint(1, 3), coeff_range=nranges[:])
    roots = (p - line).roots()[:]
    power = random.randint(1, 4)
    res = abs(sum([i ** power for i in roots]))
    s = 'If the line tangent to f(x) at x = %d intercepts g(x) at x = a_1, a_2, ..., a_n \nfind |a_1^%d+a_2^%d+...+a_n^%d| where f(x)=\n'%(x, power, power, power)+str(function)+'\ng(x)=\n'+str(p)+"\n>"
    return s, res, lambda x : float(x)

def arithmetic_game(inpt_dict):
    numdigs = inpt_dict['ndig']
    n = inpt_dict['n']
    ndigits = inpt_dict['ndigits']
    sq = inpt_dict['sq']
    cmplx = inpt_dict['cmplx']
    
    a, b = utils.arithmetic_elems(numdigs, n, cmplx=cmplx, sq = sq)
    if not isinstance(b, complex):
        bres = int(b * 10**ndigits) / (10**ndigits)
    else:
        bres = complex(int(b.real * 10**ndigits) / (10**ndigits), int(b.imag * 10**ndigits) / (10**ndigits))
    return a + "\n > ", bres, lambda x : complex(x)

def calc_game(inpt_dict):
    i = 0
    while not i:
        try:
            a, b = problem_set_calc.single_number_gen()
            moe = inpt_dict['moe']
            i = 1
        except:
            pass
    return utils.strpprint(a) + '\n > ', b, lambda x : b if abs(evl.evl(x)- b) < abs(moe * b) else b + 1

def integral_set_game(inpt_dict):
    i = 0
    while not i:
        try:
            a, b = problem_set_integral.single_number_gen()
            moe = inpt_dict['moe']
            i = 1
        except:
            pass
    return utils.strpprint(a) + '\n > ', b, lambda x : b if abs(evl.evl(x)- b) < abs(moe * b) else b + 1

def fourier_set_game(inpt_dict):
    i = 0
    while not i:
        try:
            a, b = fourier_prob_set.single_number_gen()
            moe = inpt_dict['moe']
            i = 1
        except:
            pass
    return utils.strpprint(a) + '\n > ', b, lambda x : b if abs(evl.evl(x)- b) < abs(moe * b) else b + 1

def laplace_set_game(inpt_dict):
    i = 0
    while not i:
        try:
            a, b = laplace_problem_set.single_number_gen()
            moe = inpt_dict['moe']
            i = 1
        except:
            pass
    return utils.strpprint(a) + '\n > ', b, lambda x : b if abs(evl.evl(x)- b) < abs(moe * b) else b + 1

def lincont_game(inpt_dict):
    i = 0
    while not i:
        try:
            a, b = lincont_problem_set.single_number_gen()
            moe = inpt_dict['moe']
            i = 1
        except:
            pass
    if not callable(b):

        return utils.strpprint(a) + '\n > ', b, lambda x : b if abs(evl.evl(x)- b) <= abs(moe * b) else b + 3
    else:

        st = utils.strpprint(a) + '\n > '
        res = 'Not this !'
        def check_method(x):
            if b(x):
                return res
            return res + " THIS IS WRONG! "
        
        return st, res, check_method

def complex_mult_game(inpt_dict):
    ndigits = inpt_dict['ndigits']
    n1 = complex(random.randint(10**(ndigits-1), 10**(ndigits) - 1), random.randint(10**(ndigits-1), 10**(ndigits) - 1))
    n2 = complex(random.randint(10**(ndigits-1), 10**(ndigits) - 1), random.randint(10**(ndigits-1), 10**(ndigits) - 1))
    st = "%s%s = "%(str(n1), str(n2))
    res = n1 * n2
    check_method = lambda x : complex(x)
    return st, res, check_method

def trachtenberg(inpt_dict):
    digits = inpt_dict["ndigits"]
    n1 = random.randint(10 ** (digits - 1), 10 ** (digits) - 1) 
    n2 = random.randint(10 ** (digits - 1), 10 ** (digits) - 1) 
    n1s = ' '.join([i for i in str(n1)])
    n2s = ' '.join([i for i in str(n2)])
    string = "  %s\n* %s\n"%(n1s, n2s) + ''.join(['-' for i in range(max(len(n1s), len(n2s)) + 2)]) + '\n'
    return [string, str(n1 * n2)[::-1], lambda x : x]