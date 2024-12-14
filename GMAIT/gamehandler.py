import math
import random
import time
import utils
import evaluator as evl
import statistics as st

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
    mode = inpt_dict["mode"] if inpt_dict["mode"] != 4 else random.randint(1, 3)
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
        f, string = utils.generate_eulersub(deg, nranges=nranges[:])
        a, b = random.randint(boundranges[0], boundranges[1]), random.randint(boundranges[0], boundranges[1])
        res = round(utils.numericIntegration(f, min(a, b), max(a, b)), ndigits=ndigits)
        string2 = "Evaluate the integral of the function below from %d to %d\n"%(min(a, b), max(a, b)) + string + "\nI = "
        
        
    
    elif mode == 3:
        f, string = utils.generate_trig(nranges=nranges[:])
        a, b = random.randint(boundranges[0], boundranges[1]), random.randint(boundranges[0], boundranges[1])
        res = round(utils.numericIntegration(f, min(a, b), max(a, b)), ndigits=ndigits)
        string2 = "Evaluate the integral of the function below from %d to %d\n"%(min(a, b), max(a, b)) + string + "\nI = "
    
    return [string2, res, lambda x : integral_conv_func(x, ndigits)]

def regMulDig(inpt_dict):#(digits=5):
    digits = inpt_dict["ndigits"]
    n1 = random.randint(10 ** (digits - 1), 10 ** (digits) - 1) 
    n2 = random.randint(10 ** (digits - 1), 10 ** (digits) - 1) 
    string = "%d * %d = "%(n1, n2)
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
    s = string + "\nP = " + str(period) + "\n" + "a0 + a1 + b1 = "
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
    fin_fin_str = fin_str + "\nx y z ... = "
    return [fin_fin_str, answers, lambda x : [int(i) for i in x.split(" ")]]

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
    f, s, iv = utils.random_diff_eq_ord(order=order, nranges=nranges, n=random.randint(0, 1), max_deg=max_deg)
    z = round(random.random(), ndigits=2)
    nstr = "y(0) = " + str(iv[0]) + "\n" + "y'(0) = " + str(iv[1]) + "\n" + s + "\n" + "y(%f) = "%z 
    cond = lambda x : f(z) * 0.8 <round(evl.evl(x), ndigits=2)< f(z)*1.2
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
    string = str(m) + "\n" + "det = "
    return string, res, lambda x : float(x)

def eigenValue(inpt_dict):
    dims = inpt_dict["dim"]
    nrange = inpt_dict["nranges"]
    ndigits = inpt_dict["ndigits"]
    m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
    string = str(m) + "\n" + "L = "
    eigens = min(m.eigenvalue())
    return string, round(eigens, ndigits), lambda x : float(x)

def polyDet(inpt_dict):
    dims = inpt_dict["dim"]
    nrange = inpt_dict["nranges"]
    max_deg = inpt_dict["deg"]
    m = utils.matrix.randpoly(dims=[dims, dims], max_deg=max_deg, coeff_range=nrange[:])
    res = m.det()
    z = random.randint(nrange[0], nrange[1])
    string = str(m) + "\n" + "Evaluate the determinant at x = %d "%z
    return string, res(z), lambda x : float(x)


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
