import os
import gamerunner as gr
import gamehandler as gh
import modified_main
import random
import utils

CALCULUS_PROBLEMS = [gh.subIntGame, 
                     gh.fourierSeries, 
                     gh.diffeq, 
                     gh.integral_cmplx,
                     gh.maclaurin_series,
                     gh.solvableInt,
                     gh.inv_laplace_game,
                     gh.inv_lap_mat,
                     gh.diffDet,
                     gh.tangent_line]

calc_dicts_defaults = [
    [{"nranges" : [1, 10], "boundary_ranges" : [1, 10], "deg" : 2, "ndigits" : 1, "mode" : 5}],#
    [{"nranges" : [1, 10], "period_ranges" : [1, 10], "deg" : 2, "moe" : 0.1, "exp_cond" : 1, "u_cond" : 0, "umvar_cond" : 0, "n_partite" : 1},
     {"nranges" : [1, 10], "period_ranges" : [1, 10], "deg" : 2, "moe" : 0.1, "exp_cond" : 0, "u_cond" : 0, "umvar_cond" : 0, "n_partite" : 1},
     {"nranges" : [1, 10], "period_ranges" : [1, 10], "deg" : 2, "moe" : 0.1, "exp_cond" : 1, "u_cond" : 0, "umvar_cond" : 0, "n_partite" : 2},
     {"nranges" : [1, 10], "period_ranges" : [1, 10], "deg" : 2, "moe" : 0.1, "exp_cond" : 0, "u_cond" : 0, "umvar_cond" : 0, "n_partite" : 2},
     {"nranges" : [1, 10], "period_ranges" : [1, 10], "deg" : 2, "moe" : 0.1, "exp_cond" : 1, "u_cond" : 0, "umvar_cond" : 0, "n_partite" : 3},
     {"nranges" : [1, 10], "period_ranges" : [1, 10], "deg" : 2, "moe" : 0.1, "exp_cond" : 0, "u_cond" : 0, "umvar_cond" : 0, "n_partite" : 3}],#
    [{"nranges" : [1, 10], "deg" : 2, "ord" : 2, 'inprange':[1, 2]},
     {"nranges" : [1, 10], "deg" : 2, "ord" : 3, 'inprange':[1, 2]},
     {"nranges" : [1, 10], "deg" : 2, "ord" : 4, 'inprange':[1, 2]}],
    [{"nranges" : [1, 10], "mdeg" : 2, "moe" : 0.1, "mode" : 0},
     {"nranges" : [1, 10], "mdeg" : 2, "moe" : 0.1, "mode" : 1}],#
    [{"nranges" : [1, 10], "mdeg" : 2, "moe" : 0.1}],
    [{"nranges" : [1, 10], "branges" : [-5, 5], "moe": 0.1, "type":0, "mdeg" : 3, "n_range":[1, 10], "m_range":[1, 10], "degi":3, "mdeg_c":2},
     {"nranges" : [1, 10], "branges" : [-5, 5], "moe": 0.1, "type":1, "mdeg" : 2, "n":3, "mdeg_c":2, "degi":3, "m_range":[1, 10], "mdeg_c":2},
     {"nranges" : [1, 10], "branges" : [-5, 5], "moe": 0.1, "type":2, "mdeg" : 5, "n_range":[1, 10], "degi":3, "m_range":[1, 10], "mdeg_c":2},
     {"nranges" : [1, 10], "branges" : [-5, 5], "moe": 0.1, "type":2, 'mdeg':2, 'n_range':[1, 10], "degi":3, "m_range":[1, 10], "mdeg_c":2}],###
    [{"nranges":[1, 10], "tranges":[1, 10], "mdeg":2, "moe":0.1, "diffint":1},
     {"nranges":[1, 10], "tranges":[1, 10], "mdeg":2, "moe":0.1, "diffint":0}],#
    [{"nranges" : [1, 10], "dim" : 3, "mdeg" : 2, "mdeg_rhs":1, "moe":0.1, "ndigits_t":1},
     {"nranges" : [1, 10], "dim" : 2, "mdeg" : 2, "mdeg_rhs":1, "moe":0.1, "ndigits_t":1}],
    [{"nranges" : [1, 10], "matdeg" : 1, 'mdeg' : 1, 'dim':3, 'inprange':[0, 10]},
     {"nranges" : [1, 10], "matdeg" : 1, 'mdeg' : 1, 'dim':2, 'inprange':[0, 10]}],
    [{'nranges':[1, 10]}] 
]

LINEAR_ALGEBRA_PROBLEMS = [
    gh.regDet,
    gh.polyDet,
    gh.matrixPoly,
    gh.eigenValue,
    gh.lineq,
    
]

lin_alg_defaults = [
    [{"nranges" : [1, 10000], "dim" : 3}],
    [{"nranges" : [1, 1000], "dim" : 3, "deg" : 2}],
    [{"mnranges" : [1, 100], 'deg' : 3, 'dim':3, 'pnranges':[1, 100]},
     {"mnranges" : [1, 100], 'deg' : 3, 'dim':2, 'pnranges':[1, 100]}],
    [{"nranges" : [1, 10000], "dim" : 3, "ndigits" : 1},
     {"nranges" : [1, 10000], "dim" : 2, "ndigits" : 1}],
    [{"nranges" : [1, 10000], "params" : 3, "param_ranges" : [1, 100]}]
   
]

NUMERICAL_ANALYSIS = [
    gh.numerical_analysis,
    gh.funcMatDet,
    gh.funcEval,
    
]
num_an_defaults = [
    [{"numranges" : [1, 10000], "ratranges":[1, 1000], "funranges":[0, 1], "ppartsnum":1, "varnumber":3, "purearith":0, "inpndigit":1, "resndigit":3}],
    [{"ndigits" : 1, "dig" : 3, "dim" : 3}],
    [{"ndigits" : 1, "dig" : 3, "N" : 1},
     {"ndigits" : 1, "dig" : 3, "N" : 2},
     {"ndigits" : 1, "dig" : 3, "N" : 3}],
    

]    

def run_battery(n, problem_set, input_set, ndigits=1):
    strings = []
    results = []
    functions = []
    options = []
    for i in range(n):
        ind = random.randint(0, len(problem_set) - 1)
        i2 = random.randint(0, len(input_set[ind]) - 1)
        s, r, f = problem_set[ind](input_set[ind][i2])
        strings.append("Question %d : \n"%(i+1) + s)
        
        functions.append(f)
        d = random.randint(2, 4)
        p = utils.poly([random.random()*(-1)**(round(random.random()))*(d-j) for j in range(d)])
        p.coeffs[0] = r
        arr = [p(j) for j in range(4)]
        new_arr = []
        t = 0
        while len(arr) > 0:
            j = random.randint(0, len(arr) - 1)
            q = arr[j]
            if isinstance(q, complex):
                q = complex(round(q.real, ndigits=ndigits), round(q.imag, ndigits=ndigits))
                if q.imag == 0:
                    q = q.real
            else:
                q = round(q, ndigits=ndigits)
            new_arr.append(q)
            if arr[j] == r:
                t = len(new_arr)
            arr.remove(arr[j])
        options.append(new_arr)
        results.append(t)
        
    return strings, results, options

def display_battery(n, problem_set, input_set, ndigits=1):
    strings, results, options = run_battery(n, problem_set, input_set, ndigits=ndigits)
    s = ""
    for i in range(len(strings)):
        s += strings[i]
        s += "\n"
        subs = ''.join(['%d)%s\n'%(j + 1, str(options[i][j])) for j in range(len(options[i]))])
        s += subs
        s += "\n"
    
    return s, results


PROBLEM_DATA = [
    [CALCULUS_PROBLEMS, calc_dicts_defaults, 1],
    [LINEAR_ALGEBRA_PROBLEMS, lin_alg_defaults, 0],
    [NUMERICAL_ANALYSIS, num_an_defaults, 3]
]

def main():
    while True:
        t = int(input("1-GENERATE CALCULUS PROBLEMS\n2-GENERATE LINEAR ALGEBRA PROBLEMS\n3-GENERATE NUMERICAL ANALYSIS PROBLEMS\n4-OLD INTERFACE\n"))
        fin_str = ""
        if t == 1:
            n = int(input('HOW MANY QUESTIONS ? '))
            z = 1
            while z:
                try:
                    s, r = display_battery(n, CALCULUS_PROBLEMS, calc_dicts_defaults, ndigits=1)
                    z = 0
                except:
                    z = 1
            fin_str = "\n------------QUESTIONS------------\n"+s + "\n---------------KEY---------------\n"+''.join(["%d)%d\n"%(j+1, r[j]) for j in range(len(r))])
            file = open('questions.txt', 'w')
            file.write(fin_str)
            file.close()
            
            print('PROBLEM SET IS AVAILABLE AT ./questions.txt .')
        if t == 2:
            n = int(input('HOW MANY QUESTIONS ? '))
            z = 1
            while z:
                try:
                    s, r = display_battery(n, LINEAR_ALGEBRA_PROBLEMS, lin_alg_defaults, ndigits=0)
                    z = 0
                except:
                    z = 1
            fin_str = "\n------------QUESTIONS------------\n"+s + "\n---------------KEY---------------\n"+''.join(["%d)%d\n"%(j+1, r[j]) for j in range(len(r))])        
            file = open('questions.txt', 'w')
            file.write(fin_str)
            file.close()
            
            print('PROBLEM SET IS AVAILABLE AT ./questions.txt .')
        
        if t == 3:
            n = int(input('HOW MANY QUESTIONS ? '))
            z = 1
            while z:
                try:
                    s, r = display_battery(n, NUMERICAL_ANALYSIS, num_an_defaults, ndigits=3)
                    z = 0
                except:
                    z = 1
            fin_str = "\n------------QUESTIONS------------\n"+s + "\n---------------KEY---------------\n"+''.join(["%d)%d\n"%(j+1, r[j]) for j in range(len(r))])
            
            file = open('questions.txt', 'w')
            file.write(fin_str)
            file.close()
            
            print('PROBLEM SET IS AVAILABLE AT ./questions.txt .')
        print(fin_str)
        if t == 4:
            modified_main.run()

if __name__ == "__main__":
    main()