from utils import *

def non_zero_arr_counter(arr):

    for i in range(len(arr)):
        if all([k == 0 for k in arr[i:]]):
            return i
    return len(arr)

def routh_table(p, array_inp=False):
    
    #returns the routh tabulation of p(x) in the form of a matrix.
    #if array_inp = False this means the input is a polynomial else a list
    coeffs = p.coeffs[:] if not array_inp else p[:]
    deg = p.deg if not array_inp else len(p) - 1
    ln0 = [coeffs[i] for i in range(len(coeffs) - 1, -1, -2)]
    ln1 = [coeffs[i] for i in range(len(coeffs) - 2, -1, -2)]
    if not deg % 2:
        ln1.append(0)
        
    lines = [ln0[:], ln1[:]]
    for i in range(deg - 1):
        prev_row = lines[-1]
        p_prev_row = lines[-2]
        prev_coeff = prev_row[0] #if prev_row[0] != 0 else 0.01
        new_row = []
        if not all([k == 0 for k in prev_row]):
            for j in range(len(lines[0]) - 1):
                b_k = -matrix([[p_prev_row[0], p_prev_row[j + 1]], [prev_row[0], prev_row[j+1]]]).det()
                
                c = prev_coeff
                if isinstance(b_k, (int, float, complex)) and isinstance(c, (int, float, complex)):
                    d = b_k / c
                elif isinstance(b_k, poly) and isinstance(c, (int, float, complex)):
                    d = b_k * (1/c)
                elif isinstance(b_k, Div) and isinstance(c, poly):
                    d = Div([b_k.arr[0], b_k.arr[1] * c]).simplify()
                elif isinstance(b_k, poly) and isinstance(c, Div):
                    d = Div([c.arr[1], b_k * c.arr[0]]).simplify()
                elif isinstance(b_k, Div) and isinstance(c, Div):
                    d = Div([b_k.arr[0] * c.arr[1], b_k.arr[1] * c.arr[0]]).simplify()
                else:
                    d = Div([b_k, c]).simplify()
                new_row.append(d)
        else:
            if non_zero_arr_counter(prev_row) != 1:
                new_coeff_arr = p_prev_row[:]
                x = poly([0, 1])
                z = deg - i + 1
                new_pol_arr = [(new_coeff_arr[k]) * x ** (z - 2*k - 1) for k in range(len(new_coeff_arr) - 1)]
                new_pol = sum(new_pol_arr)
                new_row_S = new_pol.diff().coeffs[::-1]
                if len(new_row_S) < len(lines[0]):
                    for k in range(len(lines[0]) - len(new_row_S)):
                        new_row_S.append(0)
                prev_coeff = new_row_S[0]
                lines[-1] = new_row_S[:]
                for j in range(len(lines[0]) - 1):
                    b_k = -matrix([[p_prev_row[0], p_prev_row[j + 1]], [new_row_S[0], new_row_S[j+1]]]).det()
                    if isinstance(b_k, (int, float, complex)) and isinstance(c, (int, float, complex)):
                        d = b_k / c
                    elif isinstance(b_k, poly) and isinstance(c, (int, float, complex)):
                        d = b_k * (1/c)
                    else:
                        d = Div([b_k, c])
                    new_row.append(d)
            else:
                return lines
        if new_row[0] == 0 and not all([k == 0 for k in new_row]):
            new_row[0] = 0.001
        lines.append(new_row[:] + [0])
    return lines
def type_ind(arr, type_obj):
    for i in range(len(arr)):
        if isinstance(arr[i], type_obj):
            return i
    return -1

def sgn_equiv_routh_table(p):
    lines = routh_table(p)
    new_lines = []
    for i in range(len(lines)):
        s = []
        for j in range(len(lines[i])):
            if isinstance(lines[i][j], Div):
                objects = lines[i][j].arr[:]
                ind = type_ind(objects, Div)
                while ind != -1:
                    pre_ind = objects[:ind]
                    post_ind = objects[ind+1:] if ind != len(objects) - 1 else []
                    new_ind = objects[ind].arr[:]
                    objects = pre_ind[:] + new_ind[:] + post_ind[:]
                    ind = type_ind(objects[:], Div)
                
                new_object = 1
                for k in objects:
                    new_object *= k
                
                s.append(new_object)
            else:
                s.append(lines[i][j])
        new_lines.append(s[:])
    
    return new_lines[:]
                
                    

def hurwitz_stable_num(p):
    '''
    checks if a polynomial with real number coefficients if hurwitz stable.
    '''
    coeffs = p.coeffs[:] if p.ndiff(p.deg)(0) >= 0 else (-p).coeffs[:]
    n = non_zero_arr_counter(coeffs[:])
    if 0 in coeffs[:n]:
        return False
    for i in coeffs[:n]:
        if i < 0:
            return False
        
    line = matrix(routh_table(p)[:]).transpose().array[0]
    cond = True
    for i in line:
        cond = cond and i * line[0] >= 0

    return cond

def hurwitz_stable_criterion(p):
    '''
    determines the criteria for the polynomial p(x) with number and poly 
    coeffs to be hurwitz stable.
    '''
    coeffs = p.coeffs[:]
    n = non_zero_arr_counter(coeffs[:])
    cond = False
    for i in coeffs[:n]:
        if not isinstance(i, poly):
            if i == 0:
                return False
            elif i < 0:
                cond = True
                break
    if cond:
        coeffs = (-p).coeffs[:n]
    else:
        coeffs = p.coeffs[:n]

def char_poly_state_eq(A):
    '''
    find the characateristic polynomial for the system of state equations :
    dx/dt = Ax + Bu where A and B are constant matrices.
    '''
    dim = len(A.array[:])
    I = matrix.ones(dim)
    x = poly([0, 1])
    p = (I*x - A).det()
    p.variable = 's'
    return p

def sstf(a, b, c, d):
    nm = matrix([[poly([-a.array[i][j], int(i == j)]) for j in range(len(a.array))] for i in range(len(a.array))])
    nm1, de = nm.adj(), nm.det()
    return rexp_poly((c * nm1 * b).array[0][0], de) + d

def tfss(p, q):
    A = [[int(i - 1 == j) for i in range(q.deg)] for j in range(q.deg - 1)]
    A.append((-q).coeffs[:-1])

    B = [[int(i == p.deg)] for i in range(1, p.deg + 1)]
    C = []
    sub = []
    for i in range(q.deg):
        if i < len(p.coeffs):
            a = p.coeffs[i]
        else:
            a = 0
        
        if i < len(q.coeffs):
           b = - q.coeffs[i] * p.coeffs[-1] 
        else:
            b = 0
        
        sub.append(a + b)
    
    C = [sub]
    return matrix(A), matrix(B), matrix(C)
