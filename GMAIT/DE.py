from utils import *
import utils_integration as uint

def solve_complete_de(p, q):
    # solves p(x, y)dx + q(x, y)dy = 0
    # where the equation above is assumed to be complete.
    
    p_int = p.integrate(0)
    q_int = q.integrate(1)
    new_arr = polymvar.zeros(d = max(len(p.array[:]), len(q.array[:]))).array[:]
    for i in range(len(new_arr)):
        for j in range(len(new_arr)):
            if p_int.array[i][j][0] == q_int.array[i][j][0]:
                new_arr[i][j][0] = p_int.array[i][j][0]
            else:
                new_arr[i][j][0] = p_int.array[i][j][0] + q_int.array[i][j][0]
    return polymvar(new_arr[:])

def integrating_factor(p, q, w=0):
    #finds the integrating factor of p(x, y)dx + q(x, y)dy = 0
    if w == 0:
        a = (p.diff(1) - q.diff(0))([poly([0, 1]), 0])
        b = q([poly([0, 1]), 0])
        m = Comp([uint.integrate_ratexp(a, b), exp()])
        
    elif w == 1:
        a = (p.diff(1) - q.diff(0))([0, poly([0, 1])])
        b = -p(([0, poly([0, 1])]))
        m = Comp([polymvar.y(), uint.integrate_ratexp(a, b), exp()])
    
    return m

def solve_de_incomp(p, q, w=0):
    pass

x = polymvar.x()
y = polymvar.y()
p =  - x*x*y + 1
q = x*x*y-x*x*x
print(p)
print(q)
print(integrating_factor(p, q, w=0))