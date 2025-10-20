from utils import *
import problemset_utils as putil
import random
import cmath

def npprify(string):
    new_arr = [[], [], [], [], [], [], []]
    for i in string:
        new_arr = connect(new_arr[:], [[" "], [" "], [" "], [i], [" "], [" "], [" "]])
    return new_arr[:]

    
     
'''
0 -> real number
1 -> polynomial
2 -> point
3 -> function
4 -> matrix
5 -> question_method
'''



inpt_dict = {
    'ndigits' : 2
}
    


number_generators = [
    ["$$ = ", lambda objects : objects[0] * objects[1], [putil.rand_complex(inpt_dict['ndigits']), putil.rand_complex(inpt_dict['ndigits'])]],
    ['||$|| = ', lambda objects : abs(objects[0]), [putil.rand_complex(inpt_dict['ndigits'])]], 
    ['∠$ = ', lambda objects : cmath.atan(objects[0].imag / objects[0].real), [putil.rand_complex(inpt_dict['ndigits'])]]

]




def single_number_gen():
    z = random.randint(0, len(number_generators) - 1)
    string, f, rand = number_generators[z]
    new_string = [[], [], [], [], [], [], []]
    k = 0
    inp_arr = []
    prev_ppr = []
    for j in string:
        if j != '$':
            new_string = connect(new_string[:], npprify(j)[:])
        else:
            m = rand[k]()
            inp_arr.append(m)
            #if z == 0:
            #    prev_ppr=npprify('D')
            #if z in [1, 2]:
            #    prev_ppr=npprify('s')
            
            nns =  m.npprint()[:] if hasattr(m, 'npprint') else npprify(str(m))[:]
            new_string = connect(new_string, nns[:])[:]
            k += 1
    
    return new_string, f(inp_arr[:])

a, b = single_number_gen()
print(strpprint(a))
print(b)