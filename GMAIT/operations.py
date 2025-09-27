from utils import *
import math
import random

#item in the operation array is an array of the following format:
# [<name>, <function>, [<input_type_1>, ...], [<output_type_1>, ...], [<random_item_generation_function_1>, ...]]
# input and output types are represented as follows : 
# number -> 0
# function -> 1
# array -> 2

def rand_num(a=0, b=10):
    return random.randint(a, b)

OPERATIONS = [
    ['derivative of $' , lambda objects : objects[0].diff(), [1], [1], [rand_func_iii]],
    ['intersection of $ and $' , lambda objects : solve_eq(objects[0], objects[1]), [1, 1], [0], [rand_func_iii, rand_func_iii]],
    ['line tangent to $ at x=$' , lambda objects : poly([objects[0](objects[1]) - objects[1] * objects[0].diff()(objects[1]), objects[0].diff()(objects[1])]), [1, 0], [1], [rand_func_iii, rand_num]],
    ['$ evaluated at $' , lambda objects : objects[0](objects[1]), [1, 0], [0], [rand_func_iii, rand_num]],
]

def generate_random_problem(n = 2):
    curr_string, current_func, inp_type, out_type, inp_arr = OPERATIONS[random.randint(0, len(OPERATIONS) - 1)]
    for i in range(n - 1):
        new_items = []
        curr_inps = inp_type[:]
        new_arr = []
        for j in OPERATIONS:
            new_arr = []
            for k in curr_inps:
                if j[3][0] == k:
                    new_arr.append(j)
            op = new_arr[random.randint(0, len(new_arr) - 1)]
            new_items.append(op)
        
        for j in range(len(new_items)):
            string, func, inp_t, out_t, inp_a = new_items[j]


def rand_problem(n, init_seed=OPERATIONS[random.randint(0, len(OPERATIONS) - 1)]):
    if n == 1:
        return init_seed[:]
    
    elif n > 1:
        #curr_string, current_func, inp_type, out_type, inp_arr = OPERATIONS[random.randint(0, len(OPERATIONS) - 1)]
        curr_string, curr_func, inp_type, out_type, inp_arr = init_seed[:]

        new_operations = []
        for j in OPERATIONS:
            new_arr = []
            for k in inp_type:
                if j[3][0] == k:
                    new_arr.append(j)
            if len(new_arr) > 1:
                new_operations.append(new_arr[random.randint(0, len(new_arr) - 1)])
            else:
                new_operations.append(new_arr[0])
        new_string = ""
        k = 0
        for i in range(len(curr_string)):
            if curr_string[i] != "$":
                new_string += curr_string[i]
            else:
                new_string += new_operations[k][0]
                k += 1
        
        new_cinp = []
        new_cout = []
        new_inp_arr = []
        for i in range(len(new_operations)):
            new_inp_arr += new_operations[i][-1]
        
        def new_func(objects):
            new_obj = []
            k = 0
            prev = 0
            for i in new_operations:
                s, f, it, ot, ia = i
                new_obj.append(f(objects[prev : prev + len(it)]))
            
            return curr_func(objects)
        for i in new_operations:
            s, f, it, ot, ia = i
            new_cinp += it[:]
            new_cout += ot[:]
            new_inp_arr += ia[:]
        
        return rand_problem(n - 1, init_seed=[new_string[:], new_func, new_cinp[:], new_cout[:], new_inp_arr[:]])
a = None
b = None
while True:
    if a is not None and b is not None:
        print(a)
        print(b.texify())
        
        break
    try:
        s, f, it, ot, ia = rand_problem(2)
        objs = [i() for i in ia]
        new_string = ""
        k = 0
        for i in range(len(s)):
            if s[i] != "$":
                new_string += s[i]
            else:
                new_string += objs[k].texify() if hasattr(objs[k], 'texify') else str(objs[k])
                k += 1

        a = new_string
        b = f(objs[:])
        
    except:
        pass