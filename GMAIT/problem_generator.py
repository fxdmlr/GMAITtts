import math 
import utils
import random

def make_question(string_array, solution_function, variable_ranges, float_l = False, ndigits = 1):
    '''
    this function makes a question with a given template and inserts
    random numbers as the variables. then proceeds to solve it with the 
    given solution function. the input of the solution function is one list
    containing the variables.
    
    the string array is akin to the following:
    
    --------------------
    string_array = ['find the distance of a ball thrown thrown from the origin with an initial velocity of v = ', 'i+', 'j at the time t = ', ' from the origin.(set g = 9.8)']
    def solution_function(inp_arr): # inp_arr = [x speed, y speed, t]
        return sqrt((inp_arr[0] * inp_arr[2])**2 + (-4.9*inp_arr[2]**2+inp_arr[0]*inp_arr[2])**2)
    
    --------------------
    
    a question generated with this function will yield somthing like : 
    
    'find the distance of a ball thrown thrown from the origin with an initial velocity of v = 2i+3j at the time t = 4 from the origin.(set g = 9.8)'
    and the solution will be : solution_function([2, 3, 4])
    
    
    
    
    '''
    variables = [random.randint(i, j) for i, j in variable_ranges]
    if float_l : 
        for i in range(len(variables)):
            err = int(random.random() * 10 ** ndigits) / (10**ndigits)
            variables[i] += err
    
    fin_str = ""
    for i in range(len(variables)):
        fin_str += string_array[i] + str(variables[i])
    
    fin_str += string_array[-1]
    res = solution_function(variables[:])
    
    return fin_str, res

