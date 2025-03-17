import utils
import math
import random

def resistor(r):
    return utils.poly([0, 1/r, 0])

def capacitor(c):
    return utils.poly([0, 0, c])

def inductor(l):
    return utils.poly([1/l, 0, 0])

def solve_resistor_circuit(admittance_matrix, input_vector):
    return utils.solveLineq(admittance_matrix.array[:], input_vector.transpose().array[0])

def solve_circuit(mod_adm_matrix, mod_input_vector, init_cond):
    return utils.solve_ndeg_ode_sys_func(mod_adm_matrix, mod_input_vector, init_cond)

def input_vector(graph_matrix, y_b, voltage_input_vect, current_input_vect):
    return graph_matrix * y_b * voltage_input_vect - graph_matrix * current_input_vect

def admittance_matrix(graph_matrix, y_b):
    return graph_matrix * y_b * graph_matrix.transpose()

def modify_element_matrix(adm_matrix, input_vector): #proccessing y_b
    new_array = []
    new_input = []
    
    for eq in range(len(adm_matrix)):
        new_eq = []
        for ent in adm_matrix[eq]:
            ind = 0
            if isinstance(ent, utils.poly):
                if ent.coeffs[0] == 0:
                    ind = 1
                    break
        for ent in adm_matrix[eq]:
            j = ent
            if (not ent) and isinstance(j, utils.poly):
                if j.coeffs[0] == 0:
                    j = utils.poly(j.coeffs[1:])
                
            new_eq.append(j)
        new_array.append(new_eq)
        new_inp = input_vector[eq]
        if ind:
            new_inp = new_inp.diff() if hasattr(new_inp, 'diff') else 0
        
        new_input.append(new_inp)
    
    return new_array[:], new_input[:]

def polynomialize2d(array):
    new_arr = []
    for i in array:
        sub_arr = []
        for j in i:
            if isinstance(j, utils.poly):
                sub_arr.append(j)
                continue
            sub_arr.append(utils.poly([j]))
        new_arr.append(sub_arr)
    
    return new_arr

def polynomialize1d(array):
    new_arr = []
    for i in array:
        if isinstance(i, utils.poly):
            new_arr.append(i)
            continue
        new_arr.append(utils.poly([i]))
    
    return new_arr

'''
G1, G2, C3, L44, L45, L55 = 1, 1, 1, 1, 1, 1
r1 = resistor(1/G1)
r2 = resistor(1/G2)
c3 = capacitor(C3)
l44, l45, l55 = inductor(L44), inductor(L45), inductor(L55)

A = [[1, 0, 1, 0, 0],
     [0, 1, -1, 1, 0],
     [0, -1, 0, 0, 1]]

ap = polynomialize2d(A[:])

y_b = [[r1, 0, 0, 0, 0],
       [0, r2, 0, 0, 0],
       [0, 0, c3, 0, 0],
       [0, 0, 0, l44, l45],
       [0, 0, 0, l45, l55]]

y_b_p = polynomialize2d(y_b[:])

js1 = 10
jl4_0 = 0
j5_0 = 0

init_cond = [[1, 0],
             [0, 0], 
             [0, 0]]
init_cond_p = polynomialize2d(init_cond[:])
input_arr = [js1, 0, 0, jl4_0, j5_0]
input_arr_p = polynomialize1d(input_arr)

adm, inp = modify_element_matrix(y_b, input_arr)
adm_mat = utils.matrix(adm[:])
a_mat = utils.matrix(A)
print(admittance_matrix(a_mat, adm_mat))
e_vect = solve_circuit(admittance_matrix(a_mat, adm_mat).array[:], [inp[0]]+inp[3:], init_cond)
'''
def connect_points(array, p1, p2, order=0):
    if order == 0:
        upper = p1[:] if p1[0] < p2[0] else p2[:]
        lower = p2[:] if p1[0] <= p2[0] else p1[:]
        new_array = array[:]
        for i in range(abs(p1[0] - p2[0])):
            new_array[lower[0] - i][lower[1]] = "|"
        
        if upper[1] > lower[1]:
            for i in range(abs(p1[1] - p2[1])):
                new_array[upper[0]][lower[1]+i] = "-"
        else:
            for i in range(abs(p1[1] - p2[1])):
                new_array[upper[0]][lower[1]-i] = "-"
        
        return new_array[:]
    elif order == 1:
        new_array = array[:]
        rightmost = p1[:] if p1[1] < p2[1] else p2[:]
        leftmost = p2[:] if p1[1] <= p2[1] else p1[:]
        
        for i in range(abs(p1[1] - p2[1])):
            new_array[p1[0]][leftmost[1] - i] = "-"
        
        if rightmost[0] > leftmost[0]:
            for i in range(abs(p1[0] - p2[0])):
                new_array[leftmost[0] + i][rightmost[1]] = "|"
        else:
            for i in range(abs(p1[0] - p2[0])):
                new_array[leftmost[0] - i][rightmost[1]] = "|"
        
        return new_array[:]

def connect_points_straight(array, p1, p2):
    new_array = array[:]
    slope = (p2[0] - p1[0]) / (p2[1] - p1[1]) if p1[1] != p2[1] else None
    
    if slope is not None:
        h = p1[0] - p1[1] * slope
        x_step = (p2[1] - p1[1]) / abs((p2[1] - p1[1]))
        curr_x = p1[1]
        prev_x = p1[1]
        char_arr = []
        x = p1[1]
        y = p1[0]
        vert = []
        while (curr_x-p2[1]) * (prev_x - p2[1]) > 0 :
            x, y = curr_x, round(slope * curr_x + h)
            
            char_arr.append([y, x])
            curr_x += x_step
            prev_x = x
        x, y = int(curr_x), int(round(slope * curr_x + h))
        char_arr.append([y, x])
        c = "\\"
        if slope < 0:
            c = "/"
        if slope == 0:
            c = "-"
        for i, j in char_arr:
            new_array[int(i)][int(j)] = c
        
        for i in range(int(abs(x - p2[1]))):
                new_array[y][int(x+x_step)] = "-"
        
        return new_array

    else:
        upper = p1[:] if p1[0] < p2[0] else p2[:]
        lower = p2[:] if p1[0] <= p2[0] else p1[:]
        for i in range(abs(p1[0] - p2[0])):
            new_array[lower[0] - i][lower[1]] = "|"
        
        return new_array

          

def draw_circuit_graph(graph_matrix):
    z = utils.matrix(graph_matrix[:]).transpose().array[:]
    new_graph_matrix = graph_matrix[:] + [[-sum(i) for i in z]]
    z = utils.matrix(new_graph_matrix[:]).transpose().array[:]
    k = math.ceil(len(new_graph_matrix) / 3)
    grid = [[" " for j in range(13)] for i in range(6*k-5)]
    points = [[6*(i // 3 ), 6 * (i%3)] for i in range(len(new_graph_matrix))]
    new_arr = []
    
    for i in z:
        inds = []
        for j in range(len(i)):
            if i[j] != 0:
                inds.append(j)
        new_arr.append(set(inds))

    ncouples = []
    for i in new_arr:
        if i not in ncouples:
            ncouples.append(i)
    
    
    couples = []
    for i in ncouples:
        couples.append((points[list(i)[0]], points[list(i)[1]]))
    
    state = 0
    for i, j in couples:
        narr = connect_points_straight(grid[:], i, j)
        print("\n")
        grid = narr[:]
    
    return grid
