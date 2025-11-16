
'''

the default configuartion of the nodes is as follows:

    0       1       2
    
    
    3       4       5
    
    
    6       7       8
    
for now we only work with 9 nodes in the network.
net_array is an n by n matrix( for a network with n nodes) 
where net_array[i][j] is a list of all of the edges
between nodes i and j. ---> net_array[i][j] = [(y1, v1, j1), (y2, v2, j2), ...]
for the net_array[i][j], the set of edges are the same in the elemnts but the sources 
should be reversed ----> y_ij = y_ji, Is_ij = -Is_ji, Vs_ij = -Vs_ji

AND

net_array[i][i] == []

'''

from utils import *
import math
import random
import pygame as pg 
import sys


def find(array, key):
    for i in range(len(array)):
        if array[i] == key:
            return i
    
    return None

class Element:
    def __init__(self, admittance):
        self.y = admittance
    
    @staticmethod
    def capacitor(C):
        return poly([0, 0, C])
    
    @staticmethod
    def inductor(L):
        return poly([1/L])#Comp([poly([0, L]), inv()])
    
    @staticmethod
    def resistor(R):
        return poly([0, 1/R])

def convert_net_Yn(net_array, solve=False):
    '''
    Scanning net_array to find the dimentions of A.
    '''
    
    edge_n = 0
    edges = []
    v_s = []
    j_s = []
    for i in range(len(net_array)):
        for j in range(i + 1):
            if len(net_array[i][j]) != 0:
                edge_n += len(net_array[i][j])
                for k in range(len(net_array[i][j])):
                    if isinstance(net_array[i][j][k], (list, tuple)):
                        edges.append([i, j, net_array[i][j][k][0]])
                        v_s.append(net_array[i][j][k][1])
                        j_s.append(net_array[i][j][k][2])
                    
                    else:
                        edges.append([i, j, net_array[i][j][k]])
                        v_s.append(0)
                        j_s.append(0)
                 
    A_arr = [[0 for j in range(edge_n)] for i in range(len(net_array))]
    G_arr = [[0 for j in range(edge_n)] for i in range(edge_n)]
    
    counter = 0
    for i, j, adm in edges:
        A_arr[i][counter] = 1
        A_arr[j][counter] = -1
        G_arr[counter][counter] = adm
        counter += 1
    
    A = matrix(A_arr[:-1])
    G = matrix(G_arr)
    js = matrix([[i] for i in j_s])
    vs = matrix([[i] for i in v_s])

    i_s = A*G*vs - A*js
    Yn = A * G * A.transpose()
    sol_simp = None
    if solve:
        sol = Yn.inverse() * i_s
        narr = []
        for i in sol.array[:]:
            narr.append([i[0].simplify()])
        
        sol_simp = matrix(narr)
    
    return Yn, i_s, sol_simp

def find_Yn(net_array, labelled=False):
    '''
    Scanning net_array to find the dimentions of A.
    the resulting matrix is multiplied by s^2
    '''
    
    edge_n = 0
    edges = []
    node_count = 0
    node_c = 0
    nodes = []
    for i in range(len(net_array)):
        node_c = 0
        for j in range(i + 1):
            if len(net_array[i][j]) != 0:
                node_c = 1
                nodes.append(i)
                nodes.append(j)
                edge_n += len(net_array[i][j])
                for k in range(len(net_array[i][j])):
                    if isinstance(net_array[i][j][k], (list, tuple)):
                        edges.append([i, j, net_array[i][j][k][0]])

                    
                    else:
                        edges.append([i, j, net_array[i][j][k]])
        
        node_count += node_c
        
    nnodes = list(set(nodes))             
    A_arr = [[0 for j in range(edge_n)] for i in range(len(nnodes))]
    G_arr = [[0 for j in range(edge_n)] for i in range(edge_n)]
    
    counter = 0
    for i, j, adm in edges:
        A_arr[find(nnodes, i)][counter] = 1
        A_arr[find(nnodes, j)][counter] = -1
        G_arr[counter][counter] = adm
        counter += 1
    
    A = matrix(A_arr[:-1])
    G = matrix(G_arr)
    Yn = A * G * A.transpose()
    
    if not labelled:
        return Yn
    return Yn, nnodes[:]

def rotation_matrix(angle):
    return matrix([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

def rotate_vect(vect, angle):
    v = matrix([[vect[0]], [vect[1]]])
    res = rotation_matrix(angle) * v
    return [res.array[0][0], res.array[1][0]]

def scale_vect(vect, scale):
    return [vect[0] * scale, vect[1] * scale]

def add_vect(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1]]

def draw_resistor(disp, start, end, name,length=30):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = length / 6
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    pg.draw.line(disp, (0, 0, 0), start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    pg.draw.line(disp, (0, 0, 0), (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/3)[:], l)[:], curr_point[:])[:]
    pg.draw.line(disp, (0, 0, 0), curr_point[:], p1[:])
    curr_point[:] = p1[:]
    for i in range(5):
        p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/3 * (-1) ** (i+1))[:], 2*l)[:], curr_point[:])[:]
        pg.draw.line(disp, (0, 0, 0), curr_point[:], p1[:])
        curr_point[:] = p1[:]
    p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/3)[:], l)[:], curr_point[:])[:]
    pg.draw.line(disp, (0, 0, 0), curr_point[:], p1[:])
    curr_point[:] = p1[:]
    font = pg.font.SysFont(None, 15)
    text_surface = font.render(name, True, (0, 0, 0))
    disp.blit(text_surface, scale_vect(add_vect(start, end), 0.5))
    pg.display.update()
    return

def draw_capacitor(disp, start, end, name, length=10, height=20):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = height
    
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    pg.draw.line(disp, (0, 0, 0), start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    pg.draw.line(disp, (0, 0, 0), (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/2)[:], l)[:], curr_point[:])[:]
    p2 = add_vect(scale_vect(rotate_vect(unit_vect, -math.pi/2)[:], l)[:], curr_point[:])[:]
    new_u_vect = scale_vect(unit_vect, 8)
    pg.draw.line(disp, (0, 0, 0), p1[:], p2[:])
    pg.draw.line(disp, (0, 0, 0), add_vect(p1[:], new_u_vect[:]), add_vect(p2[:], new_u_vect[:]))
    font = pg.font.SysFont(None, 15)
    text_surface = font.render(name, True, (0, 0, 0))
    disp.blit(text_surface, scale_vect(add_vect(start, end), 0.5))
    pg.display.update()
    return

def draw_inductor(disp, start, end, name, length=40):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = length / 4
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    pg.draw.line(disp, (0, 0, 0), start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    pg.draw.line(disp, (0, 0, 0), (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    start_angle = -math.atan((end[1] - start[1]) / (end[0] - start[0])) if end[0] != start[0] else math.pi/2
    p_rect = add_vect(curr_point, scale_vect(rotate_vect(unit_vect[:], math.pi/2), l/2))
    for i in range(4):
        pg.draw.arc(disp, (0, 0, 0), (p_rect[0] - l, p_rect[1] - l, l, l), start_angle, start_angle + math.pi)
        curr_point = add_vect(curr_point[:], scale_vect(unit_vect[:], l))[:]
        p_rect = add_vect(curr_point[:], scale_vect(rotate_vect(unit_vect[:], math.pi/2), l/2))
    
    font = pg.font.SysFont(None, 15)
    text_surface = font.render(name, True, (0, 0, 0))
    disp.blit(text_surface, scale_vect(add_vect(start, end), 0.5))
    pg.display.update()
    return

def draw_circuit(display, net_array, nodes=[]):
    '''
    currently supports only one edge between any two nodes.
    '''
    #display = pg.display.set_mode((600, 600))
    font = pg.font.SysFont(None, 15)
    
    display.fill((255, 255, 255))
    node_disp_coords = [(150, 150), (300, 150), (450, 150), (150, 300), (300, 300), (450, 300), (150, 450), (300, 450), (450, 450)]
    if nodes is None:
        for i in range(1, 9):
            text_surface = font.render(str(i), True, (0, 0, 0))
            pg.draw.circle(display, (0, 0, 0), add_vect(add_vect(node_disp_coords[i - 1], [0, -10]), [3, 3]), 10, 1)
            display.blit(text_surface, add_vect(node_disp_coords[i - 1], [0, -10]))
        
        text_surface = font.render("0", True, (0, 0, 0))
        pg.draw.circle(display, (0, 0, 0), add_vect(add_vect(node_disp_coords[- 1], [0, -10]), [3, 3]), 10, 1)
        display.blit(text_surface, add_vect(node_disp_coords[- 1], [0, -10]))
    
    else:
        for i in range(len(nodes)):
            text_surface = font.render(str(i + 1), True, (0, 0, 0))
            pg.draw.circle(display, (0, 0, 0), add_vect(add_vect(node_disp_coords[nodes[i]], [0, -10]), [3, 3]), 10, 1)
            display.blit(text_surface, add_vect(node_disp_coords[nodes[i]], [0, -10]))
        
        
    
    c = 0
    for i in range(len(net_array)):
        for j in range(i + 1):
            
            if len(net_array[i][j]) != 0:
                c += 1
                if isinstance(net_array[i][j][0], (tuple, list)):
                    elem, vs, js = net_array[i][j][0]
                    if isinstance(elem, poly):
                        if elem.deg == 1:
                            draw_resistor(display, node_disp_coords[i], node_disp_coords[j], "R%d=%dΩ"%(c, 1/elem.coeffs[1]))
                        elif elem.deg == 2:
                            draw_capacitor(display, node_disp_coords[i], node_disp_coords[j], "C%d=%dF"%(c, elem.coeffs[2]))
                        else:
                            draw_inductor(display, node_disp_coords[i], node_disp_coords[j], "L%d=%dH"%(c, 1/elem.coeffs[0]))
                else:
                    elem = net_array[i][j][0]

                    if isinstance(elem, poly):
                        if elem.deg == 1:
                            draw_resistor(display, node_disp_coords[i], node_disp_coords[j], "R%d=%dΩ"%(c, 1/elem.coeffs[1]))
                        elif elem.deg == 2:
                            
                            draw_capacitor(display, node_disp_coords[i], node_disp_coords[j], "C%d=%dF"%(c, elem.coeffs[2]))
                        else:
                            draw_inductor(display, node_disp_coords[i], node_disp_coords[j], "L%d=%dH"%(c, 1/elem.coeffs[0]))
    #pg.display.update()

'''
net_array = [[[] for j in range(9)] for i in range(9)]
net_array[0][1], net_array[1][0] = [Element.resistor(1)], [Element.resistor(1)]
net_array[2][1], net_array[1][2] = [Element.inductor(1)], [Element.inductor(1)]
net_array[0][7], net_array[7][0] = [Element.capacitor(1)], [Element.capacitor(1)]
net_array[1][7], net_array[7][1] = [Element.inductor(1)], [Element.inductor(1)]
net_array[7][2], net_array[2][7] = [Element.inductor(1)], [Element.inductor(1)]
'''
def randelem(nranges):
    seed = random.randint(1, 3)
    val = random.randint(nranges[0], nranges[1])
    if seed == 1:
        return Element.resistor(val)
    elif seed == 2:
        return Element.capacitor(val)
    elif seed == 3:
        return Element.inductor(val)
    
def generate_random_circuit(nodes_no, mesh_no, nranges):
    net_array = [[[] for j in range(9)] for i in range(9)]
    nodes = []
    for i in range(nodes_no):
        choice = random.randint(0, 8)
        while choice in nodes:
            choice = random.randint(0, 8)
        nodes.append(choice)
    
    for i in range(len(nodes)):
        elem = randelem(nranges[:])
        if i != len(nodes) - 1:
            net_array[nodes[i]][nodes[i+1]] = [elem]
            net_array[nodes[i+1]][nodes[i]] = [elem]
        else:
            net_array[nodes[i]][nodes[0]] = [elem]
            net_array[nodes[0]][nodes[i]] = [elem]
    
    for i in range(mesh_no - 1):
        elem = randelem(nranges[:])
        z1, z2 = nodes[random.randint(0, len(nodes) - 1)], nodes[random.randint(0, len(nodes) - 1)]
        while len(net_array[z1][z2]):
            z1, z2 = nodes[random.randint(0, len(nodes) - 1)], nodes[random.randint(0, len(nodes) - 1)]
        net_array[z1][z2] = [elem]
        net_array[z2][z1] = [elem]
    
    return net_array[:]

def network_function(net_array, node):
    yn = find_Yn(net_array[:])
    vect = [[0] for i in range(len(yn.array[:]))]
    vect[node][0] = 1
    solution = yn.inverse() * matrix(vect[:])
    return solution.array[node][0].simplify() if hasattr(solution.array[node][0], 'simplify') else solution.array[node][0]

def time_domain_net_function(net_array, node):
    #freq_domain_n1, freq_domain_d = network_function(net_array[:], node).arr[:]
    #freq_domain_n = freq_domain_n1 * poly([0, 0, 1])

    func =  network_function(net_array[:], node)

    return lambda t: numeric_inverse_laplace_transform(func, t, order=0, dt=0.001)#lambda t: inv_laplace_tr_rat(freq_domain_n, freq_domain_d, t)#

'''
arr = generate_random_circuit(4, 1, [1, 10])
draw_circuit(arr[:])                    
y = find_Yn(arr)
print(y)
print(network_function(arr[:], 1))


running = True
while running:
    # 4a) Handle events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
pg.quit()

sys.exit()
'''

def generate_circuit_problem(nranges, tndigits, nnode, nmesh, draw=True, labelled=True):

    for i in range(10):
        
        try:
            pg.init()
            
            display = pg.display.set_mode((600, 600))
            display.fill((255, 255, 255))
            net = generate_random_circuit(nnode, nmesh, nranges[:])[:]

            time = round(random.random(), ndigits=tndigits)
            node = random.randint(0, nnode - 2)
            if labelled:
                sol, nodes = find_Yn(net[:], labelled=True)
            else:
                nodes=[]

            td_net_f = time_domain_net_function(net, node)
            z = td_net_f(time)
            if draw:
                draw_circuit(display, net[:], nodes=nodes)
                pg.display.set_caption('Evaluate the net function for node %d at t=%f (node %d is earth)'%(node + 1, time, len(nodes)))
                pg.display.update()
                running = True
                while running:
                    for event in pg.event.get():
                        if event.type == pg.QUIT:
                            running = False
                        
                        if event.type == pg.KEYDOWN:
                            if event.key == pg.K_q:
                                running = False
                pg.quit()
            
            return time, z
        except Exception as x:
            pass
            #pg.quit()
    

#print(generate_circuit_problem([1, 10], 1, 7, 4, draw=True, labelled=True))
