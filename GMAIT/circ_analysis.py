
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


'''

from utils import *
import math
import random
import sys
import turtle

DEFAULT_SIZE = 15

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

def find_Yn_sourced(net_array, labelled=False):
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
                    
                    edges.append([i, j, net_array[i][j][k]])

        
        node_count += node_c
        
    nnodes = list(set(nodes))             
    A_arr = [[0 for j in range(edge_n)] for i in range(len(nnodes))]
    G_arr = [[0 for j in range(edge_n)] for i in range(edge_n)]
    i_s = []
    v_s = []
    counter = 0
    for i, j, adm in edges:
        A_arr[find(nnodes, i)][counter] = 1
        A_arr[find(nnodes, j)][counter] = -1
        G_arr[counter][counter] = adm[0] if isinstance(adm, (list, tuple)) else adm
        i_s.append(adm[-1] if isinstance(adm, (list, tuple)) else 0)
        v_s.append(adm[1] if isinstance(adm, (list, tuple)) else 0)
        counter += 1
    
    A = matrix(A_arr[:-1])
    G = matrix(G_arr)
    Yn = A * G * A.transpose()
    i_n = A * G * matrix([[i] for i in v_s]) - A * matrix([[i] for i in i_s])
    if not labelled:
        return Yn
    return Yn, i_n, nnodes[:]

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

def draw_line(tr, start, end):
    teleport(tr, start[0], start[1])
    tr.goto(end[0], end[1])

def write_text(tr, pos, text, size=DEFAULT_SIZE):
    teleport(tr, pos[0], pos[1])
    tr.write(text, move=False, align='center', font=('Arial', size, 'normal'))

def teleport(t, posx, posy):
    t.penup()
    t.goto(posx, posy)
    t.pendown()
    
def draw_arc(t, center, radius, theta0, theta1, n=100):
    teleport(t, center[0], center[1])
    points = []
    for i in range(n):
        theta = theta0 + (i / n) * (theta1 - theta0)
        points.append(add_vect(center[:], [radius * math.cos(theta), radius * math.sin(theta)]))
    teleport(t, points[0][0], points[0][1])
    for i in range(1, n):
        t.goto(points[i][0], points[i][1])

def draw_circle(t, center, radius):
    draw_arc(t, center[:], radius, 0, 2*math.pi)
    
def draw_resistor(tr, start, end, name,length=30):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = length / 6
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    draw_line(tr, start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    draw_line(tr, (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/3)[:], l)[:], curr_point[:])[:]
    draw_line(tr, curr_point[:], p1[:])
    curr_point[:] = p1[:]
    for i in range(5):
        p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/3 * (-1) ** (i+1))[:], 2*l)[:], curr_point[:])[:]
        draw_line(tr, curr_point[:], p1[:])
        curr_point[:] = p1[:]
    p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/3)[:], l)[:], curr_point[:])[:]
    draw_line(tr, curr_point[:], p1[:])
    curr_point[:] = p1[:]
    write_text(tr, scale_vect(add_vect(start, end), 0.5), name)
    return

def draw_capacitor(tr, start, end, name, length=10, height=10):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = height
    
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    draw_line(tr, start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    draw_line(tr, (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    p1 = add_vect(scale_vect(rotate_vect(unit_vect, math.pi/2)[:], l)[:], curr_point[:])[:]
    p2 = add_vect(scale_vect(rotate_vect(unit_vect, -math.pi/2)[:], l)[:], curr_point[:])[:]
    new_u_vect = scale_vect(unit_vect, 8)
    draw_line(tr, p1[:], p2[:])
    draw_line(tr, add_vect(p1[:], new_u_vect[:]), add_vect(p2[:], new_u_vect[:]))
    write_text(tr, scale_vect(add_vect(start, end), 0.5), name)
    return

def draw_inductor(tr, start, end, name, length=40):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = length / 4
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    draw_line(tr, start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    draw_line(tr, (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    start_angle = math.atan((end[1] - start[1]) / (end[0] - start[0])) if end[0] != start[0] else math.pi/2
    '''
    p_rect = add_vect(curr_point, scale_vect(rotate_vect(unit_vect[:], math.pi/2), l/2))
    for i in range(4):
        pg.draw.arc(disp, (0, 0, 0), (p_rect[0] - l, p_rect[1] - l, l, l), start_angle, start_angle + math.pi)
        curr_point = add_vect(curr_point[:], scale_vect(unit_vect[:], l))[:]
        p_rect = add_vect(curr_point[:], scale_vect(rotate_vect(unit_vect[:], math.pi/2), l/2))
    '''
    curr_point = add_vect(curr_point[:], scale_vect(unit_vect[:], l/2))[:]
    for i in range(4):
        draw_arc(tr, curr_point[:], l/2, start_angle, start_angle + math.pi)
        curr_point = add_vect(curr_point[:], scale_vect(unit_vect[:], l))[:]
    '''    
    font = pg.font.SysFont(None, 15)
    text_surface = font.render(name, True, (0, 0, 0))
    disp.blit(text_surface, scale_vect(add_vect(start, end), 0.5))
    pg.display.update()
    '''
    write_text(tr, add_vect(scale_vect(add_vect(start, end), 0.5), scale_vect(rotate_vect(unit_vect[:], math.pi/2), 2*l)), name)
    return


def draw_voltage_source(tr, start, end, name,length=30):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = length / 2
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    draw_line(tr, start[:], (start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist))
    draw_line(tr, (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist), end[:])
    curr_point =  [start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist]
    
    center = scale_vect(add_vect((start[0] + unit_vect[0] * ndist, start[1] + unit_vect[1] * ndist), (end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist)), 0.5)
    draw_circle(tr, center[:], l)
    
    draw_line(tr, add_vect(curr_point[:], scale_vect(unit_vect[:], l/3)), add_vect(curr_point[:], scale_vect(unit_vect[:], 0.8*l)))
    curr_point = add_vect(curr_point[:], scale_vect(unit_vect[:], 7 * l / 30 + l / 3))[:]
    draw_line(tr, add_vect(curr_point[:], scale_vect(rotate_vect(unit_vect[:], math.pi/2), 7*l/30)), add_vect(curr_point[:], scale_vect(rotate_vect(unit_vect[:], -math.pi/2), 7*l/30)))
    
    curr_point =  [end[0] - unit_vect[0] * ndist, end[1] - unit_vect[1] * ndist]
    draw_line(tr, add_vect(curr_point[:], scale_vect(unit_vect[:], -l/3)), add_vect(curr_point[:], scale_vect(unit_vect[:], -0.8*l)))
    
    write_text(tr, add_vect(scale_vect(add_vect(start, end), 0.5), scale_vect(rotate_vect(unit_vect[:], math.pi/2), 2*l)), name)
    return

def draw_current_source(tr, start, end, name,length=30):
    '''
    The length of the resistor is 30px.
    the angle is 60 degrees.
    '''
    l = length / 2
    dist = math.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
    unit_vect = [(end[0] - start[0]) / dist, (end[1] - start[1]) / dist]
    ndist = int((dist - length) / 2)
    
    curr_start = add_vect(start[:], scale_vect(rotate_vect(unit_vect[:], math.pi/2), 2*l))
    curr_end = add_vect(end[:], scale_vect(rotate_vect(unit_vect[:], math.pi/2), 2*l))
    draw_line(tr, start[:], curr_start[:])
    draw_line(tr, end[:], curr_end[:])
    draw_line(tr, curr_start[:], (curr_start[0] + unit_vect[0] * ndist, curr_start[1] + unit_vect[1] * ndist))
    draw_line(tr, (curr_end[0] - unit_vect[0] * ndist, curr_end[1] - unit_vect[1] * ndist), curr_end[:])
    curr_point =  [curr_start[0] + unit_vect[0] * ndist, curr_start[1] + unit_vect[1] * ndist]
    
    center = scale_vect(add_vect((curr_start[0] + unit_vect[0] * ndist, curr_start[1] + unit_vect[1] * ndist), (curr_end[0] - unit_vect[0] * ndist, curr_end[1] - unit_vect[1] * ndist)), 0.5)
    draw_circle(tr, center[:], l)
    
    draw_line(tr, add_vect(curr_point[:], scale_vect(unit_vect[:], l/3)), add_vect(curr_point[:], scale_vect(unit_vect[:], 5*l/3)))
    curr_point = add_vect(curr_point[:], scale_vect(unit_vect[:], 5*l/3))[:]
    
    draw_line(tr, curr_point[:], add_vect(curr_point[:], scale_vect(rotate_vect(unit_vect[:], 3*math.pi/4), 4*l/9)))
    draw_line(tr, curr_point[:], add_vect(curr_point[:], scale_vect(rotate_vect(unit_vect[:], -3*math.pi/4), 4*l/9)))
    
    write_text(tr, add_vect(scale_vect(add_vect(start, end), 0.5), scale_vect(rotate_vect(unit_vect[:], math.pi/2), 4*l)), name)
    return

def draw_branch(tr, start, end, elem_type, names):
    mid_point = start[:]
    if names[0] in ['0F', '0H', '0Ω']:
        if names[1] != 0:
            draw_voltage_source(tr, start[:], end[:], names[1])
        
        elif names[2] != 0:
            draw_current_source(tr, start[:], end[:], names[2])
        
        return
        
    if names[1] != '0V':
        mid_point = scale_vect(add_vect(start[:], end[:]), 0.5)
        draw_voltage_source(tr, start[:], mid_point[:], names[1])
        
    if elem_type == 'r':
        draw_resistor(tr, mid_point[:], end[:], names[0])
    elif elem_type == 'c':
        draw_capacitor(tr, mid_point[:], end[:], names[0])
    elif elem_type == 'i':
        draw_inductor(tr, mid_point[:], end[:], names[0])
    if names[2] != '0A':
        draw_current_source(tr, mid_point[:], end[:], names[2])
        

def draw_circuit(tr, net_array, nodes=[], dims=[600, 600]):
    '''
    currently supports only one edge between any two nodes.
    '''
    #display = pg.display.set_mode((600, 600))
   
    node_disp_coords_0 = [(150, 150), (300, 150), (450, 150), (150, 300), (300, 300), (450, 300), (150, 450), (300, 450), (450, 450)]
    node_disp_coords_1 = []
    for i, j in node_disp_coords_0:
        node_disp_coords_1.append([i * dims[0] / 600, j * dims[1] / 600])
    node_disp_coords = []
    for i, j in node_disp_coords_1:
        p = add_vect([i, -j], [-dims[0] / 2, dims[1] / 2])
        node_disp_coords.append(p[:])
    if nodes is None:
        for i in range(1, 9):
            
            write_text(tr, add_vect(node_disp_coords[i - 1], [0, -10]), str(i))
            draw_circle(tr, add_vect(add_vect(node_disp_coords[i - 1], [0, -10]), [0, 9]), 10)
            
        
        write_text(tr, add_vect(node_disp_coords[- 1], [0, -10]), '0')
        draw_circle(tr, add_vect(add_vect(node_disp_coords[- 1], [0, -10]), [0, 9]), 10)
        
    
    else:
        for i in range(len(nodes)):
            write_text(tr, add_vect(node_disp_coords[nodes[i]], [0, -10]), str(i + 1))
            draw_circle(tr, add_vect(add_vect(node_disp_coords[nodes[i]], [0, -10]), [0, 9]), 10)
            
        
        
    
    c = 0
    for i in range(len(net_array)):
        for j in range(i + 1):
            
            if len(net_array[i][j]) != 0:
                c += 1
                if isinstance(net_array[i][j][0], (tuple, list)):
                    elem, vs, js = net_array[i][j][0]
                    if isinstance(elem, poly):
                        if elem.deg == 1:
                            draw_resistor(tr, node_disp_coords[i], node_disp_coords[j], "R%d=%dΩ"%(c, 1/elem.coeffs[1]))
                        elif elem.deg == 2:
                            draw_capacitor(tr, node_disp_coords[i], node_disp_coords[j], "C%d=%dF"%(c, elem.coeffs[2]))
                        else:
                            draw_inductor(tr, node_disp_coords[i], node_disp_coords[j], "L%d=%dH"%(c, 1/elem.coeffs[0]))
                else:
                    elem = net_array[i][j][0]

                    if isinstance(elem, poly):
                        if elem.deg == 1:
                            draw_resistor(tr, node_disp_coords[i], node_disp_coords[j], "R%d=%dΩ"%(c, 1/elem.coeffs[1]))
                        elif elem.deg == 2:
                            
                            draw_capacitor(tr, node_disp_coords[i], node_disp_coords[j], "C%d=%dF"%(c, elem.coeffs[2]))
                        else:
                            draw_inductor(tr, node_disp_coords[i], node_disp_coords[j], "L%d=%dH"%(c, 1/elem.coeffs[0]))
    #pg.display.update()

def draw_circuit_sourced(tr, net_array, nodes=[], dims=[600, 600]):
    '''
    currently supports only one edge between any two nodes.
    '''
    #display = pg.display.set_mode((600, 600))
   
    node_disp_coords_0 = [(150, 150), (300, 150), (450, 150), (150, 300), (300, 300), (450, 300), (150, 450), (300, 450), (450, 450)]
    node_disp_coords_1 = []
    for i, j in node_disp_coords_0:
        node_disp_coords_1.append([i * dims[0] / 600, j * dims[1] / 600])
    node_disp_coords = []
    for i, j in node_disp_coords_1:
        p = add_vect([i, -j], [-dims[0] / 2, dims[1] / 2])
        node_disp_coords.append(p[:])
    if nodes is None:
        for i in range(1, 9):
            
            write_text(tr, add_vect(node_disp_coords[i - 1], [0, -10]), str(i))
            draw_circle(tr, add_vect(add_vect(node_disp_coords[i - 1], [0, -10]), [0, 9]), 10)
            
        
        write_text(tr, add_vect(node_disp_coords[- 1], [0, -10]), '0')
        draw_circle(tr, add_vect(add_vect(node_disp_coords[- 1], [0, -10]), [0, 9]), 10)
        
    
    else:
        for i in range(len(nodes)):
            write_text(tr, add_vect(node_disp_coords[nodes[i]], [0, -10]), str(i + 1))
            draw_circle(tr, add_vect(add_vect(node_disp_coords[nodes[i]], [0, -10]), [0, 9]), 10)
            
        
        
    
    
    for i in range(len(net_array)):
        for j in range(i + 1):
            
            if len(net_array[i][j]) != 0:
                
                if isinstance(net_array[i][j][0], (tuple, list)):
                    elem, vs, js = net_array[i][j][0]
                    if isinstance(elem, poly):
                        if elem.deg == 1:
                            draw_branch(tr, node_disp_coords[i], node_disp_coords[j], 'r', ["%dΩ"%(1/elem.coeffs[1]), '%dV'%vs, '%dA'%js])
                            #draw_resistor(tr, node_disp_coords[i], node_disp_coords[j], "%dΩ"%(1/elem.coeffs[1]))
                        elif elem.deg == 2:
                            draw_branch(tr, node_disp_coords[i], node_disp_coords[j], 'c', ["%dF"%(elem.coeffs[2]), '%dV'%vs, '%dA'%js])
                            #draw_capacitor(tr, node_disp_coords[i], node_disp_coords[j], "%dF"%(elem.coeffs[2]))
                        else:
                            draw_branch(tr, node_disp_coords[i], node_disp_coords[j], 'i', ["%dH"%(1/elem.coeffs[0]), '%dV'%vs, '%dA'%js])
                            #draw_inductor(tr, node_disp_coords[i], node_disp_coords[j], "%dH"%(1/elem.coeffs[0]))
                else:
                    elem = net_array[i][j][0]

                    if isinstance(elem, poly):
                        if elem.deg == 1:
                            draw_resistor(tr, node_disp_coords[i], node_disp_coords[j], "%dΩ"%(1/elem.coeffs[1]))
                        elif elem.deg == 2:
                            
                            draw_capacitor(tr, node_disp_coords[i], node_disp_coords[j], "%dF"%(elem.coeffs[2]))
                        else:
                            draw_inductor(tr, node_disp_coords[i], node_disp_coords[j], "%dH"%(1/elem.coeffs[0]))

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
        a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
        while a == b:
            a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
        z1, z2 = nodes[a], nodes[b]
        
        while len(net_array[z1][z2]):
            a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
            while a == b:
                a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
            z1, z2 = nodes[a], nodes[b]
        net_array[z1][z2] = [elem]
        net_array[z2][z1] = [elem]
    
    return net_array[:]

def generate_random_circuit_sourced(nodes_no, mesh_no, nranges, source_arr = [0, 0, 0, 0, 0, 1]):
    net_array = [[[] for j in range(9)] for i in range(9)]
    nodes = []
    for i in range(nodes_no):
        choice = random.randint(0, 8)
        while choice in nodes:
            choice = random.randint(0, 8)
        nodes.append(choice)
    
    for i in range(len(nodes)):
        elem = randelem(nranges[:])
        v = random.randint(nranges[0], nranges[1]) if random.choice(source_arr[:]) else 0
        i_k = random.randint(nranges[0], nranges[1]) if random.choice(source_arr[:]) else 0
        if i != len(nodes) - 1:
            net_array[nodes[i]][nodes[i+1]] = [[elem, v, i_k]]
            net_array[nodes[i+1]][nodes[i]] = [[elem, -v, -i_k]]
        else:
            net_array[nodes[i]][nodes[0]] = [[elem, v, i_k]]
            net_array[nodes[0]][nodes[i]] = [[elem, -v, -i_k]]
    
    for i in range(mesh_no - 1):
        elem = randelem(nranges[:])
        a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
        v = random.randint(nranges[0], nranges[1]) if random.choice(source_arr[:]) else 0
        i_k = random.randint(nranges[0], nranges[1]) if random.choice(source_arr[:]) else 0
        while a == b:
            a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
        z1, z2 = nodes[a], nodes[b]
        
        while len(net_array[z1][z2]):
            a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
            while a == b:
                a, b = random.randint(0, len(nodes) - 1), random.randint(0, len(nodes) - 1)
            z1, z2 = nodes[a], nodes[b]
        net_array[z1][z2] = [[elem, v, i_k]]
        net_array[z2][z1] = [[elem, -v, -i_k]]
    
    return net_array[:]

def yn_inv(yn):
    d = yn.det()
    n = len(yn.array[:])
    new_arr = [[det(minor(yn.array[:], [i, j])) * (-1)**(i + j) for i in range(n)] for j in range(n)]
    return matrix(new_arr[:]), d


def network_function(net_array, node):
    yn = find_Yn(net_array[:])

    vect = [[0] for i in range(len(yn.array[:]))]
    vect[node][0] = 1
    i, d = yn_inv(yn)

    solution = i * matrix(vect[:])
    
    return Div([solution.array[node][0], d])

def time_domain_net_function(net_array, node):
    freq_domain_n1, freq_domain_d = network_function(net_array[:], node).arr[:]
    freq_domain_n = freq_domain_n1 * poly([0, 1])
    #print(freq_domain_n)
    #print(freq_domain_d)
    #func =  network_function(net_array[:], node)
    lead_n_zeros = 0
    while freq_domain_n.coeffs[lead_n_zeros] == 0 and lead_n_zeros < len(freq_domain_n.coeffs[:]):
        lead_n_zeros += 1
    
    lead_d_zeros = 0
    while freq_domain_d.coeffs[lead_d_zeros] == 0 and lead_d_zeros < len(freq_domain_d.coeffs[:]):
        lead_d_zeros += 1
    
    k = min(lead_d_zeros, lead_n_zeros)
    freq_domain_n = poly(freq_domain_n.coeffs[k:])
    freq_domain_d = poly(freq_domain_d.coeffs[k:])

    return sym_inv_lap_rat(freq_domain_n, freq_domain_d)#lambda t: numeric_inverse_laplace_transform(func, t, order=0, dt=0.001)#lambda t: inv_laplace_tr_rat(freq_domain_n, freq_domain_d, t)#

def solve_voltages(net_array, node, labelled=True):
    
    yn, i_s, n = find_Yn_sourced(net_array[:], labelled=labelled)
    

    
    i, d = yn_inv(yn)

    solution = i * i_s
    
    return Div([solution.array[node][0], d])

def time_domain_voltages(net_array, node, labelled=True):
    freq_domain_n, freq_domain_d = solve_voltages(net_array[:], node, labelled=labelled).arr[:]
    #freq_domain_n = freq_domain_n1
    #print(freq_domain_n)
    #print(freq_domain_d)
    #func =  network_function(net_array[:], node)
    lead_n_zeros = 0
    while freq_domain_n.coeffs[lead_n_zeros] == 0 and lead_n_zeros < len(freq_domain_n.coeffs[:]):
        lead_n_zeros += 1
    
    lead_d_zeros = 0
    while freq_domain_d.coeffs[lead_d_zeros] == 0 and lead_d_zeros < len(freq_domain_d.coeffs[:]):
        lead_d_zeros += 1
    
    k = min(lead_d_zeros, lead_n_zeros)
    freq_domain_n = poly(freq_domain_n.coeffs[k:])
    freq_domain_d = poly(freq_domain_d.coeffs[k:])


    return sym_inv_lap_rat(freq_domain_n, freq_domain_d)#lambda t: numeric_inverse_laplace_transform(func, t, order=0, dt=0.001)#lambda t: inv_laplace_tr_rat(freq_domain_n, freq_domain_d, t)#




def generate_circuit_problem_unsourced(nranges, tranges, nnode, nmesh, draw=True, labelled=True):


            
    display = turtle.Screen()
    turtle.TurtleScreen._RUNNING=True
    
    tr = turtle.Turtle()
    turtle.clearscreen()
    
    tr.speed(0)
    turtle.tracer(0, 0)

    sc_size = [600, 600]#display.screensize() #display.screensize(600, 600, 'white')
    DEFAULT_SIZE = 15 * sc_size[0] / 600
    
    net = generate_random_circuit(nnode, nmesh, nranges[:])[:]

    time = random.randint(tranges[0], tranges[1])
    node = random.randint(0, nnode - 2)
    if labelled:
        sol, nodes = find_Yn(net[:], labelled=True)
    else:
        nodes=[]

    td_net_f = time_domain_net_function(net, node)
    z = td_net_f(time)
    if draw:
        draw_circuit(tr, net[:], nodes=nodes, dims=sc_size)
        display.title('Evaluate the net function for node %d at t=%f (node %d is earth)'%(node + 1, time, len(nodes)))
    #tr.color('white')   
    tr.hideturtle()
    #turtle.update()
    turtle.done()
    
    
    return time, z, td_net_f

def generate_circuit_problem(nranges, tranges, nnode, nmesh, source_arr=[0, 0, 0, 0, 0, 1], draw=True, labelled=True):


            
    display = turtle.Screen()
    turtle.TurtleScreen._RUNNING=True
    
    tr = turtle.Turtle()
    turtle.clearscreen()
    
    tr.speed(0)
    turtle.tracer(0, 0)

    sc_size = [600, 600]#display.screensize() #display.screensize(600, 600, 'white')
    DEFAULT_SIZE = 15 * sc_size[0] / 600
    
    net = generate_random_circuit_sourced(nnode, nmesh, nranges[:], source_arr=source_arr[:])[:]

    time = random.randint(tranges[0], tranges[1])
    node = random.randint(0, nnode - 2)
    if labelled:
        sol, js, nodes = find_Yn_sourced(net[:], labelled=True)
        
        cond = True
        for i in js.array:
            
            if sum([abs(j) for j in i[0].coeffs]):
                cond = False
        if cond:
            return  generate_circuit_problem(nranges, tranges, nnode, nmesh, source_arr=source_arr[:], draw=draw, labelled=labelled)

    else:
        nodes=[]

    td_net_f = time_domain_voltages(net[:], node)
    z = td_net_f(time)
    if draw:
        draw_circuit_sourced(tr, net[:], nodes=nodes, dims=sc_size)
        display.title('Evaluate the voltage of node %d at t=%f (node %d is earth)'%(node + 1, time, len(nodes)))
    #tr.color('white')   
    tr.hideturtle()
    #turtle.update()
    turtle.done()
    
    
    return time, z.real, td_net_f
