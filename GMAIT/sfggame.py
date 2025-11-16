from utils import *
import cmath
import random
import turtle

'''
signal flow graph here is defined by an array of array.
each element of the main array is a node. each node is 
represented by the list of the flows incoming from other 
nodes. 

ex:

sfg = [
    [[n1, g1], [n2, g2], ...], -> node 0 has one incoming connection from node with index n1 (on this list) with a gain of g1 and one from n2 with a gain of g2
    [[n3, g3], [n4, g4], ...], 
    ...
]
'''

DEFAULT_SIZE = 15
INF = 10000

    
def sfg_tf(array, input_node, output_node):
    eqns = []
    for node in range(len(array)):
        sub_arr = [0 for i in range(len(array))]
        for n, g in array[node]:
            sub_arr[n] = -g
        sub_arr[node] += 1        
        eqns.append(sub_arr[:])
    eq_mat = matrix([[j for j in i] for i in eqns[:]])
    #print(eq_mat)
    res_vect = matrix([[0] if i != input_node else [1] for i in range(len(array[:]))])
    neq_mat = matrix([[eqns[i][j] if j != output_node else int(i == input_node) for j in range(len(eqns[i]))]for i in range(len(eqns))]).det()
    #adj = eq_mat.adj() * res_vect
    sub = eq_mat.det()
    return neq_mat, sub#adj.array[output_node][0], sub



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

def draw_parabola(t, x1, x2, gain, peak, midline=0,  n=100):

    f = lambda x : (x - x1)*(x - x2)*peak * 4 / (x1-x2)**2 + midline
    sign = (x1 - x2) / (abs(x1 - x2))
    teleport(t, min(x1, x2), 0)
    points = []
    step = abs(x1 - x2) / (n + 1)
    for i in range(1, n + 1):
        points.append([i * step + min(x1, x2), f(i * step + min(x1, x2))])
    teleport(t, points[0][0], points[0][1])
    for i in range(1, n):
        t.goto(points[i][0], points[i][1])
    pp = [(x1 + x2) / 2, f((x1 + x2) / 2)]
    #teleport(t, pp[0], pp[1])
    draw_line(t, add_vect(pp[:], [-5 * sign, 5]), pp[:])
    draw_line(t, add_vect(pp[:], [-5 * sign, -5]), pp[:])
    write_text(t, add_vect(pp[:], [0, 3]), str(gain) if not hasattr(gain, 'npprint') else gain.specstr(var='s'), size=10)
    
           
def draw_sfg(array, input_node, output_node, tr, dims=[900, 500]):
    display = turtle.Screen()
    turtle.TurtleScreen._RUNNING=True
    tr.screen.setup(dims[0], dims[1])
    
    tr = turtle.Turtle()
    turtle.clearscreen()
    
    tr.speed(0)
    turtle.tracer(0, 0)

    sc_size = display.screensize(dims[0], dims[1], 'white')
    
    step = int(dims[0] / (len(array)+1))
    node_coords = [((i+1)*step - dims[0]/2, 0) for i in range(len(array))]
    for k in node_coords:
        draw_circle(tr, k, 5)
    peak = lambda n1, n2 : (n2 - n1) * dims[1] / (2 * len(array))
    for node in range(len(array)):
        for n1 in range(len(array[node])):
            n = array[node][n1][0]
            g = array[node][n1][1]
            if n - node != -1:
                draw_parabola(tr, node_coords[node][0], node_coords[n][0], g, peak(node, n))
            else:
                draw_line(tr, node_coords[node], node_coords[n])
                x1 = node_coords[node][0]
                x2 = node_coords[n][0]
                sign = (x1 - x2) / (abs(x1 - x2))
                pp = [(x1 + x2) / 2, 0]
                draw_line(tr, add_vect(pp[:], [-5 * sign, 5]), pp[:])
                draw_line(tr, add_vect(pp[:], [-5 * sign, -5]), pp[:])
                write_text(tr, add_vect(pp[:], [0, 3]), str(g) if not hasattr(g, 'npprint') else g.specstr(var='s'), size=10)
    write_text(tr, add_vect(node_coords[input_node], [0, 10]), 'In')
    write_text(tr, add_vect(node_coords[output_node], [0, 10]), 'Out')
    tr.hideturtle()
    #turtle.update()
    turtle.done()
    
def rand_sfg(number_of_nodes, ratio=0.2, nranges=[1, 10], mdeg=1):
    array = [[]]
    for i in range(1, number_of_nodes+2):
        type_seed = random.randint(1, 10000)
        if type_seed % 3 != 1:
            d = random.randint(1, mdeg)
            p1 = poly.rand(random.randint(0, d-1), coeff_range=nranges[:])
            p2 = poly.rand(d, coeff_range=nranges[:])
            gain = rexp_poly(p1, p2)
        else:
            gain = random.randint(nranges[0], nranges[1]) * (-1) ** random.randint(0, 1)
        if i - 1 == 0:
            array.append([[i-1, 1]])
            
        elif i - 1 == number_of_nodes:
            array.append([[i-1, 1]])
            
        else:
            array.append([[i-1, gain]])
    
    for i in range(1, len(array) - 1):
        for j in range(random.randint(0, int(len(array)*ratio))):
            type_seed = random.randint(1, 10000)
            if type_seed % 3 != 1:
                d = random.randint(1, mdeg)
                p1 = poly.rand(random.randint(0, d-1), coeff_range=nranges[:])
                p2 = poly.rand(d, coeff_range=nranges[:])
                gain = rexp_poly(p1, p2)

            else:
                gain = random.randint(nranges[0], nranges[1]) * (-1) ** random.randint(0, 1)
            new_node = random.randint(1, len(array) - 2)
            while new_node == i or new_node in [k[0] for k in array[i]]:
                new_node = random.randint(1, len(array) - 2)
            
            array[i].append([new_node, gain])
            
    
    return array
    
            
