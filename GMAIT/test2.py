import turtle
from utils import *
from math import *

t = turtle.Turtle()
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
def teleport(t, posx, posy):
    t.penup()
    t.goto(posx, posy)
    t.pendown()
def draw_arc(t, center, radius, theta0, theta1, n):
    teleport(t, center[0], center[1])
    points = []
    for i in range(n):
        theta = theta0 + (i / n) * (theta1 - theta0)
        points.append(add_vect(center[:], [radius * cos(theta), radius * sin(theta)]))
    teleport(t, points[0][0], points[0][1])
    for i in range(1, n):
        t.goto(points[i][0], points[i][1])

draw_arc(t, [10, 20], 40, 0, 3*math.pi/2, 100)
turtle.done()