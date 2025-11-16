from utils import *
import math

def solve_second_order_const(arr):
    '''
    arr = [a, b, c, d, e, f, g]
    auxx + buxy + cuyy + dux + euy + fu + g = 0
    '''
    a, b, c, d, e, f, g = arr[:]
    delta = b ** 2 - 4*a*c
    
    '''
    the new equation will be [a'u_xixi + b'uxieta, ...]
    '''
    if delta > 0:
        pr, qr = (b + math.sqrt(delta)) / (2*a), (b - math.sqrt(delta)) / (2*a)
        print(pr, qr)
        # xi = y - pr * x
        # eta = y - qr * x
        
        ux = poly([0, 0, -qr, -pr])
        uy = poly([0, 0, 1, 1])
        uyy = poly([0, 0, 0, 0, 1, 2, 1])
        uxy = poly([0, 0, 0, 0, -qr, -pr-qr, -pr])
        uxx = poly([0, 0, 0, 0, qr**2, 2*qr*pr, pr**2])
        new = (d*ux + e*uy + c*uyy + b*uxy + a*uxx + poly([g, f])).coeffs[:]
        new.reverse()
        
        # removing remaining single derivatives:
        return new[:]


        