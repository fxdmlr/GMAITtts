import math
import random
import time
import utils

def regDetGame(number_of_rounds=5, dims=3, nrange=[10, 100]):
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
        res = m.det()
        print(str(m))
        n = int(input("det = "))
        
        if n == res:
            print("Correct.")
            pts += 1
        else:
            print("Incorrect. The answer was %d."%res)
    
    end = time.time()
    
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def regDetGameDyn(tot_time=600, dims=3, nrange=[10, 100]):
    start = time.time()
    pts = 0
    number_of_rounds = 0
    while time.time() - start <= tot_time:
        m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
        res = m.det()
        print(str(m))
        n = int(input("det = "))
        number_of_rounds += 1
        end = time.time()
        if time.time() - start > tot_time:
            print("Time Elapsed before entry.")
            return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds, tot_time]
        if n == res:
            print("Correct.")
            print("Remaining time : ", round(tot_time - (time.time() - start)))
            pts += 1
        else:
            print("Incorrect. The answer was %d."%res)
            print("Remaining time : ", round(tot_time - (time.time() - start)))
    
    end = time.time()
    
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def eigenvalueGame(number_of_rounds=5, dims=3, nrange=[10, 100], ndigits=2):
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
        print(m)
        eigens = min(m.eigenvalue())
        n = float(input("L = "))
        if n == round(eigens, ndigits):
            print("Correct.")
            pts += 1
        else:
            print("Incorrect. The answer was %f."%round(eigens, ndigits))
    
    end = time.time()
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def eigenvalueGameDyn(tot_time=600, dims=3, nrange=[10, 100], ndigits=2):
    start = time.time()
    pts = 0
    number_of_rounds = 0
    
    while time.time() - start <= tot_time:
        m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
        number_of_rounds += 1
        print(m)
        eigens = min(m.eigenvalue())
        n = float(input("L = "))
        end = time.time()
        if end - start > tot_time:
            print("Time elapsed before entry.")
            return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds, tot_time]
        if n == round(eigens, ndigits):
            print("Correct.")
            print("Remaining time : ", round(tot_time - (time.time() - start)))
            pts += 1
        else:
            print("Incorrect. The answer was %f."%round(eigens, ndigits))
            print("Remaining time : ", round(tot_time - (time.time() - start)))
    
    end = time.time()
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def polyDetGame(number_of_rounds=5, dims=3, nrange=[10, 100], max_deg=2, zrange=[1, 100]):
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        m = utils.matrix.randpoly(dims=[dims, dims], max_deg=max_deg, coeff_range=nrange[:])
        res = m.det()
        print(str(m))
        z = random.randint(nrange[0], nrange[1])
        n = int(input("Evaluate the determinant at x = %d "%z))
        
        if n == res(z):
            print("Correct.")
            pts += 1
        else:
            print("Incorrect. The answer was %d."%res(z))
    
    end = time.time()
    
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def polyDetGameDyn(duration=600, dims=3, nrange=[10, 100], max_deg=2, zrange=[1, 100]):
    start = time.time()
    pts = 0
    number_of_rounds = 0
    while time.time() - start <= duration:
        m = utils.matrix.randpoly(dims=[dims, dims], max_deg=max_deg, coeff_range=nrange[:])
        res = m.det()
        print(str(m))
        z = random.randint(nrange[0], nrange[1])
        n = int(input("Evaluate the determinant at x = %d "%z))
        number_of_rounds += 1
        end = time.time()
        if end - start > duration:
            print("Time elapsed before entry.")
            return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]
        if n == res(z):
            print("Correct.")
            pts += 1
        else:
            print("Incorrect. The answer was %d."%res(z))
    
    end = time.time()
    
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def regMulGame(number_of_rounds=5, dims=3, nrange=[10, 100]):
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        m = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
        p = utils.matrix.rand(dims=[dims, dims], nrange=nrange[:])
        q = m * p
        print(str(m))
        print("____________________")
        print(str(p))
        print("prod = \n")
        arr = []
        for i in range(dims):
            s = input().split(" ")
            arr.append([int(i) for i in s])
        
        mat = utils.matrix(arr)
        
        if mat == q:
            print("Correct.")
            pts += 1
        else:
            print("Incorrect. The answer was :")
            print(q)
    
    end = time.time()
    
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

def polyMulGame(number_of_rounds=5, dims=3, nrange=[10, 100], max_deg=2, zrange=[1, 100]):
    start = time.time()
    pts = 0
    for i in range(number_of_rounds):
        z = random.randint(zrange[0], zrange[1])
        m = utils.matrix.randpoly(dims=[dims, dims], max_deg=max_deg, coeff_range=nrange[:])
        p = utils.matrix.randpoly(dims=[dims, dims], max_deg=max_deg, coeff_range=nrange[:])
        q = (m * p)(z)
        print(str(m))
        print("____________________")
        print(str(p))
        print("Evaluate the product at x = %d"%z)
        arr = []
        for i in range(dims):
            s = input().split(" ")
            arr.append([int(i) for i in s])
        
        mat = utils.matrix(arr)
        
        if mat == q:
            print("Correct.")
            pts += 1
        else:
            print("Incorrect. The answer was :")
            print(q)
    
    end = time.time()
    
    return [pts / number_of_rounds * 100, end - start, (end - start) / number_of_rounds]

