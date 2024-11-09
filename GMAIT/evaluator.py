import math

def isbasic(string):
    for i in string:
        if (not i.isdigit() and i not in ["+", "-", "/", "*", "^", ".", "(", ")", "e"]):
            return False
    return True

def firstPr(string):
    k, j= 0, 0
    arr = []
    for i in string:
        if i == "(" and j == 0:k = 1
        if i == "(": j+=1
        if i == ")": j-=1
        if k == 1:
            arr.append(i)
        if j == 0 and k == 1:
            arr.pop(0);arr.pop(-1)
            return "".join(arr)
    return ""
def num(string):
    point = 0
    num_arr = [str(i) for i in range(0, 10)]
    num_arr.append("j")
    j = -1
    for i in string:
        j += 1
        if i not in num_arr:
            if i == ".": point += 1
            elif i == "-" and j != 0: return 0
            elif i == "-" : continue
            else: return 0
    if point > 1 : return 0
    return 1
def isspec(c):
    if 33<=ord(c)<=47 and ord(c) not in [40, 41]:
        return 1
    if ord(c) == 94:
        return 1
    if c == "_":
        return 1
    return 0
def isnumber(c):
    if 48<=ord(c)<=57:
        return 1
    return 0
def evalB(string):
    b = string.replace("^", "**")
    c = eval(b)
    if abs(c - round(c)) < 0.000000000001:
        return round(c)
    return c
def evalE(string, var, funcs):
    if isbasic(string):
        return evalB(string)
    else:
        n = string.find("asinh")
        if n != -1:
            pr = firstPr(string[n + 5:])
            expr = math.asinh(evalE(pr, var, funcs))
            nstr = string.replace("asinh("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("acosh")
        if n != -1:
            pr = firstPr(string[n + 5:])
            expr = math.acosh(evalE(pr, var, funcs))
            nstr = string.replace("acosh("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("atanh")
        if n != -1:
            pr = firstPr(string[n + 5:])
            expr = math.atanh(evalE(pr, var, funcs))
            nstr = string.replace("atanh("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("acoth")
        if n != -1:
            pr = firstPr(string[n + 5:])
            expr = math.atanh(1/evalE(pr, var, funcs))
            nstr = string.replace("acoth("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("sinh")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.sinh(evalE(pr, var, funcs))
            nstr = string.replace("sinh("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("cosh")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.cosh(evalE(pr, var, funcs))
            nstr = string.replace("cosh("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("tanh")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.tanh(evalE(pr, var, funcs))
            nstr = string.replace("tanh("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("coth")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = 1/math.tanh(evalE(pr, var, funcs))
            nstr = string.replace("coth("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("asin")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.asin(evalE(pr, var, funcs))
            nstr = string.replace("asin("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("acos")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.acos(evalE(pr, var, funcs))
            nstr = string.replace("acos("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("atan")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.atan(evalE(pr, var, funcs))
            nstr = string.replace("atan("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("acot")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.atan(1/evalE(pr, var))
            nstr = string.replace("acot("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("sin")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = math.sin(evalE(pr, var, funcs))
            nstr = string.replace("sin("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("cos")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = math.cos(evalE(pr, var, funcs))
            nstr = string.replace("cos("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("tan")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = math.tan(evalE(pr, var, funcs))
            nstr = string.replace("tan("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("cot")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = 1/math.tan(evalE(pr, var, funcs))
            nstr = string.replace("cot("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("log")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = math.log10(evalE(pr, var, funcs))
            nstr = string.replace("log("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("ln")
        if n != -1:
            pr = firstPr(string[n + 2:])
            expr = math.log(evalE(pr, var, funcs))
            nstr = string.replace("ln("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("sqrt")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.sqrt(evalE(pr, var, funcs))
            nstr = string.replace("sqrt("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("abs")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = abs(evalE(pr, var, funcs))
            nstr = string.replace("abs("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("exp")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = math.exp(evalE(pr, var, funcs))
            nstr = string.replace("exp("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("rad")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = evalE(pr, var, funcs) * math.pi / 180
            nstr = string.replace("rad("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("floor")
        if n != -1:
            pr = firstPr(string[n + 5:])
            expr = math.floor(evalE(pr, var, funcs)) 
            nstr = string.replace("floor("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("ceil")
        if n != -1:
            pr = firstPr(string[n + 4:])
            expr = math.ceil(evalE(pr, var, funcs)) 
            nstr = string.replace("ceil("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        n = string.find("rnd")
        if n != -1:
            pr = firstPr(string[n + 3:])
            expr = round(evalE(pr, var, funcs)) 
            nstr = string.replace("rnd("+pr+")", "("+str(expr)+")")
            return evalE(nstr, var, funcs)
        for name, variable, st in funcs:
            n = string.find(name)
            if n != -1:
                pr = firstPr(string[n + len(name):])
                expr = evalE(str(evalE(st, [(variable, "("+str(evalE(pr, var, funcs))+")")], funcs)), var, funcs)
                nstr = string.replace(name+"("+pr+")", "("+str(expr)+")")
                return evalE(nstr, var, funcs)

        cond = False
        for i, j in var:
            if i in string:
                cond = True
            string = string.replace(i, "("+str(evalE(str(j), var, funcs))+")")  
        
        if cond : return evalE(string, var, funcs)
        return 0
    
def ins_star(string):
    new_string = string[:]
    k = len(new_string) - 1
    i = 0
    while i < k:
        if isnumber(new_string[i]) + isnumber(new_string[i+1]) == 1:
            if new_string[i+1] != ")" and new_string[i] != "(" :
                if isspec(new_string[i]) + isspec(new_string[i+1]) == 0:
                    new_string = new_string[:i+1] + "*" + new_string[i+1:]
                    k+=1
        if new_string[i] == ")" and 97<=ord(new_string[i+1])<=122:
            new_string = new_string[:i+1] + "*" + new_string[i+1:]
            k+=1
        i += 1
    return new_string
def norm_pl(string, var, funcs):# mononominals_normed = [[pow, coef], [pow, coeff], ...]
    nstring = ins_star(string)
    mononomials = nstring.split("+")
    mononomials_normed = []
    for mn in mononomials:
        ml = evalE(mn, [(var, "1")], funcs)
        pow = round(math.log(evalE("("+mn+")/"+str(ml), [(var, str(math.e))], funcs)))
        mononomials_normed.append([pow, ml])
    return mononomials_normed

def evl(string):
    variables = [("pi", math.pi), ("e", math.e)]
    return evalE(ins_star(string), variables[:], [])
