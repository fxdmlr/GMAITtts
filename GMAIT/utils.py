import math
import random
import cmath

DEFAULT_TAYLOR_N = 1000

def minor(array, pos):
    new_arr = []
    for i in range(len(array)):
        col = []
        if i == pos[0]:
            continue

        for j in range(len(array[i])):
            if j != pos[1]:
                col.append(array[i][j])

        new_arr.append(col)

    return new_arr

def dot(arr1, arr2):
    s = 0
    for i in range(len(arr1)):
        s += arr1[i] * arr2[i]
    
    return s

def sdot(arr1, arr2):
    s = arr1[0] * arr2[0]
    for i in range(1, len(arr1)):
        s += arr1[i] * arr2[i]
    
    return s

def sgn(x):
    return x >= 0 if not callable(x) else x() >= 0

def gcd(a, b):
    for i in range(min(a, b), 0, -1):
        if a%i == 0 and b%i == 0:
            return i
    return 1

def strpprint(pp):
    new_q = ["".join(i) for i in pp]
    return "\n".join(new_q)

def matrixpprint(pp):
    arr = [strpprint(i) for i in pp]
    return "\n\n".join(arr) 

def connect(arr1, arr2):
    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i] + arr2[i])
    
    return arr3[:]

def cross(v, u):
    i = det([[1, 0, 0], v[:], u[:]])
    j = det([[0, 1, 0], v[:], u[:]])
    k = det([[0, 0, 1], v[:], u[:]])
    return [i, j, k]

def det(array):
    if len(array) == 1:
        return array[0][0]

    else:
        a = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                if isinstance(array[i][j], rational):
                    array[i][j] = float(array[i][j]) 
        for i in range(len(array)):
            a.append(array[0][i] * det(minor(array, [0, i])) * (-1)**(i))
        
        z = a[0]
        for i in range(1, len(a)):
            z+=a[i]
        return z

def numericIntegration(function, c, d, dx=0.0001):
    s = 0
    a = min(c, d)
    b = max(c, d)
    i = a
    while i <= b:
        s += (function(i) + function(i+dx))*dx/2
        i += dx
    return s * sgn(d - c)

def numericDiff(function, x, dx=0.0001):
    return (function(x+dx) - function(x-dx))* (1/(2*dx))

def simpInt(function, c, d, h=0.001):
    i = c + h
    s = 0
    n = 1
    while i <= d:
        s += (2 ** (n % 2 + 1)) * function(i)
        n += 1
        i += h
    
    s += function(c) + function(d)
    return (h / 3) * s

def reallineNIntegrate(function, h=0.0001):
    return simpInt(lambda x : function(x/(1-x**2) * (1+x**2))/((1-x**2)**2) if x!=1 else 0, -1, 1, h=h)

def semiNIntegrate(function, lowerbound, h=0.0001):
    return simpInt(lambda x : function(lowerbound - 1 + 1/(1-x)) / ((1-x)**2) if x != 1 else 0, 0, 1, h=h)

class integer:
    def __init__(self, n):
        self.n = n
    
    def prime(self):
        return integer.isprime(self.n)
    
    def factorize(self):
        return integer.factorization(self.n)
    
    def __add__(self, other):
        return integer(self.n + other.n)
    
    def __mul__(self, other):
        return integer(self.n * other.n)
    
    def __div__(self, other):
        return rational([self.n, other.n])
    
    def __sub__(self, other):
        return integer(self.n - other.n)
    
    def __neg__(self):
        return integer(-self.n)
    
    def __str__(self):
        return str(self.n)
    
    def __int__(self):
        return int(self.n)
    
    def __divmod__(self, other):
        return integer(self.n % int(other))
    
    def __gt__(self, other):
        if isinstance(other, (int, float, rational)):
            return self.n > float(other)
        return self.n > other.n 
    
    def __lt__(self, other):
        if isinstance(other, (int, float, rational)):
            return self.n < float(other)
        return self.n < other.n 
    
    def __ge__(self, other):
        if isinstance(other, (int, float, rational)):
            return self.n >= float(other)
        return self.n >= other.n
    
    def __le__(self, other):
        if isinstance(other, (int, float, rational)):
            return self.n <= float(other)
        return self.n <= other.n
    
    def __eq__(self, other):
        if isinstance(other, (int, float, rational)):
            return self.n == float(other)
        return self.n == other.n
         
    @staticmethod
    def isprime(n):
        if abs(n) == 1 or n == 0:
            return False
        
        for i in range(2, int(abs(n) ** 0.5) + 1):
            if n % i == 0:
                return False
        
        return True

    @staticmethod
    def factorization(n):
        n2 = abs(n)
        if integer.isprime(n2):
            return [[n2, 1]] if n > 0 else [[-1, 1], [n2, 1]]
        if n2 == 0:
            return [[0, 1]]
        arr = []
        if n < 0:
            arr.append([-1, 1])
        for i in range(2, n2):
            k = 0
            while integer.isprime(i) and n2 % i == 0:
                if k == 0:
                    arr.append([i, 1])
                else:
                    arr[-1][-1] += 1
                n2 /= i
                k += 1
            
        return arr
    __rmul__ = __mul__
    __radd__ = __add__
    @staticmethod
    def gcd(n, m):
        for i in range(min(n, m), 0, -1):
            if n % i == 0 and m % i == 0:
                return i
    
    @staticmethod
    def lcd(n, m):
        return int(n * m / integer.gcd(n, m))
    
    @staticmethod
    def rand(nrange=[1, 10]):
        return random.randint(nrange[0], nrange[1])
        
class rational:
    def __init__(self, num):
        #num = [p, q] -> number = p / q
        self.num = num[:]
    
    def __str__(self):
        return "%s / %s" % (str(self.num[0]), str(self.num[1]))
    
    def __call__(self):
        return self.num[0] / self.num[1]
    
    def pprint(self):
        str1 = str(self.num[0])
        str2 = str(self.num[1])
        mlen = max(len(str1), len(str2))
        if str2 != "1":
            a = "".join([" "for i in range((mlen - len(str1))//2)])
            b = a+"".join([" " for i in range((mlen - len(str1))%2)])
            str12 = a + str1 + b
            c = "".join([" "for i in range((mlen - len(str2))//2)])
            d = a+"".join([" " for i in range((mlen - len(str2))%2)])
            str22 = c + str2 + d
            lines = [[str12], ["".join(["-" for i in range(mlen)])], [str22]]
        else:
            lines = [[" " for i in range(len(str1))], [str1], [" " for i in range(len(str1))]]
        
        return lines
        
    
    def simplify(self):
        '''
        p = integer(self.num[0]) if isinstance(self.num[0], int) else self.num[0]
        q = integer(self.num[1]) if isinstance(self.num[1], int) else self.num[1]
        p_fac = dict(p.factorize())
        q_fac = dict(q.factorize())
        
        nfrac = [[i, j - q_fac[i] if i in q_fac.keys() else j] for i, j in p_fac.items()]
        n2frac = []
        for i, j in q_fac.items():
            if i not in p_fac.keys():
                n2frac.append([i, -j])
        
        nfrac += n2frac[:]
        np = 1
        nq = 1
        for i, j in nfrac:
            if j >= 0 :
                np *= i ** j
            
            else:
                nq *= i ** (-j)
        '''
        x = math.gcd(self.num[0], self.num[1])
        return rational([int(self.num[0] / x), int(self.num[1] / x)])
    
    def inv(self):
        return rational([self.num[1], self.num[0]])
    
    def __add__(self, other):
        if isinstance(other, float):
            return self * rational.convRat(other, 2)
        if isinstance(other, (int, integer)):
            n2 = rational([int(other), 1])
        else:
            n2 = other
        
        return rational([self.num[0] * n2.num[1] + n2.num[0] * self.num[1], self.num[1] * n2.num[1]])
    
    def __neg__(self):
        return rational([-self.num[0], self.num[1]])
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        if isinstance(other, (int, integer, float)):
            if isinstance(other, float):
                return self * rational.convRat(other, 2)
            if self.num[0] * int(other) % self.num[1] == 0:
                return self.num[0] * int(other) / self.num[1]
            return rational([self.num[0] * int(other), self.num[1]])
        return rational([self.num[0] * other.num[0], self.num[1] * other.num[1]])
    
    def __truediv__(self, other):
        if isinstance(other, (int, integer)):
            return rational([self.num[0], self.num[1] * int(other)])
        
        elif isinstance(other, rational):
            return self * other.inv()
        
    def __abs__(self):
        return self if self() >= 0 else -self
    
    def __float__(self):
        return self.num[0] / self.num[1]
    
    def __eq__(self, other):
        return float(self) == float(other)
    __rmul__ = __mul__
    __radd__ = __add__
    @staticmethod
    def rand(nrange=[1000, 10000]):
        return rational([random.randint(nrange[0], nrange[1]), random.randint(nrange[0], nrange[1])])
    @staticmethod
    def convRat(num, digits_after):
        n = digits_after
        b = int(num * 10 ** (n))
        c = 10 ** (n)
        return rational([b, c])
    

        
            
    
class AlgebraicReal:
    def __init__(self, num):
        # num = [rational_part(a) : rational, [coeff(b) : int, nth_root of (x:rational), n:int>0], [coeff(b2),n2th_root of (x2), n2], ...] -> num = a + b(x)^(1/n) + b2(x2)^(1/n2)+...
        self.num = num[:]
    def __call__(self):
        s = self.num[0]() if hasattr(self.num[0], '__call__') else self.num[0]
        for r, x, n in self.num[1:]:
            z = (r() if hasattr(r, '__call__') else r) * (x() if hasattr(x, '__call__') else x) ** (1 / n)
            s += z
            
        return s
            
    def __str__(self):
        '''
        array = []
        #[str(self.num[0])] + ["%s * (%s)^(1 / %d)"%(str(c), str(x), n) for c, x, n in self.num[1:]])
        if self.num[0] != 0:
            array.append(str(self.num[0]))
        
        for c, x, n in self.num[1:]:
            if c != 0:
                if c == 1:
                    array.append("(%s)^(1/%d)"%(str(x), n))
                else:
                    array.append("%s * (%s)^(1 / %d)"%(str(c), str(x), n))
        
        return " + ".join(array)
        '''
        return strpprint(self.pprint())
    def pprint(self):
        if self.num[0] != 0:
            lines = self.num[0].pprint() if hasattr(self.num[0], 'pprint') else [[" "], [str(self.num[0])], [" "]]
        elif hasattr(self.num[0], 'pprint'):
            if self.num[0]() != 0:
                lines = self.num[0].pprint() if hasattr(self.num[0], 'pprint') else [[" "], [str(self.num[0])], [" "]]
            else:
                lines = [[], [], []]
        else:
            lines = [[], [], []]
            
        for r, x, n in self.num[1:]:
            if r == 0:
                continue
            temp_lines = [["  "], [" +"], ["  "]]
            #temp_lines += r.pprint() if hasattr(r, 'pprint') else [[" " for i in range(len(str(r)) + 2)], [" " + str(r) + " "], [" " for i in range(len(str(r)) + 2)]]
            if hasattr(r, 'pprint'):
                z = r.pprint()
                q = connect(connect([[" "], [" "], [" "]], z), [[" "], [" "], [" "]])[:]
                temp_lines1 = connect(temp_lines, q)[:]
            else:
                if r != 1:
                    temp_lines1 = connect(temp_lines, [[" " for i in range(len(str(r)) + 2)], [" " + str(r) + " "], [" " for i in range(len(str(r)) + 2)]])[:]
                else:
                    temp_lines1 = temp_lines[:]
                
            temp_lines2 = connect(temp_lines1, [[str(n) + " "], ["".join([" " for i in range(len(str(n)) - 1)]) + "\\/"], [" " for i in range(len(str(n)) + 1)]])[:]
            temp_lines3 = connect(temp_lines2, [["_" for i in range(len(str(x)))], [str(x)], [" " for i in range(len(str(x)))]])[:]
            #print(temp_lines3)
            lines = connect(lines, temp_lines3[:])[:]
        return lines
    def simplify(self):
        rat_part = [self.num[0].simplify() if hasattr(self.num[0], 'simplify') else self.num[0]]
        arr = []
        for coeff, x, n in self.num[1:]:
            xfact = integer.factorization(x)
            ncoeff = coeff
            nxfact = xfact[:]
            for i in range(len(nxfact)):
                if nxfact[i][1] >= n:
                    ncoeff *= nxfact[i][0] ** (int(nxfact[i][1] / n))
                    nxfact[i][1] -= int(nxfact[i][1] / n) * n
            z = 1
            for i, j in nxfact:
                z *= i ** j
            
            if z == 1:
                rat_part[0] += ncoeff
                continue
            arr.append([ncoeff, z, n])
        
        new_arr = []
        for coeff, x, n in arr:
            t = 0
            for i in range(len(new_arr)):
                if x == new_arr[i][1] and n == new_arr[i][2]:
                    new_arr[i][0] += coeff
                    t = 1
            if t == 0:
                new_arr.append([coeff, x, n])
                    
                
        return AlgebraicReal(rat_part + new_arr)   
    
    def __add__(self, other):
        if isinstance(other, (int, integer, rational)):
            new_arr = self.num[:]
            new_arr[0] += other
            return AlgebraicReal(new_arr[:]).simplify()
        else:
            rat_part = [self.num[0] + other.num[0]]      
            irr_part = self.num[1:] + other.num[1:]
            return AlgebraicReal(rat_part + irr_part).simplify()
    
    def __neg__(self):
        narr = [-i for i in self.num]
        return AlgebraicReal(narr)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        if isinstance(other, (int, integer, rational)):
            new_arr = self.num[:]
            new_arr[0] = other * new_arr[0]
            for i in range(1, len(new_arr)):
                new_arr[i][0] = other * new_arr[i][0]
            
            return AlgebraicReal(new_arr[:])
        
        rat_part = [self.num[0] * other.num[0]]
        irr_part = []
        for i in range(len(self.num)):
            for j in range(1, len(other.num)):
                if i == 0:
                    irr_part.append([self.num[0] * other.num[j][0], other.num[j][1], other.num[j][2]])
                
                else:
                    n = self.num[i][2]
                    m = other.num[j][2]
                    l = integer.lcd(n, m)
                    irr_part.append([self.num[i][0] * other.num[j][0], (self.num[i][1] ** int(l / n)) * (other.num[j][1] ** int(l / m)), l])
        for i in range(1, len(self.num)):
            irr_part.append([other.num[0] * self.num[i][0], self.num[i][1], self.num[i][2]])
        return AlgebraicReal(rat_part + irr_part).simplify()
    __rmul__ = __mul__
    __radd__ = __add__
    def __pow__(self, other):
        s = AlgebraicReal([1, [0, 1, 1]])
        for i in range(other):
            s *= self
        
        return s.simplify()

    @staticmethod
    def rand(num=1, nrange_coeff=[1, 100], nrange_surd=[100, 1000], nrange_root=[1, 5]):
        arr = [rational.rand(nrange=nrange_coeff[:])]
        for i in range(num):
            arr.append([random.randint(nrange_coeff[0], nrange_coeff[1]), random.randint(nrange_surd[0], nrange_surd[1]), random.randint(nrange_root[0], nrange_root[1])])
        
        return AlgebraicReal(arr[:]).simplify()
    
    @staticmethod
    def randpure(num=1, nrange_coeff=[1, 100], nrange_surd=[100, 1000], nrange_root=[2, 5]):
        arr = [0]
        for i in range(num):
            arr.append([random.randint(nrange_coeff[0], nrange_coeff[1]), random.randint(nrange_surd[0], nrange_surd[1]), random.randint(nrange_root[0], nrange_root[1])])
        
        return AlgebraicReal(arr[:]).simplify()
    
    @staticmethod
    def randpurer(num=1, nrange_surd=[100, 1000], nrange_root=[2, 5]):
        arr = [0]
        for i in range(num):
            arr.append([1, random.randint(nrange_surd[0], nrange_surd[1]), random.randint(nrange_root[0], nrange_root[1])])
        
        return AlgebraicReal(arr[:]).simplify()
    
        
class poly:
    def __init__(self, coeffs, variable_type=0):
        self.coeffs = coeffs[:]
        for i in range(len(self.coeffs)):
            if isinstance(self.coeffs[i], float):
                if int(self.coeffs[i]) == self.coeffs[i]:
                    self.coeffs[i] = int(self.coeffs[i])
        self.deg = len(coeffs) - 1
        self.variable_type = variable_type
        self.variable = "x"
        if variable_type == 1:
            self.variable = "y"
        elif variable_type == 2:
            self.variable = "z"
    
    def __call__(self, x):
        s = 0
        for i in range(self.deg + 1):
            s += self.coeffs[i] * x ** i
        
        return s
    def __neg__(self):
        return -1 * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __str__(self):
        '''
        string = []
        for i in range(self.deg + 1):
            if i < self.deg:
                if i > 1:
                    if self.coeffs[i] == 0:
                        continue;
                    elif self.coeffs[i] == 1 or self.coeffs[i] == -1:
                        string += ["%s%s^%d"%("+" if self.coeffs[i] == 1 else "-", "x", i)]
                    else:
                        string += ["%s%s%s^%d"%("+" if math.copysign(1, self.coeffs[i])==1 else "" , str(self.coeffs[i]), "x", i)]
                elif i == 1:
                    if self.coeffs[i] == 0:
                        continue;
                    elif self.coeffs[i] in [1, -1]:
                        string += ["%s%s"%("+" if math.copysign(1, self.coeffs[i])==1 else "-" , "x")]
                    else:
                        string += ["%s%s%s"%("+" if math.copysign(1, self.coeffs[i])==1 else "" ,str(self.coeffs[i]), "x")]
                elif i == 0:
                    if self.coeffs[i] == 0:
                        continue;
                    string += ["%s%s"%("+" if math.copysign(1, self.coeffs[i])==1 else "" , str(self.coeffs[i]))]
            
            else:
                if i > 1:
                    if self.coeffs[i] == 0:
                        continue;
                    elif self.coeffs[i] in [1, -1]:
                        string += ["%s%s^%d"%("+" if math.copysign(1, self.coeffs[i])==1 else "-" , "x", i)]
                    else:
                        string += ["%s%s%s^%d"%("+" if math.copysign(1, self.coeffs[i])==1 else "", str(self.coeffs[i]), "x", i)]
                elif i == 1:
                    if self.coeffs[i] == 0:
                        continue;
                    elif self.coeffs[i] in [1, -1]:
                        string += ["%s%s"%("+" if math.copysign(1, self.coeffs[i])==1 else "-" , "x")]
                    else:
                        string += ["%s%s%s"%("+" if math.copysign(1, self.coeffs[i])==1 else "" ,str(self.coeffs[i]), "x")]
                elif i == 0:
                    if self.coeffs[i] == 0:
                        continue;
                    string += ["%s"%(str(self.coeffs[i]))]

        string.reverse()
        return "".join(string)
        '''
        return strpprint(self.pprint())
    
    def pprint(self):
        new_array = self.coeffs[:]
        new_array.reverse()
        lines = [[], [], []]
        for i in range(len(new_array)):
            temp_lines1 = [[" "], ["+" if sgn(new_array[i]) else "-"], [" "]]
            if new_array[i] == 0:
                continue
            if not isinstance(new_array[i], rational):
                if i == self.deg:
                    temp_lines2 = connect(temp_lines1, [[" " for i in range(len(str(new_array[i])))], [str(abs(new_array[i]))], [" " for i in range(len(str(new_array[i])))]])
                elif i == self.deg - 1:
                    temp_lines2 = connect(temp_lines1, [[" " for i in range(len(str(new_array[i])) + 1)], [str(abs(new_array[i])) + self.variable], [" " for i in range(len(str(new_array[i])) + 1)]])
                    
                else:
                    if abs(new_array[i]) != 1:
                        temp_lines2 = connect(temp_lines1, [["".join([" " for i in range(len(str(abs(new_array[i]))) + 1)]) + str(self.deg - i)], [str(abs(new_array[i])) + self.variable + "".join([" " for i in range(len(str(self.deg - i)))])], [" " for i in range(len(str(abs(new_array[i]))) + len(str(self.deg - i)))]])
                    else:
                        temp_lines2 = connect(temp_lines1, [["".join([" "]) + str(self.deg - i)], [self.variable + "".join([" " for i in range(len(str(self.deg - i)))])], [" " for i in range(1 + len(str(self.deg - i)))]])
                lines = connect(lines, temp_lines2)
            else:
                if i == self.deg:
                    temp_lines2 = connect(temp_lines1, abs(new_array[i]).pprint())
                elif i == self.deg - 1:
                    temp_lines2 = connect(temp_lines1, connect(abs(new_array[i]).pprint(), [[" "], [self.variable], [" "]]))
                else:
                    if abs(new_array[i]) != 1:
                        temp_lines2 = connect(temp_lines1, connect(abs(new_array[i]).pprint(), [[" " + str(self.deg - i)], [self.variable+ "".join([" " for i in range(len(str(self.deg - i)))])], [" "+"".join([" " for i in range(len(str(self.deg - i)))])]]))
                    else:
                        temp_lines2 = connect(temp_lines1, [[" " + str(self.deg - i)], [self.variable+ "".join([" " for i in range(len(str(self.deg - i)))])], [" "+"".join([" " for i in range(len(str(self.deg - i)))])]])
                lines = connect(lines, temp_lines2)
        
        return lines[:]
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            x = self.coeffs[:]
            x[0] += other
            return poly(x[:])
        
        elif isinstance(other, poly):
            large_poly = self if self.deg >= other.deg else other
            small_poly = self if self.deg < other.deg else other
            
            res_arr = small_poly.coeffs[:] + [0 for i in range(large_poly.deg - small_poly.deg)]
            for i in range(len(large_poly.coeffs)):
                res_arr[i] += large_poly.coeffs[i]
            
            return poly(res_arr[:])
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            x = [other * i for i in self.coeffs[:]]
            return poly(x)
        
        elif isinstance(other, poly):
            arr = [0 for i in range(self.deg + other.deg + 1)]
            for i in range(len(self.coeffs)):
                for j in range(len(other.coeffs)):
                    arr[i + j] += self.coeffs[i] * other.coeffs[j]
            
            return poly(arr[:])
    
    def __pow__(self, other):
        if isinstance(other, int):
            p = poly([1])
            for i in range(other):
                p *= self
            return p
    def __eq__(self, other):
        if not isinstance(other, poly):
            return False
        return self.coeffs[:] == other.coeffs[:]
    
    def __truediv__(self, other):
        
        def normalize(poly):
            while poly and poly[-1] == 0:
                poly.pop()
            if poly == []:
                poly.append(0)


        def poly_divmod(num, den):
            #Create normalized copies of the args
            num = num[:]
            normalize(num)
            den = den[:]
            normalize(den)

            if len(num) >= len(den):
                #Shift den towards right so it's the same degree as num
                shiftlen = len(num) - len(den)
                den = [0] * shiftlen + den
            else:
                return [0], num

            quot = []
            divisor = float(den[-1])
            for i in range(shiftlen + 1):
                #Get the next coefficient of the quotient.
                mult = num[-1] / divisor
                quot = [mult] + quot

                #Subtract mult * den from num, but don't bother if mult == 0
                #Note that when i==0, mult!=0; so quot is automatically normalized.
                if mult != 0:
                    d = [mult * u for u in den]
                    num = [u - v for u, v in zip(num, d)]

                num.pop()
                den.pop(0)

            normalize(num)
            return quot, num
        return poly(poly_divmod(self.coeffs[:], other.coeffs)[0])
    
    def __mod__(self, other):
        def normalize(poly):
            while poly and poly[-1] == 0:
                poly.pop()
            if poly == []:
                poly.append(0)


        def poly_divmod(num, den):
            #Create normalized copies of the args
            num = num[:]
            normalize(num)
            den = den[:]
            normalize(den)

            if len(num) >= len(den):
                #Shift den towards right so it's the same degree as num
                shiftlen = len(num) - len(den)
                den = [0] * shiftlen + den
            else:
                return [0], num

            quot = []
            divisor = float(den[-1])
            for i in range(shiftlen + 1):
                #Get the next coefficient of the quotient.
                mult = num[-1] / divisor
                quot = [mult] + quot

                #Subtract mult * den from num, but don't bother if mult == 0
                #Note that when i==0, mult!=0; so quot is automatically normalized.
                if mult != 0:
                    d = [mult * u for u in den]
                    num = [u - v for u, v in zip(num, d)]

                num.pop()
                den.pop(0)

            normalize(num)
            return quot, num
        return poly(poly_divmod(self.coeffs[:], other.coeffs)[1])

    def __round__(self, ndigits = 3):
        return poly([round(i, ndigits=ndigits) for i in self.coeffs])
    
    def diff(self):
        array = []
        for i in range(1, len(self.coeffs)):
            array.append(i * self.coeffs[i])
        
        return poly(array)
    
    def integrate(self, c=0):
        array = [c]
        for i in range(len(self.coeffs)):
            array.append(self.coeffs[i] / (i + 1))
        
        return poly(array)
    
    def resultant(self, other):
        array = []
        for i in range(self.deg + other.deg):
            l = []
            for j in range(self.deg + other.deg):
                l.append(0)
            array.append(l)
        for i in range(other.deg):
            for j in range(self.deg + 1):
                array[j + i][i] = self.coeffs[j]
        
        for i in range(self.deg):
            for j in range(other.deg + 1):
                array[j + i][i + other.deg] = other.coeffs[j]

        return det(array)
    
    def disc(self):
        n = self.deg
        return (1/self.coeffs[-1])*((-1)**(n*(n-1)/2)) * self.resultant(self.diff())
    
    def roots(self):
        if self.deg == 1:
            return rational([-self.coeffs[0], self.coeffs[1]]).simplify()
        
        if self.deg == 2:
            a = self.coeffs[2]
            b = self.coeffs[1]
            c = self.coeffs[0]
            d = b**2 - 4*a*c
            return [(-b + cmath.sqrt(d))/(2*a), (-b - cmath.sqrt(d))/(2*a)]
        
        if self.deg == 3:
            a = self.coeffs[3]
            b = self.coeffs[2]
            c = self.coeffs[1]
            d = self.coeffs[0]
            
            d0 = b**2 - 3*a*c
            d1 = 2*b**3-9*a*b*c+27*d*a**2
            C = ((d1 + cmath.sqrt(d1**2-4*d0**3)) / 2)**(1/3)
            r1 = (-1/(3*a)) * (b + C + d0/C)
            r2, r3 = (self / poly([-r1, 1])).roots()
            
            return [r1, r2, r3]
            
    
    __rmul__ = __mul__
    __radd__ = __add__
    
    def convToMVar(self):
        new_array = []
        for i in range(len(self.coeffs)):
            arr = []
            for j in range(len(self.coeffs)):
                arr2 = []
                for k in range(len(self.coeffs)):
                    arr2.append(0)
                arr.append(arr2)
            new_array.append(arr)
        
        if self.variable_type == 0:
            for i in range(len(self.coeffs)):
                new_array[i][0][0] = self.coeffs[i]
            return polymvar(new_array)
        
        elif self.variable_type == 1:
            for i in range(len(self.coeffs)):
                new_array[0][i][0] = self.coeffs[i]
            return polymvar(new_array)
        
        elif self.variable_type == 2:
            for i in range(len(self.coeffs)):
                new_array[0][0][i] = self.coeffs[i]
            return polymvar(new_array)
        
        
        
    @staticmethod
    def rand(deg, coeff_range = [0, 10]):
        coeffs = [(-1)**random.randint(1, 10) * random.randint(coeff_range[0], coeff_range[1]) for i in range(deg + 1)]
        return poly(coeffs)
    
    @staticmethod
    def randrat(deg, coeff_range = [0, 10]):
        coeffs = [(-1)**random.randint(1, 10) * rational.rand(nrange=coeff_range[:]).simplify() for i in range(deg + 1)]
        return poly(coeffs)
    
    @staticmethod
    def newtonsmethod(pl, start, max_iter):
        x_i = start
        x_ig = start
        for i in range(max_iter):
            x_ig = x_i
            x_i -= pl(x_i) / pl.diff()(x_i)
            x_i = round(x_i, ndigits=10)
            if x_i == x_ig:
                break
        
        return x_i

class PowSeries:
    def __init__(self, c_n, name=None):
        self.function = c_n
        self.name = name
    
    def poly(self, n):
        return poly([self.function(i) for i in range(n + 1)])
    
    def __call__(self, x, n=DEFAULT_TAYLOR_N):
        return self.poly(n)(x)

    def diff(self):
        new_function = lambda n : self.function(n+1) * (n+1)
        return PowSeries(new_function)
    
    def integrate(self):
        new_function = lambda n : self.function(n-1) /(n) if n != 0 else 0
        return PowSeries(new_function)
    
    def __add__(self, other):
        if isinstance(other, (int, float, poly)):
            other_pol = other
            if isinstance(other, (int, float)):
                other_pol = poly([other])
            
            new_cn = lambda n : self.function(n) + other.coeffs[n] if n <= other.deg else self.function(n)
            return PowSeries(new_cn)
        else:
            new_cn = lambda n : self.function(n) + other.function(n) 
            return PowSeries(new_cn)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_cn = lambda n : other * self.function(n)
            return PowSeries(new_cn)
        
        elif isinstance(other, poly):
            new_cn = lambda n : sum([self.function(n - i) * other.coeffs[i] for i in range(min(other.deg + 1, n+1))])
            return PowSeries(new_cn)
        
        else:
            new_cn = lambda n : sum([self.function(n - i) * other.function(i) for i in range(n+1)])
            return PowSeries(new_cn)
    
    def __neg__(self):
        return PowSeries(lambda n : -self.function(n))
    
    def __sub__(self, other):
        return self + (-other)
    
    def __str__(self, n=3):
        if self.name is not None:
            return strpprint(self.pprint())
        return str(self.poly(n))
    
    def pprint(self, n=3):
        if self.name is not None:
            return [[" " for i in range(len(self.name))], [l for l in self.name], [" " for i in range(len(self.name))]]
        return self.poly(n).pprint()
    
    __radd__ = __add__
    __rmul__ = __mul__
    
def solveDEseries(coeffs, rhs, init_val, n):
    '''
    coeffs = [1, 2, 3]
    rhs = PowSeries
    
    1y+2y'+3y''=rhs
    ans = a1c_{n-2} + a2c_{n-1} + a3c_{n}... -> [a3, a2, a1]
    
    init_val = [1, 2, 3, 4, ...]
    c0 = 1; c1 = 2; c2 = 3; c3 = 4; ...
    '''
    deg = len(coeffs) - 1
    k = 0
    vals = init_val[:]
    v_array = vals[:]
    while k < n:
        arr = [coeffs[i] * math.factorial(k+i)/math.factorial(k) for i in range(deg + 1)]
        x = dot(arr[:-1], vals[:])
        new_n = (rhs.function(k)-x)/arr[-1]
        v_array.append(new_n)
        new_vals = vals[1:] + [new_n]
        vals = new_vals[:]
        k+=1
    
    return v_array[:]
    

    
        
SIN = PowSeries(lambda n : (n%2) * (-1)**int((n-1)/2)/(math.factorial(n)))
COS = PowSeries(lambda n : (1-n%2) * (-1)**int(n/2)/math.factorial(n))
EXP = PowSeries(lambda n : 1/math.factorial(n))
EXP_ = PowSeries(lambda n : (-1)**n/math.factorial(n))
LOG_1_X = PowSeries(lambda n : -1/n if n != 0 else 0)
SINH = (1/2) * (EXP - EXP_)
COSH = (1/2) * (EXP + EXP_)


class matrix:
    def __init__(self, array, name=None):
        self.name = name
        self.array = array[:]
    def __str__(self):
        '''
        new_arr = []
        longest_length = 0
        for i in self.array:
            z = []
            for j in i:
                s = str(j)
                if len(s) > longest_length:
                    longest_length = len(s)
                z.append(s)

            new_arr.append(z)
            
        fstring = []
        for i in new_arr:
            for j in range(len(i)):
                x = ""
                for k in range(int((longest_length - len(i[j]))/2)):
                    x += " "
                z = "".join([x, i[j], x, " " if longest_length - len(i[j]) % 2 == 1 else "",  ", " if j != len(i) - 1 else ""])
                fstring.append(z)
            fstring.append("\n")
            
        return "".join(fstring)
        '''
        return matrixpprint(self.pprint())
    def pprint(self):
        if self.name is not None:
            return [[" " for i in range(len(self.name))], [l for l in self.name], [" " for i in range(len(self.name))]]
        tot_cells = []
        for i in self.array:
            cells = []
            for j in i:
                lines = [[], [], []]
                if hasattr(j, 'pprint'):
                    lines = connect(lines, j.pprint())
                else:
                    lines = connect(lines, [[" " for i in range(len(str(j)))], [k for k in str(j)], [" " for i in range(len(str(j)))]])
                    
                cells.append(lines)
            
            tot_cells.append(cells)
        
        longest_length = 0
        for i in tot_cells:
            for j in i:
                if max([len(k) for k in j]) > longest_length:
                    longest_length = max([len(k) for k in j])
        tot_lines = []
        for i in range(len(tot_cells)):
            lines = [[], [], []]
            for j in range(len(tot_cells[i])):
               mlen = max([len(k) for k in tot_cells[i][j]])
               new_cell = connect(tot_cells[i][j], [[" " for k in range((longest_length - mlen)//2)]for h in range(3)])
               new_cell2 = connect([[" " for k in range((longest_length - mlen)//2 + (longest_length - mlen) % 2)]for h in range(3)], new_cell)
               lines = connect(lines, new_cell2)
               lines = connect(lines, [["   "], [" , "], ["   "]])
            tot_lines.append(lines)
        
        return tot_lines[:]
            
    def __call__(self, x):
        new_arr = []
        for i in range(len(self.array)):
            sub = []
            for j in range(len(self.array[0])):
                sub.append(self.array[i][j](x) if isinstance(self.array[i][j], poly) else self.array[i][j])
            new_arr.append(sub)
        return matrix(new_arr[:])
        
    
    def __add__(self, other):
        self_copy = self.array[:]
        for i in range(len(self.array)):
            for j in range(len(self.array[i])):
                self_copy[i][j] += other.array[i][j]

        return matrix(self_copy)
    
    def __mul__(self, other):
        if isinstance(other, (int, float, poly)):
            x = self.array[:]
            for i in range(len(x)):
                for j in range(x[i]):
                    x[i][j] = other * x[i][j]

            return matrix(x)
        
        elif isinstance(other, matrix):
            if len(self.array[0]) != len(other.array):
                raise Exception("Dimensions not compatible.")
            
            else:
                arr = []
                for i in range(len(self.array)):
                    sub_arr = []
                    for j in range(len(other.array[0])):
                        sub_arr.append(sum([self.array[i][k] * other.array[k][j] for k in range(len(self.array[i]))]))
                    arr.append(sub_arr[:])
                
                return matrix(arr)
    
    def det(self):
        return det(self.array[:])
    
    def __eq__(self, other):
        return self.array[:] == other.array[:]
    
    def charpoly(self):
        new_arr = [[j for j in i] for i in self.array[:]]
        for i in range(len(self.array)):
            new_arr[i][i] = poly([float(new_arr[i][i]), -1])
        
        return matrix(new_arr[:]).det()
    
    def eigenvalue(self):
        return [i.real for i in self.charpoly().roots()[:]]
    
    __rmul__ = __mul__
    __radd__ = __add__
    @staticmethod
    def rand(dims=[3, 3], nrange=[1, 10]):
        arr = []
        for i in range(dims[0]):
            sub = []
            for j in range(dims[1]):
                sub.append(random.randint(nrange[0], nrange[1]))

            arr.append(sub)
        
        return matrix(arr)
    
    @staticmethod
    def randpoly(dims = [3, 3], max_deg = 1, coeff_range = [1, 10]):
        arr = []
        for i in range(dims[0]):
            sub = []
            for j in range(dims[1]):
                sub.append(poly.rand(random.randint(0, max_deg), coeff_range=coeff_range[:]))

            arr.append(sub)
        
        return matrix(arr)
    @staticmethod
    def randrat(dims=[3, 3], nrange=[1, 10]):
        arr = []
        for i in range(dims[0]):
            sub = []
            for j in range(dims[1]):
                sub.append(rational.rand(nrange[:]).simplify())

            arr.append(sub)
        
        return matrix(arr)
class vect:
    def __init__(self, array):
        self.array = array[:]
        self.dim = len(array)
        
    def __add__(self, other):
        return vect([self.array[i] + other.array[i] for i in range(self.dim)])
    
    def __mul__(self, other):
        if isinstance(other, (int, float, poly, PowSeries, polymvar)):
            return vect([i * other for i in self.array])
        
        elif isinstance(other, (vect, list, tuple)):
            if isinstance(other, vect):
                return dot(self.array[:], other.array[:])
            
            else:
                return dot(self.array[:], other[:])
        
        else:
            raise Exception(ValueError)
    
    def __neg__(self):
        return (-1) * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __str__(self):
        return "<" + ", ".join([str(i) for i in self.array]) + ">"
    
    def length(self):
        return math.sqrt(sum([i**2 for i in self.array[:]]))
    
    def cross(self, other):
        return vect(cross(self.array[:], other.array[:]))
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    
class pcurve:
    def __init__(self, farr):
        self.array = farr[:]
        self.dim = len(farr)
    
    def __call__(self, x):
        return vect([i(x) if callable(i) else i for i in self.array])
    
    def __str__(self):
        return "<" + ", ".join([str(i) for i in self.array]) + ">"
    
    def __add__(self, other):
        return pcurve([self.array[i] + other.array[i] for i in range(self.dim)])
    
    def __mul__(self, other):
        if isinstance(other, (int, float, poly, PowSeries)):
            return pcurve([i * other for i in self.array])
        
        elif isinstance(other, (vect, list, tuple)):
            if isinstance(other, vect):
                return dot(self.array[:], other.array[:])
            
            else:
                return dot(self.array[:], other[:])
        
        elif isinstance(other, pcurve):
            return dot(self.array[:], other.array[:])
        
        else:
            raise ValueError
    
    def __neg__(self):
        return (-1) * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def cross(self, other):
        return vect(cross(self.array[:], other.array[:]))
    
    def diff(self):
        a = []
        for i in self.array:
            if hasattr(i, 'diff'):
                a.append(i.diff())
        
        return pcurve(a[:])
    
    def length(self):
        return lambda x : math.sqrt(sum([i(x)**2 if callable(i) else i ** 2 for i in self.array[:]]))
    
    def alen(self, a, b):
        dvt = self.diff().length()
        return numericIntegration(dvt, a, b)
    
    def pprint(self):
        pstr = [[" "], ["<"], [" "]]
        for i in self.array:
            npstr = connect(pstr[:], connect(i.pprint()[:], [[" "], [","], [" "]]))[:]
            pstr = npstr[:]
        
        return connect(pstr[:], [[" "], [">"], [" "]])
    
    def __str__(self):
        return strpprint(self.pprint())
    
    def curvature(self):
        return lambda x : self.diff()(x).cross(self.diff().diff()(x)).length()/self.diff().length()(x)**3
    
    def T(self):
        return lambda x : self.diff()(x) * (1/self.diff().length()(x))
    
    def N(self):
        return lambda x : numericDiff(self.T(), x) * (1/numericDiff(self, x).length())* (1 / self.curvature()(x))
    
    def B(self):
        return lambda x : self.T()(x).cross(self.N()(x))
    
    __radd__ = __add__
    __rmul__ = __mul__
    @staticmethod
    def rand(max_deg=2, nranges=[1, 10]):
        return pcurve([poly.rand(random.randint(0, max_deg), coeff_range=nranges[:]) for i in range(3)])
        

class polymvar:
    def __init__(self, array):
        '''
        for now the number of variables is always 3.
        
        array[i][j][k] = c where c is the coefficient of
        x^i y^j z^k 
        '''
        self.array = array[:]

    def __call__(self, x, y, z):
        s = 0
        for i in range(len(self.array)):
            for j in range(len(self.array)):
                for k in range(len(self.array)):
                    s += self.array[i][j][k] * (x ** i) * (y ** j) * (z ** k)
        
        return s
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_array = self.array[:]
            new_array[0][0][0] += other
            return polymvar(new_array[:])
        
        elif isinstance(other, polymvar):
            new_array = self.array[:] if len(self.array) >= len(other.array) else other.array[:]
            m_arr = other.array[:] if len(self.array) >= len(other.array) else self.array[:]
            narr = [[[0 for k in range(len(new_array))] for j in range(len(new_array))] for i in range(len(new_array))]
            q = len(m_arr)
            for i in range(len(new_array)):
                for j in range(len(new_array)):
                    for k in range(len(new_array)):
                        narr[i][j][k] += (new_array[i][j][k] + m_arr[i][j][k]) if i < q and j < q and k < q else (new_array[i][j][k])
            
            return polymvar(narr[:])
        
        elif isinstance(other, poly):
            return self + other.convToMVar()
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_array = self.array[:]
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][j][k] *= other
            return polymvar(new_array)
        
        elif isinstance(other, polymvar):
            new_array = []
            for i in range(len(self.array) + len(other.array)):
                arr = []
                for j in range(len(self.array) + len(other.array)):
                    arr2 = []
                    for k in range(len(self.array) + len(other.array)):
                        arr2.append(0)
                    arr.append(arr2)
                new_array.append(arr)
            for i1 in range(len(self.array)):
                for j1 in range(len(self.array)):
                    for k1 in range(len(self.array)):
                        for i2 in range(len(other.array)):
                            for j2 in range(len(other.array)):
                                for k2 in range(len(other.array)):
                                    new_array[i1+i2][j1+j2][k1+k2] += self.array[i1][j1][k1] * other.array[i2][j2][k2]
            
            return polymvar(new_array[:])
        
        elif isinstance(other, poly):
            return self * other.convToMVar()
    
    def __pow__(self, other):
        ns = self
        for i in range(other - 1):
            ns *= self
        return ns
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    __rmul__ = __mul__
    __radd__ = __add__
    
    
    def pprint(self):
        new_array = self.array[:]
        lines = [[], [], []]
        for i in range(len(new_array)):
            for j in range(len(new_array)):
                for k in range(len(new_array)):
            
                    temp_lines1 = [[" "], ["+" if sgn(new_array[i][j][k]) else "-"], [" "]]
                    if new_array[i][j][k] == 0:
                        continue
                    temp_lines2 = [[], [], []]
                    sarr = [[" " for l in range(len(str(abs(new_array[i][j][k]))))], [l for l in str(abs(new_array[i][j][k]))], [" " for l in range(len(str(abs(new_array[i][j][k]))))]]
                    #nsubarr = connect(sub_arr[:], [[" "] + [l for l in str(i)] + [" "] + [l for l in str(j)] + [" "] + [l for l in str(k)], ["x"] + [" " for l in range(len(str(i)))] + ["y"] + [" " for l in range(len(str(j)))] + ["z"] + [" " for l in range(len(str(k)))], [" " for l in range(3 + len(str(i)) + len(str(j)) + len(str(k)))]])
                    
                    if i != 0:
                        z = [[" "] + [l for l in str(i)], ["x"] + [" " for l in range(len(str(i)))], [" " for l in range(1 + len(str(i)))]]
                        if i == 1:
                            z = [[" "], ["x"] , [" "]]
                        sarr = connect(sarr, z)[:]
                    if j != 0:
                        z = [[" "] + [l for l in str(j)], ["y"] + [" " for l in range(len(str(j)))], [" " for l in range(1 + len(str(j)))]]
                        if j == 1:
                            z = [[" "], ["y"] , [" "]]
                        sarr = connect(sarr, z)[:]
                    if k != 0:
                        z = [[" "] + [l for l in str(k)], ["z"] + [" " for l in range(len(str(k)))], [" " for l in range(1 + len(str(k)))]]
                        if k == 1:
                            z = [[" "], ["z"] , [" "]]
                        sarr = connect(sarr, z)[:]                    
                    lines = connect(lines[:], connect(temp_lines1[:], sarr[:]))
        
        return lines[:]
    
    def __str__(self):
        return strpprint(self.pprint())
    
    def diff(self, wrt):
        new_array = [[[0 for k in range(len(self.array))] for j in range(len(self.array))] for i in range(len(self.array))]
        if wrt == 0:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        if i != 0:
                            new_array[i - 1][j][k] = i * self.array[i][j][k]
            
            
            return polymvar(new_array[:])
        
        if wrt == 1:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        if j != 0:
                            new_array[i][j - 1][k] = j * self.array[i][j][k]
            
            return polymvar(new_array[:])
        
        if wrt == 2:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        if k != 0:
                            new_array[i][j][k - 1] = k * self.array[i][j][k]
            
            return polymvar(new_array[:])
        
        else:
            return 0
        
    
    def integrate(self, wrt):
        new_array = [[[0 for k in range(len(self.array) + 1)] for j in range(len(self.array) + 1)] for i in range(len(self.array) + 1)]
        if wrt == 0:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i + 1][j][k] = self.array[i][j][k] / (i + 1)
            
            return polymvar(new_array[:])
        
        if wrt == 1:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][j + 1][k] = self.array[i][j][k] / (j + 1)
            
            return polymvar(new_array[:])
        
        if wrt == 2:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][j][k + 1] = self.array[i][j][k] / (k + 1)
            
            return polymvar(new_array[:])
        
        else:
            return 0
        
    def c_integrate(self, curve, t0, t1):
        new_f = self(curve.array[0], curve.array[1], curve.array[2])
        f = lambda x : new_f(x) * curve.diff().length()(x)
        return numericIntegration(f, t0, t1, dx=0.00001)


    def peval(self, x, y, z):
        if [x, y, z].count(None) == 0:
            return self(x, y, z)
        
        new_array = [[[0 for k in range(len(self.array))] for j in range(len(self.array))] for i in range(len(self.array))]
        cond = True
        if [x, y, z].count(None) < 2 :
            return self.peval(x, None, None).peval(None, y, None).peval(None, None, z)
        
        if x is not None: 
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[0][j][k] += self.array[i][j][k] * (x**i)
            cond = False
        
        if y is not None:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][0][k] += (new_array[i][j][k] * (y**j)) if not cond else self.array[i][j][k] * (y**j)
            
            cond = False
        
        if z is not None:   
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][j][0] += (new_array[i][j][k] * (z**k)) if not cond else self.array[i][j][k] * (z**k)
            
            cond = False
        
        if cond:
            return self
        
        return polymvar(new_array[:])
    
    def grad(self):
        return vect([self.diff(0), self.diff(1), self.diff(2)])

    @staticmethod
    def rand(max_deg=2, nrange=[1, 10]):
        new_array = []
        for i in range(max_deg):
            arr = []
            for j in range(max_deg):
                arr2 = []
                for k in range(max_deg):
                    arr2.append(random.randint(nrange[0], nrange[1]) * random.randint(0, 1))
                arr.append(arr2)
            new_array.append(arr)
        
        return polymvar(new_array[:])

class vectF:
    def __init__(self, array):
        self.array = array[:]
        self.dim = len(array)
        
    def __add__(self, other):
        return vect([self.array[i] + other.array[i] for i in range(self.dim)])
    
    def __call__(self, x, y, z):
        return vectF([i(x, y, z) if callable(i) else i for i in self.array[:]])
    
    def __mul__(self, other):
        if isinstance(other, (int, float, poly, PowSeries, polymvar)):
            return vect([i * other for i in self.array])
        
        elif isinstance(other, (vect, list, tuple, pcurve, vectF)):
            if isinstance(other, (vect, vectF, pcurve)):
                return sdot(self.array[:], other.array[:])
            
            else:
                return sdot(self.array[:], other[:])
        
        else:
            raise Exception(ValueError)
    
    def __neg__(self):
        return (-1) * self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __str__(self):
        return "<" + ", ".join([str(i) for i in self.array]) + ">"
    
    def length(self):
        return math.sqrt(sum([i**2 for i in self.array[:]]))
    
    def cross(self, other):
        return vect(cross(self.array[:], other.array[:]))
    
    def div(self):
       return sum([self.array[i].diff(i) for i in range(len(self.array))])
    
    def curl(self):
        a = self.array[2].diff(1)-self.array[1].diff(2)
        b = self.array[0].diff(2)-self.array[2].diff(0)
        c = self.array[1].diff(0)-self.array[0].diff(1)
        return vect([a, b, c])
    
    def integrate(self, curve):
        nf = self(curve.array[0], curve.array[1], curve.array[2])
        dl = curve.diff()
        n = nf * dl
        return n.integrate()
    
    
    __radd__ = __add__
    __rmul__ = __mul__
    @staticmethod
    def rand(max_deg=2, nranges=[1, 10]):
        return vectF([polymvar.rand(max_deg=max_deg, nrange=nranges[:]) for i in range(3)])
    
    @staticmethod
    def randclsd(nranges = [1, 10], b=0):
        s = SIN
        c = COS
        s.name = "sin(t)"
        c.name = "cos(t)"
        return vectF([random.randint(nranges[0], nranges[1]) * c,
                      random.randint(nranges[0], nranges[1]) * s ,
                      0])

def laplace_t(function):
    return lambda s : simpInt(lambda x : function(x) * math.exp(-s*x), 0, 1000)

def fourier_series(f, period):
    a_n_d = lambda n : (lambda x : f(x) * math.cos(2 * n * math.pi * x / period))
    b_n_d = lambda n : (lambda x : f(x) * math.sin(2 * n * math.pi * x / period)) 
    a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
    b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
    a_0 = numericIntegration(f, -period/2, period/2) / period
    return [a_n, b_n, a_0]

def generate_integrable_ratExpr(deg=3, nranges = [1, 10]):
    p = poly([1])
    p_deg = random.randint(0, deg)
    q = poly([1])
    q_deg = random.randint(0, deg)
    for i in range(p_deg // 2):
        s1 = random.randint(1, 2) % 2
        p *= s1 * poly.rand(2, coeff_range=nranges[:]) + (1-s1)*poly.rand(1, coeff_range=nranges[:])*poly.rand(1, coeff_range=nranges[:])
    for i in range(q_deg // 2):
        s2 = random.randint(1, 2) % 2
        q *= s2 * poly.rand(2, coeff_range=nranges[:]) + (1-s2)*poly.rand(1, coeff_range=nranges[:])*poly.rand(1, coeff_range=nranges[:])
    for i in range(p_deg % 2):
        p *= poly.rand(1, coeff_range=nranges[:])
    for i in range(q_deg % 2):
        q *= poly.rand(1, coeff_range=nranges[:])
    
    str1 = str(p)
    str2 = str(q)
    str1cpy = str1[:]
    str2cpy = str2[:]
    len_measure1 = len(str1cpy.split("\n")[0])
    len_measure2 = len(str2cpy.split("\n")[0])
    str3 = "".join(["-" for j in range(max(len_measure1, len_measure2))])
    z = str1 + "\n" + str3 + "\n" + str2 + "\n"
    return [p, q, z]

def generate_eulersub(deg=2, nranges=[1, 10]):
    rat1 = generate_integrable_ratExpr(deg=deg, nranges=nranges[:])
    sq_term = poly.rand(2, coeff_range=nranges[:])
    for i in range(len(sq_term.coeffs[:])):
        sq_term.coeffs[i] = abs(sq_term.coeffs[i])
    sqf = lambda x : math.sqrt(sq_term(x)) if sq_term(x) > 0 else 1
    rat2seed = random.randint(1, 2)%2
    rat2 = (lambda x : sqf(x)) if rat2seed else (lambda x : 1/sqf(x))
    tot_func = lambda x : (rat1[0](x) / rat1[1](x)) * rat2(x)
    p1, q1, z1 = rat1[:]
    z3 = sq_term.pprint()[:]
    z3_cp2 = ["\\", "/"] + z3[1]
    z3_cp1 = ["  ", "/"] + z3[0]
    v = [" " for i in range(len(z3_cp1))]
    z3_cpy = [
                [" ", " "] + ["-" for i in range(len(z3[2]) - 2)], 
                z3_cp1[:],
                z3_cp2[:],
                v
            ]
    z1, z2 = p1.pprint(), q1.pprint()
    l1 = max([len("".join(i)) for i in z1])
    l2 = max([len("".join(i)) for i in z2])
    l3 = max(l1, l2)
    p1pprintmod = p1.pprint()[:-1]
    q1pprintmod = q1.pprint()[:-1]
    for i in range(len(p1pprintmod)):
        string = p1pprintmod[i] + [" " for i in range(l3-len("".join(p1pprintmod[i])))]
        p1pprintmod[i] = string
        
    for i in range(len(q1pprintmod)):
        string = q1pprintmod[i] + [" " for i in range(l3-len("".join(q1pprintmod[i])))]
        q1pprintmod[i] = string
    
    p1pprint = [[" " for i in range(l3)]] + [[" " for i in range(l3)]] + p1pprintmod[:]
    q1pprint = q1pprintmod[:]+[[" " for i in range(l3)]]
    ratstr1 = p1pprint + [["-"for i in range(l3)]] + q1pprint
    ratstr1 = connect([["/"], ["|"], ["|"], ["|"], ["|"], ["|"], ["|"], ["\\"]], ratstr1)
    ratstr1 = connect(ratstr1, [["\\"], ["|"], ["|"], ["|"], ["|"], ["|"], ["|"], ["/"]])
    
    
    if rat2seed:
            z3_cpy2 = [
                v,
                v,
                [" ", "  "] + ["-" for i in range(len(z3[2]) - 2)],
                z3_cp1[:],
                z3_cp2[:],
                v,
                v,
                v    
            ]
            
            finstr = connect(ratstr1, z3_cpy2)
            return [tot_func, strpprint(finstr)]
    
    else:
        z3_cpy2 = [
                v,
                v,
                v,
                [" " for i in range((len(v)-1) // 2)] + ["1"] + [" " for i in range((len(v)-1) // 2 + (len(v)-1) % 2)],
                ["-" for i in range(len(v))],
                [" ", "  "] + ["-" for i in range(len(z3[2]) - 2)], 
                z3_cp1[:],
                z3_cp2[:],
            ]
        finstr = connect(ratstr1, z3_cpy2)
        return [tot_func, strpprint(finstr)]

def generate_trig(nranges=[1, 10]):
    a, s, c = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    a2, s2, c2 = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    a3, t = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    modeseed = random.randint(1, 3) 
    if modeseed == 1:
        p = lambda x : a + s * math.sin(x) + c * math.cos(x)
        q = lambda x : a2 + s2 * math.sin(x) + c2 * math.cos(x)
        p_str = "%d + %dsin(x) + %dcos(x)" % (a, s, c)
        q_str = "%d + %dsin(x) + %dcos(x)" % (a2, s2, c2)
        return [lambda x : p(x) / q(x), p_str + "\n" + "".join(["-" for i in range(max(len(p_str), len(q_str)))]) + "\n" + q_str]
    
    elif modeseed == 2:
        l = random.randint(nranges[0], nranges[1])
        s = random.randint(1, 2) % 2
        p = lambda x : l / (a3 + t * ((-1)**s) * math.tan(x))
        q_str = "%d + %dtan(x)"%(a3, t * ((-1)**s)) if not s else "%d - %dtan(x)"%(a3, t)
        t_str = str(l) + "\n" + "".join(["-" for i in range(max(len(str(l)), len(q_str)))]) + "\n" + q_str
        return [p, t_str]
    
    elif modeseed == 3:
        return generate_trig_prod(nranges=nranges[:])

def generate_trig_prod(nranges=[1, 10]):
    a, b = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    function = lambda x : (math.sin(x)**a)*(math.cos(x))**b
    x = [i for i in str(a)] if a != 1 else [" "]
    y = [i for i in str(b)] if a != 1 else [" "]
    string_array = [[" ", " ", " "] + x + [" ", " ", " ", " ", " "] + y + [" "],
                    ["s", "i", "n"] + [" " for i in range(len(str(a)))] + ["x", " "] + ["c", "o", "s"] + [" " for i in range(len(str(b)))] + ["x"]]
    string = "\n".join(["".join(i) for i in string_array])
    return [function, string]

def generate_fourier_s(nranges=[1, 10], deg=2, p_range=[1, 5], exp_cond=False, u_cond=False, umvar_cond=False):
    p1 = poly.rand(deg, coeff_range=nranges[:])
    c1 = random.randint(nranges[0], nranges[1])
    period = 2*random.randint(p_range[0], p_range[1])
    rand_exp = lambda x : math.exp(c1 * x)
    f = (lambda x : p1(x) * rand_exp(x))
    if not exp_cond:
        f = lambda x : p1(x)
    if umvar_cond:
        x, y, z = rndF(nranges=nranges), rndF(nranges=nranges), rndF(nranges=nranges)
        p1 = polymvar.rand(max_deg=2, nrange=nranges[:])
        f = lambda t : p1(x[0](t), y[0](t), z[0](t))
        a_n, b_n, a_0 = fourier_series(f, period)
        nstr = connect(p1.pprint(), [[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "w", "h", "e", "r", "e", " ", "x", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, x[1])
        nstr = connect(pnstr, [[" ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "a", "n", "d", " ", "y", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, y[1])
        nstr = connect(pnstr, [[" ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "a", "n", "d", " ", "z", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, z[1])
        return [f, period, a_n, b_n, a_0, strpprint(pnstr), p1, c1]
    if u_cond:
        z = rndF(nranges=nranges)
        f = lambda x : p1(z[0](x))
        a_n, b_n, a_0 = fourier_series(f, period)
        nstr = connect(p1.pprint(), [[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "w", "h", "e", "r", "e", " ", "x", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, z[1])
        return [f, period, a_n, b_n, a_0, strpprint(pnstr), p1, c1]
    
     
        
    a_n_d = lambda n : (lambda x : f(x) * math.cos(2 * n * math.pi * x / period))
    b_n_d = lambda n : (lambda x : f(x) * math.sin(2 * n * math.pi * x / period)) 
    a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
    b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
    a_0 = numericIntegration(f, -period/2, period/2) / period
    if not exp_cond:
        return [f, period, a_n, b_n, a_0, strpprint(p1.pprint()), p1, c1]
    poly_pprint = connect(connect([["/"], ["|"], ["\\"]], p1.pprint()), [["\\"], ["|"], ["/"]])
    array = [[" "], ["e"], [" "]]
    for i in str(c1):
        array[0].append(i)
    array[0].append("x")
    for i in range(len(str(c1)) + 1):
        array[1].append(" ")
        array[2].append(" ")
    
    full_pprint = connect(array, poly_pprint)
    return [f, period, a_n, b_n, a_0, strpprint(full_pprint), p1, c1]

def fourier_s_poly(p1, p_range=[1, 5]):
    period = 2*random.randint(p_range[0], p_range[1])
    f = lambda x : p1(x)
    a_n_d = lambda n : (lambda x : f(x) * math.cos(2 * n * math.pi * x / period))
    b_n_d = lambda n : (lambda x : f(x) * math.sin(2 * n * math.pi * x / period)) 
    a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
    b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
    a_0 = numericIntegration(f, -period/2, period/2) / period
    return [period, a_n, b_n, a_0]

def randFunction(nranges=[1, 10], n=2, max_deg=2):
    functions = [(SINH, [[" ", " ", " ", " ", " ", " ", " "], ["s", "i", "n", "h", "(", "x", ")"], [" ", " ", " ", " ", " ", " ", " "]]), 
                 (COSH, [[" ", " ", " ", " ", " ", " ", " "], ["c", "o", "s", "h", "(", "x", ")"], [" ", " ", " ", " ", " ", " ", " "]]), 
                 (EXP, [[" ", "x"], ["e"," "], [" ", " "]])]
    return functions[random.randint(0, len(functions) - 1)]

def rndF(nranges=[1, 10]):
    a, b, c, d, e = [random.randint(nranges[0], nranges[1]) for i in range(5)]
    functions = [(lambda x : math.sinh(a * x), [[" ", " ", " ", " ", " "] + [" " for i in str(a)] + [ " ", " "], ["s", "i", "n", "h", "("] + [i for i in str(a)] + ["x", ")"], [" ", " ", " ", " ", " "] + [" " for i in str(a)] + [ " ", " "]]), 
                 (lambda x : math.cosh(b * x), [[" ", " ", " ", " ", " "] + [" " for i in str(b)] + [ " ", " "], ["c", "o", "s", "h", "("] + [i for i in str(b)] + ["x", ")"], [" ", " ", " ", " ", " "] + [" " for i in str(b)] + [ " ", " "]]), 
                 (lambda x : math.exp(c * x), [[" "] + [i for i in str(c)] + ["x"], ["e"," "] + [" " for i in str(c)], [" ", " "] + [" " for i in str(c)]]),
                 (lambda x : math.sin(d * x), [[" ", " ", " ", " "] + [" " for i in str(d)] + [ " ", " "], ["s", "i", "n", "("] + [i for i in str(d)] + ["x", ")"], [" ", " ", " ", " "] + [" " for i in str(d)] + [ " ", " "]]), 
                 (lambda x : math.cos(e * x), [[" ", " ", " ", " "] + [" " for i in str(e)] + [ " ", " "], ["c", "o", "s", "("] + [i for i in str(e)] + ["x", ")"], [" ", " ", " ", " "] + [" " for i in str(e)] + [ " ", " "]]), 
                 ]
    return functions[random.randint(0, len(functions) - 1)]


def runge_kutta_2nd(coeffs, function, start, step, init_vals):
    '''
    init_vals = [y(x0), y'(x0)]
    '''
    def f(x):
        z_i = init_vals[-1]
        y_i = init_vals[0]
        x_i = start
        while x_i < x:
            k1 = step * z_i
            L1 = step * (1/coeffs[-1]) * (function(x_i) - coeffs[0] - coeffs[1]*y_i)
            
            k2 = step * (z_i+L1/2)
            L2 = step * (1/coeffs[-1]) * (function(x_i+step/2) - coeffs[0] - coeffs[1]*(y_i + k1/2))
            
            k3 = step * (z_i + L2/2)
            L3 = step * (1/coeffs[-1]) * (function(x_i+step/2) - coeffs[0] - coeffs[1]*(y_i + k2/2))
            
            k4 = step * (z_i + L3)
            L4 = step * (1/coeffs[-1]) * (function(x_i+step) - coeffs[0] - coeffs[1]*(y_i + k3))
            
            y_i += (1/6)*(L1+2*L2+2*L3+L4)
            z_i += (1/6)*(k1+2*k2+2*k3+k4)
            x_i += step
        return y_i
    
    return f
def random_diff_eq_2(nranges=[1, 10], n=2, max_deg=2):
    coeffs = [random.randint(nranges[0], nranges[1]) * (-1)**random.randint(0, 1) for i in range(3)]
    ppr = [[], [], []]
    for i in range(len(coeffs) - 1, -1, -1):
        if coeffs[i] != 0:
            if abs(coeffs[i]) != 1:
                ppr = connect(ppr[:], [[" " for j in range(len(str(abs(coeffs[i]))) + 2)]+["'" for j in range(i)],["+" if coeffs[i] > 0 else "-"]+[j for j in str(abs(coeffs[i]))] + ["y"] + [" " for j in range(i)], [" " for j in range(len(str(abs(coeffs[i]))) + i + 2)]])[:]
            else:
                ppr = connect(ppr[:], [[" " for j in range(1+ 2)]+["'" for j in range(i)],["+" if coeffs[i] > 0 else "-"]+ ["y"] + [" " for j in range(i)], [" " for j in range(i + 3)]])[:]  
    ppr = connect(ppr[:], [["   "], [" = "], ["   "]])
    f, pprint_s = randFunction(nranges=nranges[:], n=n, max_deg=max_deg) if random.randint(0, 1) else (PowSeries(lambda n : 1 if n == 0 else 0), [[], [], []])
    h = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    s = h.pprint()
    fin_ppr = connect(ppr[:], connect(pprint_s[:], connect([[" "], ["("], [" "]], connect(s, [[" "], [")"], [" "]]))))
    init_vals = [random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])]
    n = f * h
    fin_func = poly(solveDEseries(coeffs, n, init_vals, 100))
    return fin_func, strpprint(fin_ppr), init_vals

def random_parameterinc_curve(nranges=[1, 10], max_deg=2, dims=2):
    return [poly.rand(random.randint(0, max_deg), coeff_range=nranges[:]) for i in range(dims)]



def random_pfd(nrange=[1, 10], max_deg=2):
    z = []
    for j in range(max_deg):
        x = (-1)**random.randint(1, 2) * random.randint(nrange[0], nrange[1])
        while x in z:
            x = (-1)**random.randint(1, 2) * random.randint(nrange[0], nrange[1])
        z.append(x)
    z.sort()
    z.reverse()
    q = [poly([j, 1]) for j in z]
    a = 1
    for j in q:
        a *= j
    v = [random.randint(nrange[0], nrange[1]) * ((-1)**random.randint(1,2)) for j in range(len(q))]
    p = poly([0])
    for j in range(len(q)):
        pol_arr = q[:j] + q[j+1:] if j < len(q) - 1 else q[:j]
        r = poly([1])
        for k in pol_arr:
            r *= k
        p += r*v[j]
    str1 = str(p)
    str2 = str(a)
    str1cpy = str1[:]
    str2cpy = str2[:]
    len_measure1 = len(str1cpy.split("\n")[0])
    len_measure2 = len(str2cpy.split("\n")[0])
    str3 = "".join(["-" for j in range(max(len_measure1, len_measure2))])
    return [p, q, "\n".join([str1, str3, str2])]
