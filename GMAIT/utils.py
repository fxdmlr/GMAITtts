import math
import random
import cmath
import decimal


DEFAULT_TAYLOR_N = 1000
def heaviside(t):
    if t > 0:
        return 1
    else:
        return 0
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))


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
    if isinstance(x, complex):
        return 1
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
    return "\n".join(arr) 

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
            k = array[0][i] * det(minor(array, [0, i])) * (-1)**(i)
            a.append(k.simplify() if hasattr(k, 'simplify') else k)
        
        z = a[0]
        
        for i in range(1, len(a)):
            z+=a[i]
        
        return z

def numericIntegration(function, c, d, dx=0.0001):
    s = 0
    a = min(c, d)
    b = max(c, d)
    i = a+dx
    while i <= b - dx:
        s += (function(i) + function(i+dx))*dx/2
        i += dx
    return s * sgn(d - c)

def numericDiff(function, x, dx=0.0001):
    return (function(x+dx) - function(x-dx))* (1/(2*dx))

def nDiff(function, n, x, dx=0.0001):
    if n == 0:
        return function(x)
    if n == 1:
        return numericDiff(function, x, dx=dx)
    else:
        return nDiff(lambda t : numericDiff(function, t, dx=dx), n - 1, x, dx=dx)

def cndiff(function, z, dz=complex(0.0001, 0.0001)):
    return (function(z + dz) - function(z))/dz

def cnint(function, path, start, end, dt=0.0001):
    s = complex(0, 0)
    np_real = lambda t : path(t).real
    np_imag = lambda t : path(t).imag
    path_diff = lambda t: complex(numericDiff(np_real, t), numericDiff(np_imag, t))
    i = start
    while i < end:
        s += function(path(i)) * path_diff(i) * dt
        i += dt
    
    return s



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
    def __init__(self, n, ndigits=4):
        self.n = n
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def pprint(self):
        a  = [[" " for i in str(self.n)], [i for i in str(self.n)], [" " for i in str(self.n)]]
        po = [[" "], ["("], [" "]]
        pc = [[" "], [")"], [" "]]
        return a[:]
    
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
    def rand(nrange=[1, 10], ndigits=4):
        return integer(random.randint(nrange[0], nrange[1]))
        
class rational:
    def __init__(self, num, ndigits=4):
        #num = [p, q] -> number = p / q
        self.num = num[:]
        try:
            self.n = self.num[0] / self.num[1]
        except:
            self.n = 1
        
        #self.n = truncate(self.n, ndigits)
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
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
        po = [[" "], ["("], [" "]]
        pc = [[" "], [")"], [" "]]
        return lines[:]
        
    
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
    def rand(nrange=[1000, 10000], ndigits=4):
        return rational([random.randint(nrange[0], nrange[1]), random.randint(nrange[0], nrange[1])], ndigits=ndigits)
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
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
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
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        if len(self.coeffs) == 0:
            return 0
        s = self.coeffs[0]
        if isinstance(x, matrix):
            s = self.coeffs[0] * matrix.ones(len(x.array[:]))
        for i in range(1, self.deg + 1):
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
        return strpprint(self.npprint())
    
    def pprint(self, prev_pprint=[[" "], ["x"], [" "]]):
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
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        new_array = self.coeffs[:]
        new_array.reverse()
        lines = [[], [], [], [], [], [], []]
        right_pr = [[" "], [" "], [" "], ["("], [" "], [" "], [" "]]
        left_pr = [[" "], [" "], [" "], [")"], [" "], [" "], [" "]]
        nppr = prev_ppr[:]
        if len(prev_ppr[0]) > 1:
            nppr = connect(right_pr, connect(prev_ppr[:], left_pr))
        
        for i in range(len(new_array)):
            pow, coeff_abs, sgn_ppr = self.deg - i, abs(new_array[i].real), [[" "], [" "], [" "], [("+" if i != 0 else " ") if sgn(new_array[i].real) else "-"], [" "], [" "], [" "]]
            if new_array[i].imag != 0:
                coeff_abs = new_array[i]
                sgn_ppr =  [[" "], [" "], [" "], ["+"], [" "], [" "], [" "]]
            coeff_abs_ppr = [[" " for j in str(coeff_abs)],
                             [" " for j in str(coeff_abs)],
                             [" " for j in str(coeff_abs)],
                             [j for j in str(coeff_abs)],
                             [" " for j in str(coeff_abs)],
                             [" " for j in str(coeff_abs)],
                             [" " for j in str(coeff_abs)]]
            pow_ppr = [[" " for j in str(pow)],
                       [" " for j in str(pow)],
                       [j for j in str(pow)],
                       [" " for j in str(pow)],
                       [" " for j in str(pow)],
                       [" " for j in str(pow)],
                       [" " for j in str(pow)]]
            if pow == 1:
                pow_ppr = [[], [], [], [], [], [], []]
            if coeff_abs == 1 and pow != 0:
                coeff_abs_ppr = [[], [], [], [], [], [], []]
            if coeff_abs == 0:
                continue
            else:
                if pow != 0:
                    lines = connect(lines[:], connect(sgn_ppr[:], connect(coeff_abs_ppr[:], connect(nppr[:], pow_ppr[:]))))[:]
                else:
                    lines = connect(lines[:], connect(sgn_ppr[:], coeff_abs_ppr[:]))[:]
        return lines[:]
    
    def texify(self, prev_tex = 'x'):
        s = ""
        for i in range(len(self.coeffs) - 1, -1, -1):
            if self.coeffs[i] != 0:
                n = str(abs(self.coeffs[i])) if abs(self.coeffs[i]) != 1 else ""
                if abs(self.coeffs[i]) == 1 and i == 0:
                    n = '1'
                s += "-" if self.coeffs[i] < 0 else ("+" if i != len(self.coeffs) - 1 else "")
                s += n + prev_tex + "^{" if i > 1 else (n + prev_tex if i == 1 else n) 
                s += str(i) + "}" if i > 1 else ""
        
        return s
                 
    
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
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
        
        if isinstance(other, (int, float, complex)):
            x = [other * i for i in self.coeffs[:]]
            return poly(x)
        
        elif isinstance(other, poly):
            arr = [0 for i in range(self.deg + other.deg + 1)]
            for i in range(len(self.coeffs)):
                for j in range(len(other.coeffs)):
                    arr[i + j] += self.coeffs[i] * other.coeffs[j]
            
            return poly(arr[:])
        
        elif isinstance(other, (Div, Prod, Sum)):
            return other * self
    
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

    
    def diff(self, wrt=0):
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
    
    def roots(self, prevs=[]):
        zeros = 0
        i = 0
        while self.coeffs[i] == 0 and i < len(self.coeffs):
            zeros += 1
            i += 1
        if zeros > 0:
            new_p = poly(self.coeffs[zeros:])
            return new_p.roots(prevs=[0 for i in range(zeros)])
        
        if self.deg == 1:
            return [-self.coeffs[0] / self.coeffs[1]] + prevs[:]
        
        if self.deg == 2:
            a = self.coeffs[2]
            b = self.coeffs[1]
            c = self.coeffs[0]
            d = b**2 - 4*a*c
            return [(-b + cmath.sqrt(d))/(2*a), (-b - cmath.sqrt(d))/(2*a)] + prevs[:]
        
        if self.deg == 3:
            a = self.coeffs[3]
            b = self.coeffs[2]
            c = self.coeffs[1]
            d = self.coeffs[0]
            
            d0 = b**2 - 3*a*c
            d1 = 2*b**3-9*a*b*c+27*d*a**2
            if d0 == d1 == 0:
                r1 = (-1/(3*a)) * (b)
                r2, r3 = (self / poly([-r1, 1])).roots()
                return [r1, r2, r3] + prevs[:]
                
            C = ((d1 + cmath.sqrt(d1**2-4*d0**3)) / 2)**(1/3)
            if C == 0:
                C = ((d1 - cmath.sqrt(d1**2-4*d0**3)) / 2)**(1/3)
            
            r1 = (-1/(3*a)) * (b + C + d0/C)
            r2, r3 = (self / poly([-r1, 1])).roots()
            
            return [r1, r2, r3] + prevs[:]
        
        if self.deg == 4:
            A = self.coeffs[4]
            B = self.coeffs[3]
            C = self.coeffs[2]
            D = self.coeffs[1]
            E = self.coeffs[0]
            a = -3*B**2/(8*A**2) + C/A
            b =  B**3/(8*A**3) - B*C/(2*A**2) + D/A
            c = -3*B**4/(256*A**4)+ C*B**2/(16*A**3) - B*D/(4*A**2)+ E/A
            d = -B / (4*A)
            #depressed_quartic = [c, b, a, 0, 1]
            if b == 0:
                r1 = cmath.sqrt(-a/2 + 0.5*cmath.sqrt(a**2-4*c))
                r2 = cmath.sqrt(-a/2 - 0.5*cmath.sqrt(a**2-4*c))
                r3 = -cmath.sqrt(-a/2 + 0.5*cmath.sqrt(a**2-4*c))
                r4 = -cmath.sqrt(-a/2 - 0.5*cmath.sqrt(a**2-4*c))
                
                return [r1+d, r2+d, r3+d, r4+d] + prevs[:]
            else:
                p = -a**2/12 - c
                q = -a**3/108+a*c/3-b**2/8
                w = (-q/2 + cmath.sqrt(q**2/4 + p**3/27))**(1/3)
                if w != 0:
                    y = a/6 + w-p/(3*w)
                else:
                    y = a/6
                r1 = 0.5*(-cmath.sqrt(2*y - a) + cmath.sqrt(-2*y-a+2*b/cmath.sqrt(2*y - a)))
                r2 = 0.5*(-cmath.sqrt(2*y - a) - cmath.sqrt(-2*y-a+2*b/cmath.sqrt(2*y - a)))
                r3 = 0.5*(cmath.sqrt(2*y - a) + cmath.sqrt(-2*y-a-2*b/cmath.sqrt(2*y - a)))
                r4 = 0.5*(cmath.sqrt(2*y - a) - cmath.sqrt(-2*y-a-2*b/cmath.sqrt(2*y - a)))
                return [r1+d, r2+d, r3+d, r4+d] + prevs[:]

        else:
            guess = complex(0, 0)
            max_iter = 100
            current_root = poly.newtonsmethod(self, guess, max_iter)
            new_poly = self / poly([-current_root, 1])
            return new_poly.roots(prevs=prevs[:]+[current_root])
            
            
    
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
    def rand(deg, coeff_range = [0, 10], sgn_sensitive=1):
        if sgn_sensitive:
            coeffs = [(-1)**random.randint(1, 10) * random.randint(coeff_range[0], coeff_range[1]) for i in range(deg + 1)]
        else:
            coeffs = [random.randint(coeff_range[0], coeff_range[1]) for i in range(deg + 1)]
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
            #if pl.diff()(x_i) == 0:
            #    x_i += 0.05
            x_i -= pl(x_i) / pl.diff()(x_i)
            if not isinstance(x_i, complex) : 
                x_i = round(x_i, ndigits=10)
            else:
                x_i = complex(round(x_i.real, ndigits=10), round(x_i.imag, ndigits=10))
            if x_i == x_ig:
                break
        
        return x_i


class PowSeries:
    def __init__(self, c_n, name=None):
        self.function = c_n
        self.name = name
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def poly(self, n):
        return poly([self.function(i) for i in range(n + 1)])
    
    def __call__(self, x, n=DEFAULT_TAYLOR_N):
        return self.poly(n)(x)

    def diff(self, wrt=0):
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
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def identity(self):
        arr = [[1 if i == j else 0 for j in range(len(self.array[i]))] for i in range(len(self.array))]
        return matrix(arr)
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
    
    def pprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        if self.name is not None:
            return [[" " for i in range(len(self.name))], [l for l in self.name], [" " for i in range(len(self.name))]]
        tot_cells = []
        for i in self.array:
            cells = []
            for j in i:
                lines = [[], [], []]
                if hasattr(j, 'npprint'):
                    lines = connect(lines, j.npprint(prev_ppr=prev_ppr[:])[2:5])
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
        op = [[["/"], ["|"], ["|"]]] + [[["|"], ["|"], ["|"]] for i in range((len(tot_lines) - 2)*heaviside(len(tot_lines) - 2))] + [[["|"], ["|"], ["\\"]]]
        clsd = [[["\\"], ["|"], ["|"]]] + [[["|"], ["|"], ["|"]] for i in range((len(tot_lines) - 2)*heaviside(len(tot_lines) - 2))] + [[["|"], ["|"], ["/"]]]
        new_lines = []
        for i in range(len(tot_lines)):
            #print(lines[i])
            new_lines.append(connect(op[i], connect(tot_lines[i], clsd[i])))
        return new_lines#tot_lines[:]
    
    def texify(self, prev_tex='x'):
        s = '\\begin{bmatrix}'
        for i in self.array:
            for j in i:
                if hasattr(j, 'texify'):
                    s += j.texify()+' &'
                else:
                    s += str(j)+' &'
            s = s[:-1]
            s += '\\\\'
        s += '\\end{bmatrix}'
        return s
    
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
        if isinstance(other, (int, float, poly, Div, Sum, Prod)):
            x = self.array[:]
            for i in range(len(x)):
                for j in range(len(x[i])):
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
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def det(self):
        return det(self.array[:])
    
    def __eq__(self, other):
        return self.array[:] == other.array[:]
    
    def __pow__(self, other):
        if isinstance(other, int):
            mat = matrix([[1 if i == j else 0 for j in range(len(self.array[i]))] for i in range(len(self.array))])
            for i in range(other):
                mat *= self
            
            return mat
    
    def charpoly(self):
        new_arr = [[j for j in i] for i in self.array[:]]
        for i in range(len(self.array)):
            new_arr[i][i] = poly([float(new_arr[i][i]), -1])
        
        return matrix(new_arr[:]).det()
    
    def eigenvalue(self):
        return [i for i in self.charpoly().roots()[:]]
    
    def transpose(self):
        new_array = []
        for i in range(len(self.array[0])):
            curr = []
            for j in range(len(self.array[:])):
                curr.append(self.array[j][i])
            new_array += [curr[:]]
        
        return matrix(new_array[:])
    
    def inverse(self):
        d = self.det()
        n = len(self.array[:])
        if det == 0:
            raise Exception(ValueError, 'Determinant is zero.')
        if isinstance(d, (int, float)):
            new_array = [[det(minor(self.array[:], [i, j])) / d * (-1)**(i + j) for i in range(n)] for j in range(n)]
        else:
            new_array = [[Div([det(minor(self.array[:], [i, j])) , d * (-1)**(i + j)]) for i in range(n)] for j in range(n)]
        return matrix(new_array[:])
        
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
    
    @staticmethod
    def ones(dim):
        arr = [[int(i == j) for j in range(dim)] for i in range(dim)]
        return matrix(arr)

class vect:
    def __init__(self, array):
        self.array = array[:]
        self.dim = len(array)
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
        
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return vect([i + other for i in self.array[:]])
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
    
    def diff(self, wrt=0):
        return vect([i.diff(wrt=wrt) if hasattr(i, 'diff') else 0 for i in self.array[:]])
    
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    
class pcurve:
    def __init__(self, farr):
        self.array = farr[:]
        self.dim = len(farr)
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
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
    
    def diff(self, wrt=0):
        a = []
        for i in self.array:
            if hasattr(i, 'diff'):
                a.append(i.diff(wrt=wrt))
        
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
    
    def curvature(self, wrt=0):
        return lambda x : self.diff(wrt=wrt)(x).cross(self.diff(wrt=wrt).diff(wrt=wrt)(x)).length()/self.diff(wrt=wrt).length()(x)**3
    
    def T(self, wrt=0):
        return lambda x : self.diff(wrt=wrt)(x) * (1/self.diff(wrt=wrt).length()(x))
    
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
        self.x_ppr = [[" "], ["x"] , [" "]]
        self.y_ppr = [[" "], ["y"] , [" "]]
        self.z_ppr = [[" "], ["z"] , [" "]]
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])

    def __call__(self, *args):
        s = 0
        if len(args) == 1:
            if isinstance(args[0], list):
                if len(args[0]) == 1:
                    x = args[0][0]
                    y, z = 0, 0
                elif len(args[0]) == 2:
                    x, y = args[0][:]
                    z = 0
                elif len(args[0]) == 3:
                    x, y, z = args[0][:]
                    
        elif len(args) == 2:
            x, y = args[:]
            z = 0
        elif len(args) == 3:
            x, y, z = args[:]
        
        for i in range(len(self.array)):
            for j in range(len(self.array)):
                for k in range(len(self.array)):
                    s += self.array[i][j][k] * (x ** i) * (y ** j) * (z ** k)
        
        return s
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            new_array = [[[k for k in j]for j in i] for i in self.array[:]]
            new_array[0][0][0] += other
            return polymvar(new_array[:])
        
        elif isinstance(other, polymvar):
            if len(self.array) >= len(other.array):
                new_array = [[[k for k in j]for j in i] for i in self.array[:]]  
            else:
                new_array = [[[k for k in j]for j in i] for i in other.array[:]]
            if len(self.array) >= len(other.array):
                m_arr =  [[[k for k in j]for j in i] for i in other.array[:]]  
            else:
                m_arr = [[[k for k in j]for j in i] for i in self.array[:]]
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
            new_array = [[[k for k in j]for j in i] for i in self.array[:]]
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
        first_term = True
        for i in range(len(new_array)):
            for j in range(len(new_array)):
                for k in range(len(new_array)):
            
                    temp_lines1 = [[" "], ["+" if sgn(new_array[i][j][k]) else "-"], [" "]]
                    if first_term:
                        temp_lines1 = [[" "], [" " if sgn(new_array[i][j][k]) else "-"], [" "]]
                    if new_array[i][j][k] == 0:
                        continue
                    first_term = False
                    temp_lines2 = [[], [], []]
                    if abs(new_array[i][j][k]) != 1:
                        sarr = [[" " for l in range(len(str(abs(new_array[i][j][k]))))], [l for l in str(abs(new_array[i][j][k]))], [" " for l in range(len(str(abs(new_array[i][j][k]))))]]
                    else:
                        sarr = [[], [], []]
                    #nsubarr = connect(sub_arr[:], [[" "] + [l for l in str(i)] + [" "] + [l for l in str(j)] + [" "] + [l for l in str(k)], ["x"] + [" " for l in range(len(str(i)))] + ["y"] + [" " for l in range(len(str(j)))] + ["z"] + [" " for l in range(len(str(k)))], [" " for l in range(3 + len(str(i)) + len(str(j)) + len(str(k)))]])
                    
                    if i != 0:
                        z = [[" "] + [l for l in str(i)], ["x"] + [" " for l in range(len(str(i)))], [" " for l in range(1 + len(str(i)))]]
                        if i == 1:
                            z = self.x_ppr[:]
                        sarr = connect(sarr, z)[:]
                    if j != 0:
                        z = [[" "] + [l for l in str(j)], ["y"] + [" " for l in range(len(str(j)))], [" " for l in range(1 + len(str(j)))]]
                        if j == 1:
                            z = self.y_ppr[:]
                        sarr = connect(sarr, z)[:]
                    if k != 0:
                        z = [[" "] + [l for l in str(k)], ["z"] + [" " for l in range(len(str(k)))], [" " for l in range(1 + len(str(k)))]]
                        if k == 1:
                            z = self.z_ppr[:]
                        sarr = connect(sarr, z)[:]                    
                    lines = connect(lines[:], connect(temp_lines1[:], sarr[:]))
        
        return lines[:]
    
    def npprint(self, prev_ppr = None):
        new_array = self.pprint()
        line = [" " for i in new_array[0]]
        return [line[:], line[:]] + new_array[:] + [line[:], line[:]]
    
    def __str__(self):
        return strpprint(self.pprint())
    
    def diff(self, wrt = 0):
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
        
    def c_integrate(self, curve, t0, t1, wrt=0):
        new_f = self(curve.array[0], curve.array[1], curve.array[2])
        f = lambda x : new_f(x) * curve.diff(wrt = wrt).length()(x)
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
    
    def evalArray(self, array):
        if len(array) < 3:
            narray = array[:]
            for i in range(3-len(array)):
                narray += [0]
            array = narray[:]
                
        x, y, z = array[:]
        s = 0
        for i in range(len(self.array)):
            for j in range(len(self.array)):
                for k in range(len(self.array)):
                    s += self.array[i][j][k] * (x ** i) * (y ** j) * (z ** k)
        
        return s
       
    
    def grad(self):
        return vect([self.diff(0), self.diff(1), self.diff(2)])
    
    def delvar(self, var):
        new_array = [[self.array[i][j] for j in range(len(self.array))] for i in range(len(self.array))]
        if var == 0:
            for i in range(1, len(self.array)):
                for j in range(len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][j][k] = 0
            
            return polymvar(new_array)
        
        if var == 1:
            for i in range(len(self.array)):
                for j in range(1, len(self.array)):
                    for k in range(len(self.array)):
                        new_array[i][j][k] = 0
            
            return polymvar(new_array)
        
        if var == 2:
            for i in range(len(self.array)):
                for j in range(len(self.array)):
                    for k in range(1, len(self.array)):
                        new_array[i][j][k] = 0
            
            return polymvar(new_array)

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
    @staticmethod
    def x():
        zero_pol = [[[0 for j in range(2)] for i in range(2)]for k in range(2)]
        zero_pol[1][0][0] = 1
        return polymvar(zero_pol[:])
    
    @staticmethod
    def y():
        zero_pol = [[[0 for j in range(2)] for i in range(2)]for k in range(2)]
        zero_pol[0][1][0] = 1
        return polymvar(zero_pol[:])
    
    @staticmethod
    def z():
        zero_pol = [[[0 for j in range(2)] for i in range(2)]for k in range(2)]
        zero_pol[0][0][1] = 1
        return polymvar(zero_pol[:])

class vectF:
    def __init__(self, array):
        self.array = array[:]
        self.dim = len(array)
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
        
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
        
class Sum:
    def __new__(cls, array):
        arr = []
        for i in array:
            if i == 0 or i == poly([0]):
                continue
            
            
            arr.append(i)
        '''
        if len(arr) == 0:
            return 0
        '''
        if len(arr) == 1:
            #return type(array[0]).__new__(**array[0].user_args) 
            return arr[0]
        else:
            return super(Sum, cls).__new__(cls)
    def __init__(self, array):
        self.arr = []
        for i in array:
            if i == 0 or i == poly([0]):
                continue
            if isinstance(i, Sum):
                self.arr[:] += i.arr[:]
                
                continue
            self.arr.append(i)
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
        
    def simplify(self):
            
        narr = []
        for i in self.arr:
            if i == 0 or i == poly([0]):
                continue
            if hasattr(i, 'simplify'):
                narr.append(i.simplify())
                continue
            narr.append(i)
        
        new_array = []
        types_array = []
        for i in narr:
            if isinstance(i, (int, float)):
                i = poly([i])
            if type(i) not in types_array:
                types_array.append(type(i))
                new_array.append(i)
            
            else:
                new_array[types_array.index(type(i))] += i
        
        
        #self.arr[:] = new_array[:]
        return Sum(new_array[:])
    
    def __call__(self, x):
        s = 0
        for i in self.arr:
            if not callable(i):
                s += i
            else:
                s += i(x)
        
        return s
    
    def __add__(self, other):
        if isinstance(other, Sum):
            new_arr  = self.arr[:] + other.arr[:]
        elif isinstance(other, (int, float)):
            new_arr  = self.arr[:] + [other]
        else:
            new_arr = self.arr[:] + [other]  
        
        return Sum(new_arr[:])
    
    def __mul__(self, other):
        if isinstance(other, Sum):
            new_arr  = []
            for i in self.arr[:]:
                for j in other.arr[:]:
                    new_arr.append(Prod([i, j]))
            
            return Sum(new_arr[:])
        
        elif isinstance(other, (int, float)):
            return Sum([other * i for i in self.arr[:]])
        
        elif isinstance(other, Prod):
            return Sum([Prod([i] + other.arr[:]) for i in self.arr[:]])
        
        else:
            return Prod([self, other])
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        array = [[], [], [], [], [], [], []] 
        if len(self.arr) == 0:
            pass
        elif hasattr(self.arr[0], 'npprint'):
                array[:] = connect(array[:], self.arr[0].npprint(prev_ppr=prev_ppr[:])[:])[:] 
        elif isinstance(self.arr[0], (int, float)):
            arg = [[" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [j for j in str(self.arr[0])], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))]]
            array[:] = connect(array[:], arg[:])[:]
        for i in self.arr[1:]:
            if hasattr(i, 'npprint'):
                array[:] = connect(array[:], connect([[" "], [" "], [" "], ["+"], [" "], [" "], [" "]], i.npprint(prev_ppr=prev_ppr[:])[:]))[:] 
            elif isinstance(i, (int, float)):
                arg = [[" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))], [j for j in str(i)], [" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))]]
                array[:] = connect(array[:], connect([[" "], [" "], [" "], ["+"], [" "], [" "], [" "]], arg[:]))[:]
        return array
    
    def texify(self, prev_tex='x'):
        s = ""
        arr = []
        for i in self.arr[:]:
            if hasattr(i, "texify"):
                arr.append(i.texify())
            else:
                arr.append(str(i))
        return "+".join(arr)
    
    def diff(self, wrt=0):
        return Sum([i.diff(wrt=wrt) if hasattr(i, 'diff') else 0 for i in self.arr[:]]) 
    
    def __str__(self):
        return strpprint(self.npprint())
    
    def newtonsmethod(self, starting_point = 0.1, n = 1000):
        x_n = starting_point
        df = self.diff()
        for i in range(n):
            x_n -= self(x_n) / df(x_n)
        
        return x_n
    
    __rmul__ = __mul__
    __radd__ = __add__

class Prod:
    def __new__(cls, array):
        arr = []
        for i in array:
            if i == 1 or i == poly([1]):
                continue
            arr.append(i)
        if len(arr) == 1:
            #return type(array[0]).__new__(**array[0].user_args) 
            return arr[0]
        
        
        if 0 in arr or poly([0]) in arr:
            return 0
        else:
            return super(Prod, cls).__new__(cls)
    def __init__(self, array):
        self.arr = []
        for i in array:
            if i == 1 or i == poly([1]):
                continue
            if isinstance(i, Prod):
                self.arr[:] += i.arr[:]
                
                continue
            self.arr.append(i)
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
        

    def simplify(self):
        narr = []
        for i in self.arr:
            if i == 1 or i == poly([1]):
                continue
            
            if hasattr(i, 'simplify'):
                narr.append(i.simplify())
                continue
            
            narr.append(i)
        
        
        new_array = []
        types_array = []
        for i in narr:
            if isinstance(i, (int, float)):
                i = poly([i])
            if type(i) not in types_array:
                types_array.append(type(i))
                new_array.append(i)
            
            else:
                new_array[types_array.index(type(i))] *= i

        #self.arr[:] = new_array[:]
        return Prod(new_array[:])
    def __call__(self, x):
        s = 1
        for i in self.arr:
            if not callable(i):
                s *= i
            else:
                s *= i(x)
        
        return s
    
    def __add__(self, other):
        if isinstance(other, Sum):
            new_arr  = [self] + other.arr[:]
        
        else:
            new_arr = [self, other] 
        
        return Sum(new_arr[:])
    
    def __mul__(self, other):
        if isinstance(other, Prod):
            new_arr  = Prod(self.arr[:] + other.arr[:])
        
        elif isinstance(other, (int, float)):
            new_arr  = Prod(self.arr[:] + [other])
        
        elif isinstance(other, Sum):
            new_arr = Sum([Prod(self.arr[:] + [i]) for i in other.arr[:]])
        
        else:
            new_arr = Prod(self.arr[:] + [other])
            
        
        return new_arr
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        array = [[], [], [], [], [], [], []] 
        spaces = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        op = [[" "], [" "], [" "], ["("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [")"], [" "], [" "], [" "]]
        op_inv = [[" "], [" "], [" "], ["("], [" "], [" "], [" "]]
        clsd_inv = [[" "], [" "], [" "], [")"], [" "], [" "], [" "]]
        if hasattr(self.arr[0], 'npprint'):
                array[:] = connect(array[:], connect(op, connect(self.arr[0].npprint(prev_ppr=prev_ppr)[:], clsd)))[:]  
        elif isinstance(self.arr[0], (int, float)):
            arg = [[" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [j for j in str(self.arr[0])], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))]]
            if sgn(self.arr[0]):
                array[:] = connect(array[:], arg[:])[:] 
            else:
                array[:] = connect(array[:], connect(op, connect(arg[:], clsd)))[:]
        for i in self.arr[1:]:
            array[:] = connect(array[:], spaces[:])[:]
            if hasattr(i, 'npprint'):
                array[:] = connect(array[:], connect(op, connect(i.npprint(prev_ppr=prev_ppr)[:], clsd)))[:] 
            elif isinstance(i, (int, float)):
                arg = [[" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))], [j for j in str(i)], [" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))], [" " for j in range(len(str(i)))]]
                if sgn(i):
                    array[:] = connect(array[:], arg[:])[:] 
                else:
                    array[:] = connect(array[:], connect(op, connect(arg[:], clsd)))[:]
        return array
    
    def texify(self, prev_tex='x'):
        s = ""
        for i in self.arr[:]:
            if hasattr(i, 'texify'):
                s += "("+i.texify(prev_tex=prev_tex)+")"
            else:
                s += str(i)
        
        return s
    
    def diff(self, wrt=0):
        return Sum([Prod(self.arr[:i] + [self.arr[i].diff(wrt=wrt) if hasattr(self.arr[i], "diff") else 0] + self.arr[i+1:]) for i in range(len(self.arr[:]))])
    
    def __str__(self):
        return strpprint(self.npprint())   
    
    def newtonsmethod(self, starting_point = 0.1, n = 1000):
        x_n = starting_point
        df = self.diff()
        for i in range(n):
            x_n -= self(x_n) / df(x_n)
        
        return x_n 
    
    __rmul__ = __mul__
    __radd__ = __add__
class Comp:
    def __init__(self, array):
        self.arr = array[:]
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def simplify(self):
        narr = [i.simplify() if hasattr(i, 'simplify') else i for i in self.arr[:]]
        #self.arr[:] = narr[:]
        return Comp(narr)
    
    def __call__(self, x):
        s = self.arr[0](x)
        for i in self.arr[1:]:
            if not callable(i):
                s = i
            else:
                k = s
                s = i(k)
        
        return s
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
    
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        array = prev_ppr[:]
        for i in self.arr[:]:
            if hasattr(i, 'npprint'):
                array[:] = i.npprint(prev_ppr=array[:])[:]
            else:
                arg = [[" " for j in range(len(str(i)))], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [j for j in str(self.arr[0])], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))], [" " for j in range(len(str(self.arr[0])))]]
                array[:] = connect(array[:], arg[:])[:]
        
        return array[:]
    
    def texify(self, prev_tex='x'):
        s = prev_tex[:]
        for i in self.arr[:]:
            if hasattr(i, 'texify'):
                s = i.texify(prev_tex=s)[:]
            else:
                s += str(self.arr[0])
        
        return s
        
    def diff(self, wrt=0):
        array = []
        for i in range(len(self.arr[:])):
            if i == 0:
                array.append(self.arr[i].diff(wrt=wrt))
            else:
                array.append(Comp(self.arr[:i] + [self.arr[i].diff(wrt=wrt) if hasattr(self.arr[i], 'diff') else 0]))
        
        return Prod(array[:])
    
    def __str__(self):
        return strpprint(self.npprint())
    
    def newtonsmethod(self, starting_point = 0.1, n = 1000):
        x_n = starting_point
        df = self.diff()
        for i in range(n):
            x_n -= self(x_n) / df(x_n)
        
        return x_n
    
    __rmul__ = __mul__
    __radd__ = __add__

class Div:
    def __init__(self, array):
        self.arr = array[:]
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def simplify(self):
        a = self.arr[0]
        b = self.arr[1]
        if hasattr(self.arr[0], 'simplify'):
            a = self.arr[0].simplify()
        if hasattr(self.arr[1], 'simplify'):
            b = self.arr[1].simplify()
        
        return Div([a, b])
    
    def __call__(self, x):
        n1 = self.arr[0](x) if callable(self.arr[0]) else self.arr[0]
        n2 = self.arr[1](x) if callable(self.arr[1]) else self.arr[1]
        return n1 / n2
    
    def __add__(self, other):
        if not isinstance(other, Div):
            return Div([Sum([self.arr[0], Prod([other, self.arr[1]])]), self.arr[1]])
        
        else:
            return Div([Sum([Prod([self.arr[0], other.arr[1]]), Prod([self.arr[1], other.arr[0]])]), Prod([self.arr[1], other.arr[1]])])
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Div([self.arr[0] * other, self.arr[1]])
        if not isinstance(other, Div):
            return Div([Prod([self.arr[0], other]), self.arr[1]])
        else:
            return Div([Prod([self.arr[0], other.arr[0]]), Prod([self.arr[1], other.arr[1]])])
    
    def __truediv__(self, other):
        return Div([self.arr[0], Prod([self.arr[1], other])])
    
    def inv(self):
        return Div([self.arr[1], self.arr[0]])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        z1 = self.arr[0].npprint(prev_ppr=prev_ppr[:])[:] if hasattr(self.arr[0], 'npprint') else poly([self.arr[0]]).npprint(prev_ppr=prev_ppr[:])[:]
        z2 = self.arr[1].npprint(prev_ppr=prev_ppr[:])[:] if hasattr(self.arr[1], 'npprint') else poly([self.arr[1]]).npprint(prev_ppr=prev_ppr[:])[:]
        max_len = max(len(z1[0]), len(z2[0]))
        min_len = min(len(z1[0]), len(z2[0]))
        s = int((max_len - min_len)/2)
        spaces = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        arr1 = [[], [], [], [], [], [], []]
        arr2 = [[], [], [], [], [], [], []]
        for i in range(s):
            arr1[:] = connect(arr1[:], spaces[:])
        for i in range(max_len - min_len - s):
            arr2[:] = connect(arr2[:], spaces[:])
        div_sign = ["-" for i in range(max(len(z1[0]), len(z2[0])))]
        if len(z1[0]) < len(z2[0]):
            new_ppr = connect(arr1, connect(z1, arr2))[1:4] + [div_sign[:]] + z2[1:4]
        
        if len(z1[0]) >= len(z2[0]):
            new_ppr = z1[1:4] + [div_sign[:]] + connect(arr1, connect(z2, arr2))[1:4] 
        
        return new_ppr[:]
    
    def texify(self, prev_tex='x'):
        a = self.arr[0].texify(prev_tex=prev_tex[:]) if hasattr(self.arr[0], 'texify') else str(self.arr[0])
        b = self.arr[1].texify(prev_tex=prev_tex[:]) if hasattr(self.arr[1], 'texify') else str(self.arr[1])
        return '\\frac{%s}{%s}'%(a, b)
    
    def diff(self, wrt=0):
        n1 = Sum([Prod([self.arr[0].diff(wrt=wrt) if hasattr(self.arr[0], 'diff') else 0, self.arr[1]]), Prod([self.arr[1].diff(wrt=wrt)if hasattr(self.arr[1], 'diff') else 0, self.arr[0], -1])])
        n2 = Prod([self.arr[1], self.arr[1]])
        return Div([n1, n2])
    
    def __str__(self):
        return strpprint(self.npprint())
    
    def __neg__(self):
        return Div([-self.arr[0], self.arr[1]])
    
    def __sub__(self, other):
        return self + (-other)
    
    def newtonsmethod(self, starting_point = 0.1, n = 1000):
        x_n = starting_point
        df = self.diff()
        for i in range(n):
            x_n -= self(x_n) / df(x_n)
        
        return x_n
    
    __rmul__ = __mul__
    __radd__ = __add__


class sin:
    def __init__(self):
        self.function = cmath.sin
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "], 
             ["s", "i", "n"],
             [" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    
    def texify(self, prev_tex="x"):
        return '\\sin (%s)'%prev_tex[:]
    
    def diff(self, wrt=0):
        return cos()

class cos:
    def __init__(self):
        self.function = cmath.cos
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "], 
             ["c", "o", "s"],
             [" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\cos (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return Comp([sin(), poly([0, -1])])


class tan:
    def __init__(self):
        self.function = cmath.tan
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "], 
             ["t", "a", "n"],
             [" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\tan (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return Sum([1, Comp([tan(), poly([0, 0, 1])])]) 
     
class inv:
    def __init__(self):
        self.function = lambda x : 1 / x
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        div_sign = ["-" for i in prev_ppr[0]]
        one_ind = int(len(prev_ppr[0]) / 2)
        one = [" " if i != one_ind else "1" for i in range(len(prev_ppr[0]))]
        new_ppr = [[" " for i in prev_ppr[0]], [" " for i in prev_ppr[0]]] + [one[:]] + [div_sign[:]] + prev_ppr[1:4]
        return new_ppr[:]
        
    def texify(self, prev_tex="x"):
        return 'frac{1}{%s}'%prev_tex[:]
    
    def diff(self, wrt=0):
        return Comp([poly([0, 0, -1]), inv()])
    
class sqrt:
    def __init__(self):
        self.function = cmath.sqrt
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        rad = [[" ", " "],
               [" ", " "],
               [" ", " /"],
               ["\\", "/"],
               [" ", " "],
               [" ", " "],
               [" ", " "]]
        new_ppr = prev_ppr[:]
        for i in range(1, len(prev_ppr[0])):
            new_ppr[1][i] = "_"
            
        return connect(rad, new_ppr[:])[:]
    def texify(self, prev_tex="x"):
        return '\\sqrt{%s}'%prev_tex[:]
    def diff(self, wrt=0):
        return Comp([Prod([2, sqrt()]), inv()])

class asin:
    def __init__(self):
        self.function = cmath.asin
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "], 
             ["a", "s", "i", "n"],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\sin^{-1} (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return Comp([poly([1, 0, -1]), sqrt(), inv()])

class atan:
    def __init__(self):
        self.function = cmath.atan
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "], 
             ["a", "t", "a", "n"],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\tan^{-1} (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return Comp([poly([1, 0, 1]), inv()])

class sinh:
    def __init__(self):
        self.function = cmath.sinh
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "], 
             ["s", "i", "n", "h"],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\sinh (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return cosh()

class cosh:
    def __init__(self):
        self.function = cmath.cosh
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "], 
             ["s", "i", "n", "h"],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\cosh (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return cosh

class tanh:
    def __init__(self):
        self.function = cmath.tanh
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "], 
             ["t", "a", "n", "h"],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "],
             [" ", " ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\tanh (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return Comp([tanh, poly([1, 0, -1])])
class log:
    def __init__(self):
        self.function = cmath.log
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "], 
             ["l", "o", "g"],
             [" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return '\\log (%s)'%prev_tex[:]
    def diff(self, wrt=0):
        return inv()

class exp:
    def __init__(self):
        self.function = cmath.exp
        self.user_args = dict([(key,val) for key,val in locals().items() if key!='self' and key!='__class__'])
    
    def __call__(self, x):
        return self.function(x)
    
    def __add__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other + self
        
        else:
            return Sum([self, other])
        
    def __mul__(self, other):
        if isinstance(other, (Prod, Sum)):
            return other * self
        
        else:
            return Prod([self, other])
    
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        op = [[" "], [" "], [" "], [" ("], [" "], [" "], [" "]]
        clsd = [[" "], [" "], [" "], [" )"], [" "], [" "], [" "]]
        if len(prev_ppr[0]) == 1:
            op = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
            clsd = [[" "], [" "], [" "], [" "], [" "], [" "], [" "]]
        s = [[" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "], 
             ["e", "x", "p"],
             [" ", " ", " "],
             [" ", " ", " "],
             [" ", " ", " "]]
        return connect(s[:], connect(op[:], connect(prev_ppr[:], clsd[:])))[:]
    def texify(self, prev_tex="x"):
        return 'e^{%s}'%prev_tex[:]
    def diff(self, wrt=0):
        return exp()

class Symbol:
    def __init__(self, string):
        self.string = string
        
    def npprint(self, prev_ppr=[[" "], [" "], [" "], ["x"], [" "], [" "], [" "]]):
        return [[" " for j in range(len(self.string))] for i in range(3)] + [[i for i in self.string]] + [[" " for j in range(len(self.string))] for i in range(3)]
    
    def __call__(self, x):
        return x
    
    def __add__(self, other):
        if isinstance(other, (int, poly, float)):
            return Sum([self, other])
        return Sum([self, other])
    
    def __mul__(self, other):
        if isinstance(other, (int, poly, float)):
            return Prod([self, other])
        return Prod([self, other])
    
    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.string == other.string
        return False
    
    def diff(self, wrt=0):
        return 1
    
    __rmul__ = __mul__
    __radd__ = __add__
    
asinh = Comp([Sum([poly([0, 1]), Comp([poly([1, 0, 1]), sqrt()])]), log()])
acosh = Comp([Sum([poly([0, 1]), Comp([poly([-1, 0, 1]), sqrt()])]), log()])
atanh = Sum([Comp([Comp([poly([1, 1]), sqrt()]), log()]), Prod([-1, Comp([Comp([poly([1, -1]), sqrt()]), log()])])])

class DummyFunc:
    def __init__(self, function, string, x, ndigits=4):
        self.function = function
        self.n = function(x)#truncate(function(x), ndigits)
        self.string = string[:]
        self.x = x
        s = self.string + "(" + str(self.x) + ")"
        self.ppr = [[" " for i in s], [i for i in s], [" " for i in s]]
    
    def pprint(self):
        return self.ppr[:]

class DummyFuncN:
    def __init__(self, n, ndigits=4):
        self.n = n
        self.real = self.n.real
        self.imag = self.n.imag
        self.string = str(n)
        s = self.string[:]
        self.ppr = [[" " for i in s], [" " for i in s], [" " for i in s], [i for i in s], [" " for i in s], [" " for i in s], [" " for i in s]]
    
    def npprint(self, prev_ppr = []):
        return self.ppr[:]
    
    def __call__(self, x):
        return self.n
    
    def __add__(self, other):
        return Sum([self, other])
    
    def __mul__(self, other):
        return Prod([self, other])
    
    def __neg__(self):
        return self * (-1)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return Div([self, other])
    
    __rmul__ = __mul__
    __radd__ = __add__
    

def generate_random_function_integral(nranges=[0, 100], max_layer=1, max_sum=1, max_prod=1, max_deg=2):
    function_array = [sin(), cos(), tan(), log(), exp(), sqrt(), asin(), asin(), atan(), atan(), asin(), asin(), sqrt(), sqrt(), sqrt(), log(), log(), log()]
    p=1/3
    tot_array = function_array + [poly.rand(random.randint(1, max_deg), coeff_range=nranges[:]) for i in range(int(len(function_array)*(1/(1-p) - 1)))]
    s = []
    for k in range(random.randint(1, max_sum)):
        s.append(Prod([Comp([tot_array[random.randint(1, len(tot_array) - 1)] for j in range(random.randint(1, max_layer))]) for i in range(random.randint(1, max_prod))]))
    
    return Sum(s)
def generate_random_function_integral_II(nranges=[0, 100], n=9, k=5, max_deg=2, comp=4, sums=3, prod=1):
    function_array = [sin(), cos(), tan(), log(), exp(), sqrt(), asin(), atan()]
    p=1/3
    tot_array = function_array + [poly.rand(random.randint(1, max_deg), coeff_range=nranges[:]) for i in range(int(len(function_array)*(1/(1-p) - 1)))]
    s = [tot_array[random.randint(0, len(tot_array) - 1)] for i in range(n)]
    a = [Sum for i in range(sums)] + [Comp for i in range(comp)] + [Prod for i in range(prod)]
    while len(s) > 3:
        funs = []
        for i in range(int(len(s) / k)):
            seed = a[random.randint(0, len(a) - 1)]
            sub_arr = []
            for j in range(random.randint(2, int(len(s) / 1.2))):
                sub_arr.append(s[random.randint(0, len(s) - 1)])
            funs.append(seed(sub_arr))
        
        s[:] = funs[:]
    seed = a[random.randint(0, 2)]
    q = seed(s[:]).simplify()
    return q

def rand_func_iii(nranges=[0, 10], max_deg=4, n=5, fweights=[1, 1, 1, 1, 1, 1, 1, 1, 1], wweights=[1, 1, 1, 1], prev_wraps=[]):
    functions = [sin(), cos(), tan(), log(), exp(), sqrt(), asin(), atan(), 'p']
    function_array = []
    for i in range(len(fweights)):
        for j in range(fweights[i]):
            function_array.append(functions[i])
    if n == 1:
        f = function_array[random.randint(0, len(function_array) - 1)]
        if f == 'p':
            f = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
        
        if isinstance(f, sqrt):
            return Comp([poly.rand(random.randint(0, max_deg), coeff_range=nranges[:], sgn_sensitive=0), sqrt()])
        return f
    
    else:
        wrappers_array = [Sum, Comp, Prod, Div]
        s = 0
        if Div in prev_wraps:
            s = 1
            wrappers_array.remove(Div)
        wrappers = []
        for i in range(len(wweights) - s):
            for j in range(wweights[i]):
                wrappers.append(wrappers_array[i])
        f = function_array[random.randint(0, len(function_array) - 1)]
        if f == sqrt:
            f = Comp([poly.rand(random.randint(0, max_deg), coeff_range=nranges[:], sgn_sensitive=0), sqrt()])
        if f == 'p':
            f = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
        wrapper = wrappers[random.randint(0, len(wrappers) - 1)]
        return wrapper([f, rand_func_iii(nranges=nranges[:], n=n-1, fweights=fweights[:], wweights=wweights[:], prev_wraps=prev_wraps+[wrapper])]).simplify()

def generate_integral_problem(nranges=[0, 100], boundary_ranges=[-10, 10],n=9, k=5, max_deg=2, comp=4, sums=3, prod=1):
    
    lb = random.randint(boundary_ranges[0], boundary_ranges[1] - 1)
    hb = random.randint(lb + 1, boundary_ranges[1])
    while True:
        try:
            p = generate_random_function_integral_II(nranges=nranges, n=n, k=k, max_deg=max_deg, comp=comp, sums=sums, prod=prod)
            result = p(hb) - p(lb)
            break
        except:
            continue
    int_ppr = [[" ", "/"], 
               ["/", " "],
               ["|", " "],
               ["|", " "],
               ["|", " "],
               ["|", " "],
               ["|", " "],
               [" ", "/"],
               ["/", " "]]
    for i in range(max(len(str(lb)), len(str(hb)))):
        curr = [[str(hb)[i] if i < len(str(hb)) else " "],
                [" "],
                [" "],
                [" "],
                [" "],
                [" "],
                [" "],
                [" "],
                [str(lb)[i] if i < len(str(lb)) else " "]]
        int_ppr[:] = connect(int_ppr[:], curr[:])[:]
    x = p.diff().npprint()
    r = [" " for i in x[0]]
    dx = [[" ", " "],
          [" ", " "],
          [" ", " "],
          [" ", " "],
          ["d", "x"],
          [" ", " "],
          [" ", " "],
          [" ", " "],
          [" ", " "]]
    int_ppr[:] = connect(int_ppr[:], [r] + x + [r])[:]
    int_ppr[:] = connect(int_ppr[:], dx[:])[:]
    string = strpprint(int_ppr)[:]
    return result, string, lb, hb

def generate_integral_problem_II(nranges=[0, 100], boundary_ranges=[-10, 10],n=9, k=5, max_deg=2, comp=4, sums=3, prod=1):
    
    lb = random.randint(boundary_ranges[0], boundary_ranges[1] - 1)
    hb = random.randint(lb + 1, boundary_ranges[1])
    while True:
        try:
            p = generate_random_function_integral_II(nranges=nranges, n=n, k=k, max_deg=max_deg, comp=comp, sums=sums, prod=prod)
            result = numericIntegration(p, lb, hb)
            break
        except:
            continue
    int_ppr = [[" ", "/"], 
               ["/", " "],
               ["|", " "],
               ["|", " "],
               ["|", " "],
               ["|", " "],
               ["|", " "],
               [" ", "/"],
               ["/", " "]]
    for i in range(max(len(str(lb)), len(str(hb)))):
        curr = [[str(hb)[i] if i < len(str(hb)) else " "],
                [" "],
                [" "],
                [" "],
                [" "],
                [" "],
                [" "],
                [" "],
                [str(lb)[i] if i < len(str(lb)) else " "]]
        int_ppr[:] = connect(int_ppr[:], curr[:])[:]
    x = p.npprint()
    r = [" " for i in x[0]]
    dx = [[" ", " "],
          [" ", " "],
          [" ", " "],
          [" ", " "],
          ["d", "x"],
          [" ", " "],
          [" ", " "],
          [" ", " "],
          [" ", " "]]
    int_ppr[:] = connect(int_ppr[:], [r] + x + [r])[:]
    int_ppr[:] = connect(int_ppr[:], dx[:])[:]
    string = strpprint(int_ppr)[:]
    return result, string, lb, hb

def generate_integral_problem_iii(nranges=[0, 100], boundary_ranges=[-10, 10],n=9, max_deg=2, fweights=[1, 1, 1, 1, 1, 1, 1, 1, 1], wweights=[1, 1, 1, 1]):
    
    lb = random.randint(boundary_ranges[0], boundary_ranges[1] - 1)
    hb = random.randint(lb + 1, boundary_ranges[1])
    while True:
        try:
            p = rand_func_iii(nranges=nranges, max_deg=max_deg, n=n, fweights=fweights, wweights=wweights)
            result = numericIntegration(p, lb, hb)
            int_ppr = [[" ", " ", "/"], 
                    [" ", "/", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "/", " "],
                    ["/", " ", " "]]
            for i in range(max(len(str(lb)), len(str(hb)))):
                curr = [[str(hb)[i] if i < len(str(hb)) else " "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [str(lb)[i] if i < len(str(lb)) else " "]]
                int_ppr[:] = connect(int_ppr[:], curr[:])[:]
            x = p.npprint()
            r = [" " for i in x[0]]
            dx = [[" ", " "],
                [" ", " "],
                [" ", " "],
                [" ", " "],
                ["d", "x"],
                [" ", " "],
                [" ", " "],
                [" ", " "],
                [" ", " "]]
            int_ppr[:] = connect(int_ppr[:], [r] + x + [r])[:]
            int_ppr[:] = connect(int_ppr[:], dx[:])[:]
            string = strpprint(int_ppr)[:]
            return result, string, lb, hb
        except:
            continue
'''
r, s, l, h = generate_integral_problem_iii(nranges=[1, 10], boundary_ranges=[-10, 10], n=5, max_deg=4, fweights=[1, 1, 1, 4, 4, 5, 0, 1, 9], wweights=[1, 0, 1, 0])
print(s)
print(r)
'''
#p = rand_func_iii(nranges=[1, 10], n=4, max_deg=4, fweights=[0, 0, 0, 2, 0, 5, 0, 0, 9], wweights=[1, 0, 0, 1])
#print(strpprint(p.npprint()))

def jacobian(farray):
    array = [[farray[i].diff(j) for j in range(len(farray))] for i in range(len(farray))]
    return matrix(array)

def laplace_t(function):
    return lambda s : simpInt(lambda x : function(x) * math.exp(-s*x), 0, 1000)

def fourier_series(f, period):
    a_n_d = lambda n : (lambda x : f(x) * math.cos(2 * n * math.pi * x / period))
    b_n_d = lambda n : (lambda x : f(x) * math.sin(2 * n * math.pi * x / period)) 
    a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
    b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
    a_0 = numericIntegration(f, -period/2, period/2) / period
    return [a_n, b_n, a_0]

def fourier_ct(function, start, end, dx=0.0001):
    return lambda w : numericIntegration(lambda x : function(x) * math.cos(w * x), start, end, dx=dx) * math.sqrt(2/math.pi)

def fourier_st(function, start, end, dx=0.0001):
    return lambda w : numericIntegration(lambda x : function(x) * math.sin(w * x), start, end, dx=dx) * math.sqrt(2/math.pi)

def fourier_t(function, start, end, dx=0.0001):
    return lambda w : numericIntegration(lambda x : function(x) * cmath.exp(complex(0, -w * x)), start, end, dx=dx) / math.sqrt(2 * math.pi)


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

def generate_integrable_ratExpr_tex(deg=3, nranges = [1, 10]):
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
    z = Div([p, q]).texify()
    return [p, q, z]
def generate_ratExpr(max_deg=3, nranges=[1, 10]):
    p = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    q = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    str1 = str(p)
    str2 = str(q)
    str1cpy = str1[:]
    str2cpy = str2[:]
    len_measure1 = len(str1cpy.split("\n")[0])
    len_measure2 = len(str2cpy.split("\n")[0])
    str3 = "".join(["-" for j in range(max(len_measure1, len_measure2))])
    z = str1 + "\n" + str3 + "\n" + str2 + "\n"
    return [p, q, z]

def generate_ratExpr_tex(max_deg=3, nranges=[1, 10]):
    p = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    q = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    str1 = str(p)
    str2 = str(q)
    str1cpy = str1[:]
    str2cpy = str2[:]
    len_measure1 = len(str1cpy.split("\n")[0])
    len_measure2 = len(str2cpy.split("\n")[0])
    str3 = "".join(["-" for j in range(max(len_measure1, len_measure2))])
    z = str1 + "\n" + str3 + "\n" + str2 + "\n"
    return [p, q, Div([p, q]).texify()]

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
def generate_eulersub_rand(deg=2, nranges=[1, 10]):
    rat1 = generate_ratExpr(max_deg=deg, nranges=nranges[:])
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
    
def generate_eulersub_rand_tex(deg=2, nranges=[1, 10]):
    p = poly.rand(deg, coeff_range=nranges[:])
    q = poly.rand(deg, coeff_range=nranges[:])
    r = poly.rand(deg, coeff_range=nranges[:])
    f = Prod([Div([p, q]), sqrt(r)])
    return f, f.texify()
    
  

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
    
def generate_trig_tex(nranges=[1, 10]):
    a, s, c = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    a2, s2, c2 = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    a3, t = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    modeseed = random.randint(1, 3) 
    if modeseed == 1:
        p = lambda x : a + s * math.sin(x) + c * math.cos(x)
        q = lambda x : a2 + s2 * math.sin(x) + c2 * math.cos(x)
        p_str = "%d + %d\sin x + %d\cos x" % (a, s, c)
        q_str = "%d + %d\sin x + %d\cos x" % (a2, s2, c2)
        return [lambda x : p(x) / q(x), '\\frac{%s}{%s}'%(p_str, q_str)]
    
    elif modeseed == 2:
        l = random.randint(nranges[0], nranges[1])
        s = random.randint(1, 2) % 2
        p = lambda x : l / (a3 + t * ((-1)**s) * math.tan(x))
        q_str = "%d + %d\tan x"%(a3, t * ((-1)**s)) if not s else "%d - %d\tan x"%(a3, t)
        t_str = "\\frac{%d}{%s}"%(l, q_str)
        return [p, t_str]
    
    elif modeseed == 3:
        return generate_trig_prod_tex(nranges=nranges[:])

def generate_trig_prod(nranges=[1, 10]):
    a, b = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    function = lambda x : (math.sin(x)**a)*(math.cos(x))**b
    x = [i for i in str(a)] if a != 1 else [" "]
    y = [i for i in str(b)] if a != 1 else [" "]
    string_array = [[" ", " ", " "] + x + [" ", " ", " ", " ", " "] + y + [" "],
                    ["s", "i", "n"] + [" " for i in range(len(str(a)))] + ["x", " "] + ["c", "o", "s"] + [" " for i in range(len(str(b)))] + ["x"]]
    string = "\n".join(["".join(i) for i in string_array])
    return [function, string]

def generate_trig_prod_tex(nranges=[1, 10]):
    a, b = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    function = lambda x : (math.sin(x)**a)*(math.cos(x))**b
    string = "\\sin^{%d} x \\cos^{%d} x"%(a, b)
    return [function, string]

def generate_fourier_ct(nranges=[1, 10], n_partite=1, deg=2, p_range=[1, 5], exp_cond=False, u_cond=False, umvar_cond=False):
    p1 = poly.rand(deg, coeff_range=nranges[:])
    c1 = random.randint(nranges[0], nranges[1])
    period = random.randint(p_range[0], p_range[1])
    rand_exp = lambda x : math.exp(c1 * x)
    f = (lambda x : p1(x) * rand_exp(x))
    if n_partite > 1:
        arr = [0]
        step = period // n_partite
        for i in range(n_partite - 1):
            arr.append(random.randint(arr[-1], arr[-1] + step))
        arr.append(period)
        p_array = [poly.rand(deg, coeff_range=nranges[:]) for i in range(n_partite)]
        def g(x):
            for i in range(n_partite):
                if arr[i] <= x < arr[i+1]:
                    return p_array[i](x)
            
            return 0
        
        z = fourier_ct(g, 0, period)
        str_arr = ["if %d <= x < %d then f(x) = \n"%(arr[i], arr[i+1]) + strpprint(p_array[i].pprint()) + "\n" for i in range(n_partite)]
        string = "".join(str_arr)
        return [g, period, z, string, p1, c1]



    if not exp_cond:
        f = lambda x : p1(x)
    if umvar_cond:
        x, y, z = rndF(nranges=nranges), rndF(nranges=nranges), rndF(nranges=nranges)
        p1 = polymvar.rand(max_deg=2, nrange=nranges[:])
        f = lambda t : p1(x[0](t), y[0](t), z[0](t))
        fct = fourier_ct(f, 0, period)
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
        return [f, period, fct, strpprint(pnstr), p1, c1]
    if u_cond:
        z = rndF(nranges=nranges)
        f = lambda x : p1(z[0](x))
        fct = fourier_ct(f, 0, period)
        nstr = connect(p1.pprint(), [[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "w", "h", "e", "r", "e", " ", "x", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, z[1])
        return [f, period, fct, strpprint(pnstr), p1, c1]
    
     
        
    fct = fourier_ct(f, 0, period)
    if not exp_cond:
        return [f, period, fct, strpprint(p1.pprint()), p1, c1]
    poly_pprint = connect(connect([["/"], ["|"], ["\\"]], p1.pprint()), [["\\"], ["|"], ["/"]])
    array = [[" "], ["e"], [" "]]
    for i in str(c1):
        array[0].append(i)
    array[0].append("x")
    for i in range(len(str(c1)) + 1):
        array[1].append(" ")
        array[2].append(" ")
    
    full_pprint = connect(array, poly_pprint)
    return [f, period, fct, strpprint(full_pprint), p1, c1]

def generate_fourier_st(nranges=[1, 10], n_partite=1, deg=2, p_range=[1, 5], exp_cond=False, u_cond=False, umvar_cond=False):
    p1 = poly.rand(deg, coeff_range=nranges[:])
    c1 = random.randint(nranges[0], nranges[1])
    period = random.randint(p_range[0], p_range[1])
    rand_exp = lambda x : math.exp(c1 * x)
    f = (lambda x : p1(x) * rand_exp(x))
    if n_partite > 1:
        arr = [0]
        step = period // n_partite
        for i in range(n_partite - 1):
            arr.append(random.randint(arr[-1], arr[-1] + step))
        arr.append(period)
        p_array = [poly.rand(deg, coeff_range=nranges[:]) for i in range(n_partite)]
        def g(x):
            for i in range(n_partite):
                if arr[i] <= x < arr[i+1]:
                    return p_array[i](x)
            
            return 0
        
        z = fourier_st(g, 0, period)
        str_arr = ["if %d <= x < %d then f(x) = \n"%(arr[i], arr[i+1]) + strpprint(p_array[i].pprint()) + "\n" for i in range(n_partite)]
        string = "".join(str_arr)
        return [g, period, z, string, p1, c1]



    if not exp_cond:
        f = lambda x : p1(x)
    if umvar_cond:
        x, y, z = rndF(nranges=nranges), rndF(nranges=nranges), rndF(nranges=nranges)
        p1 = polymvar.rand(max_deg=2, nrange=nranges[:])
        f = lambda t : p1(x[0](t), y[0](t), z[0](t))
        fct = fourier_st(f, 0, period)
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
        return [f, period, fct, strpprint(pnstr), p1, c1]
    if u_cond:
        z = rndF(nranges=nranges)
        f = lambda x : p1(z[0](x))
        fct = fourier_st(f, 0, period)
        nstr = connect(p1.pprint(), [[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "w", "h", "e", "r", "e", " ", "x", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, z[1])
        return [f, period, fct, strpprint(pnstr), p1, c1]
    
     
        
    fct = fourier_st(f, 0, period)
    if not exp_cond:
        return [f, period, fct, strpprint(p1.pprint()), p1, c1]
    poly_pprint = connect(connect([["/"], ["|"], ["\\"]], p1.pprint()), [["\\"], ["|"], ["/"]])
    array = [[" "], ["e"], [" "]]
    for i in str(c1):
        array[0].append(i)
    array[0].append("x")
    for i in range(len(str(c1)) + 1):
        array[1].append(" ")
        array[2].append(" ")
    
    full_pprint = connect(array, poly_pprint)
    return [f, period, fct, strpprint(full_pprint), p1, c1]

def generate_fourier_t(nranges=[1, 10], n_partite=1, deg=2, p_range=[1, 5], exp_cond=False, u_cond=False, umvar_cond=False):
    p1 = poly.rand(deg, coeff_range=nranges[:])
    c1 = random.randint(nranges[0], nranges[1])
    period = 2*random.randint(p_range[0], p_range[1])
    rand_exp = lambda x : math.exp(c1 * x)
    f = (lambda x : p1(x) * rand_exp(x))
    if n_partite > 1:
        arr = [0]
        step = period // n_partite
        for i in range(n_partite - 1):
            arr.append(random.randint(arr[-1], arr[-1] + step))
        arr.append(period)
        p_array = [poly.rand(deg, coeff_range=nranges[:]) for i in range(n_partite)]
        def g(x):
            for i in range(n_partite):
                if arr[i] <= x < arr[i+1]:
                    return p_array[i](x)
            
            return 0
        
        z = fourier_t(g, 0, period)
        str_arr = ["if %d <= x < %d then f(x) = \n"%(arr[i], arr[i+1]) + strpprint(p_array[i].pprint()) + "\n" for i in range(n_partite)]
        string = "".join(str_arr)
        return [g, period, z, string, p1, c1]



    if not exp_cond:
        f = lambda x : p1(x)
    if umvar_cond:
        x, y, z = rndF(nranges=nranges), rndF(nranges=nranges), rndF(nranges=nranges)
        p1 = polymvar.rand(max_deg=2, nrange=nranges[:])
        f = lambda t : p1(x[0](t), y[0](t), z[0](t))
        fct = fourier_t(f, 0, period)
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
        return [f, period, fct, strpprint(pnstr), p1, c1]
    if u_cond:
        z = rndF(nranges=nranges)
        f = lambda x : p1(z[0](x))
        fct = fourier_t(f, 0, period)
        nstr = connect(p1.pprint(), [[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "], 
                                     [" ", "w", "h", "e", "r", "e", " ", "x", " ", "=", " "],
                                     [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]])[:]
        pnstr = connect(nstr, z[1])
        return [f, period, fct, strpprint(pnstr), p1, c1]
    
     
        
    fct = fourier_t(f, 0, period)
    if not exp_cond:
        return [f, period, fct, strpprint(p1.pprint()), p1, c1]
    poly_pprint = connect(connect([["/"], ["|"], ["\\"]], p1.pprint()), [["\\"], ["|"], ["/"]])
    array = [[" "], ["e"], [" "]]
    for i in str(c1):
        array[0].append(i)
    array[0].append("x")
    for i in range(len(str(c1)) + 1):
        array[1].append(" ")
        array[2].append(" ")
    
    full_pprint = connect(array, poly_pprint)
    return [f, period, fct, strpprint(full_pprint), p1, c1]


def generate_fourier_s(nranges=[1, 10], n_partite=1, deg=2, p_range=[1, 5], exp_cond=False, u_cond=False, umvar_cond=False):
    p1 = poly.rand(deg, coeff_range=nranges[:])
    c1 = random.randint(nranges[0], nranges[1])
    period = 2*random.randint(p_range[0], p_range[1])
    rand_exp = lambda x : math.exp(c1 * x)
    f = (lambda x : p1(x) * rand_exp(x))
    if n_partite > 1:
        arr = [-period/2]
        step = period // n_partite
        for i in range(n_partite - 1):
            arr.append(random.randint(arr[-1], arr[-1] + step))
        arr.append(period/2)
        p_array = [poly.rand(deg, coeff_range=nranges[:]) for i in range(n_partite)]
        def g(x):
            for i in range(n_partite):
                if arr[i] <= x < arr[i+1]:
                    return p_array[i](x)
            
            return 0
        
        a_n_d = lambda n : (lambda x : g(x) * math.cos(2 * n * math.pi * x / period))
        b_n_d = lambda n : (lambda x : g(x) * math.sin(2 * n * math.pi * x / period)) 
        a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
        b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
        a_0 = numericIntegration(g, -period/2, period/2) / period
        str_arr = ["if %d <= x < %d then f(x) = \n"%(arr[i], arr[i+1]) + strpprint(p_array[i].pprint()) + "\n" for i in range(n_partite)]
        string = "".join(str_arr)
        return [g, period, a_n, b_n, a_0, string, p1, c1]



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


def generate_fourier_s_tex(nranges=[1, 10], n_partite=1, deg=2, p_range=[1, 5], exp_cond=False, u_cond=False, umvar_cond=False):
    p1 = poly.rand(deg, coeff_range=nranges[:])
    c1 = random.randint(nranges[0], nranges[1])
    period = 2*random.randint(p_range[0], p_range[1])
    rand_exp = lambda x : math.exp(c1 * x)
    f = (lambda x : p1(x) * rand_exp(x))
    if n_partite > 1:
        arr = [-period/2]
        step = period // n_partite
        for i in range(n_partite - 1):
            arr.append(random.randint(arr[-1], arr[-1] + step))
        arr.append(period/2)
        p_array = [poly.rand(deg, coeff_range=nranges[:]) for i in range(n_partite)]
        def g(x):
            for i in range(n_partite):
                if arr[i] <= x < arr[i+1]:
                    return p_array[i](x)
            
            return 0
        
        a_n_d = lambda n : (lambda x : g(x) * math.cos(2 * n * math.pi * x / period))
        b_n_d = lambda n : (lambda x : g(x) * math.sin(2 * n * math.pi * x / period)) 
        a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
        b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
        a_0 = numericIntegration(g, -period/2, period/2) / period
        str_arr = ["\\text{if }%d \leq x < %d \text{ then } f(x) = \n"%(arr[i], arr[i+1]) + p_array[i].texify() + "\\\\" for i in range(n_partite)]
        string = "".join(str_arr)
        return [g, period, a_n, b_n, a_0, string, p1, c1]



    if not exp_cond:
        f = lambda x : p1(x)
    
        
    a_n_d = lambda n : (lambda x : f(x) * math.cos(2 * n * math.pi * x / period))
    b_n_d = lambda n : (lambda x : f(x) * math.sin(2 * n * math.pi * x / period)) 
    a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
    b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
    a_0 = numericIntegration(f, -period/2, period/2) / period
    if not exp_cond:
        return [f, period, a_n, b_n, a_0, p1.texify(), p1, c1]
    
    return [f, period, a_n, b_n, a_0, Prod([p1, Comp([poly([0, c1]), exp()])]).texify(), p1, c1]

def fourier_s_poly(p1, p_range=[1, 5]):
    period = 2*random.randint(p_range[0], p_range[1])
    f = lambda x : p1(x)
    a_n_d = lambda n : (lambda x : f(x) * math.cos(2 * n * math.pi * x / period))
    b_n_d = lambda n : (lambda x : f(x) * math.sin(2 * n * math.pi * x / period)) 
    a_n = lambda n : numericIntegration(a_n_d(n), -period/2, period/2) / (period/2)
    b_n = lambda n : numericIntegration(b_n_d(n), -period/2, period/2) / (period/2)
    a_0 = numericIntegration(f, -period/2, period/2) / period
    return [period, a_n, b_n, a_0]

def randFunction(nranges=[1, 10], n=2, max_deg=2, symbol="x"):
    functions = [(SINH, [[" ", " ", " ", " ", " ", " ", " "], ["s", "i", "n", "h", "(", symbol, ")"], [" ", " ", " ", " ", " ", " ", " "]]), 
                 (COSH, [[" ", " ", " ", " ", " ", " ", " "], ["c", "o", "s", "h", "(", symbol, ")"], [" ", " ", " ", " ", " ", " ", " "]]), 
                 (EXP, [[" ", symbol], ["e"," "], [" ", " "]])]
    return functions[random.randint(0, len(functions) - 1)]

def rndF(nranges=[-10, 10], symbol="x"):
    a, b, c, d, e = [random.randint(nranges[0], nranges[1]) for i in range(5)]
    functions = [(lambda x : math.sinh(a * x), [[" ", " ", " ", " ", " "] + [" " for i in (str(a) if a != 1 else "") ] + [ " ", " "], ["s", "i", "n", "h", "("] + [i for i in (str(a) if a != 1 else "")] + [symbol, ")"], [" ", " ", " ", " ", " "] + [" " for i in (str(a) if a != 1 else "")] + [ " ", " "]]), 
                 (lambda x : math.cosh(b * x), [[" ", " ", " ", " ", " "] + [" " for i in (str(b) if a != 1 else "")] + [ " ", " "], ["c", "o", "s", "h", "("] + [i for i in (str(b) if a != 1 else "")] + [symbol, ")"], [" ", " ", " ", " ", " "] + [" " for i in (str(b) if a != 1 else "")] + [ " ", " "]]), 
                 (lambda x : math.exp(c * x), [[" "] + [i for i in (str(c) if a != 1 else "")] + [symbol], ["e"," "] + [" " for i in (str(c) if a != 1 else "")], [" ", " "] + [" " for i in (str(c) if a != 1 else "")]]),
                 (lambda x : math.sin(d * x), [[" ", " ", " ", " "] + [" " for i in (str(d) if a != 1 else "")] + [ " ", " "], ["s", "i", "n", "("] + [i for i in (str(d) if a != 1 else "")] + [symbol, ")"], [" ", " ", " ", " "] + [" " for i in (str(d) if a != 1 else "")] + [ " ", " "]]), 
                 (lambda x : math.cos(e * x), [[" ", " ", " ", " "] + [" " for i in (str(e) if a != 1 else "")] + [ " ", " "], ["c", "o", "s", "("] + [i for i in (str(e) if a != 1 else "")] + [symbol, ")"], [" ", " ", " ", " "] + [" " for i in (str(e) if a != 1 else "")] + [ " ", " "]]), 
                 ]
    return functions[random.randint(0, len(functions) - 1)]
def rndFeval(symbol="x"):
    functions = [(lambda x : math.sinh(x), [[" ", " ", " ", " ", " "] +  [ " ", " "], ["s", "i", "n", "h", "("] + [symbol, ")"], [" ", " ", " ", " ", " "]  + [ " ", " "]]), 
                 (lambda x : math.cosh(x), [[" ", " ", " ", " ", " "]  + [ " ", " "], ["c", "o", "s", "h", "("]  + [symbol, ")"], [" ", " ", " ", " ", " "] + [ " ", " "]]), 
                 (lambda x : math.exp(x), [[" "]  + [symbol], ["e"," "] , [" ", " "] ]),
                 (lambda x : math.sin(x), [[" ", " ", " ", " "]  + [ " ", " "], ["s", "i", "n", "("] + [symbol, ")"], [" ", " ", " ", " "] + [ " ", " "]]), 
                 (lambda x : math.cos(x), [[" ", " ", " ", " "]  + [ " ", " "], ["c", "o", "s", "("] + [symbol, ")"], [" ", " ", " ", " "] + [ " ", " "]]),
                 (lambda x : math.log(x), [[" ", " ", " ", " "]  + [ " ", " "], ["l", "o", "g", "("] + [symbol, ")"], [" ", " ", " ", " "] + [ " ", " "]]),
                 (lambda x : math.tan(x), [[" ", " ", " ", " "]  + [ " ", " "], ["t", "a", "n", "("] + [symbol, ")"], [" ", " ", " ", " "] + [ " ", " "]]),
                 (lambda x : math.asin(x), [[" ", " ", " ", " ", " "] +  [ " ", " "], ["a", "s", "i", "n", "("] + [symbol, ")"], [" ", " ", " ", " ", " "]  + [ " ", " "]]),
                 (lambda x : math.acos(x), [[" ", " ", " ", " ", " "] +  [ " ", " "], ["a", "c", "o", "s", "("] + [symbol, ")"], [" ", " ", " ", " ", " "]  + [ " ", " "]]),
                 (lambda x : math.atan(x), [[" ", " ", " ", " ", " "] +  [ " ", " "], ["a", "t", "a", "n", "("] + [symbol, ")"], [" ", " ", " ", " ", " "]  + [ " ", " "]]),
                  
                 ]
    return functions[random.randint(0, len(functions) - 1)]

def CrndF(nranges=[-10, 10], symbol="z"):
    a, b, c, d, e, f = [random.randint(nranges[0], nranges[1]) for i in range(6)]
    functions = [(lambda x : cmath.sinh(a * x), [[" ", " ", " ", " "] + [" " for i in str(a)] + [" ",  " ", " "], ["s", "i", "n", "h", "("] + [i for i in str(a)] + [ symbol, ")"], [ " ", " ", " ", " "] + [" " for i in str(a)] + [" " , " ", " "]]), 
                 (lambda x : cmath.cosh(b * x), [[ " ", " ", " ", " "] + [" " for i in str(b)] + [" ", " ", " "], ["c", "o", "s", "h", "("] + [i for i in str(b)] + [ symbol, ")"], [ " ", " ", " ", " "] + [" " for i in str(b)] + [" " , " ", " "]]), 
                 (lambda x : cmath.exp(c * x), [[" "] + [i for i in str(c)] + [symbol], ["e", " "] + [" " for i in str(c)], [ " ", " "] + [" " for i in str(c)]]),
                 (lambda x : cmath.sin(d * x * cmath.pi), [[ " ", " ", " "] + [" " for i in str(d)] + [ " ", " "], ["s", "i", "n", "("] + [i for i in str(d)] + [symbol, ")"], [" ", " ", " "] + [" " for i in str(d)] + [" ", " ", " "]]), 
                 (lambda x : cmath.cos(e * x * cmath.pi), [[ " ", " ", " "] + [" " for i in str(e)] + [ " ", " "], ["c", "o", "s", "("] + [i for i in str(e)] + [ symbol, ")"], [" ", " ", " "] + [" " for i in str(e)] + [" ", " ", " "]]), 
                 (lambda x : cmath.log(f * x), [[" " for i in range(len(str(f)) + 5)], ["L", "n", "("] + [i for i in str(f)] + [symbol, ")"], [" " for i in range(len(str(f)) + 5)]])
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


def random_parameterinc_curve(nranges=[1, 10], max_deg=2, dims=2):
    return [poly.rand(random.randint(0, max_deg), coeff_range=nranges[:]) for i in range(dims)]



def random_pfd(nrange=[1, 10], max_deg=2, pprint=False):
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
    if not pprint:
        return [p, q, "\n".join([str1, str3, str2])]
    else:
        return [p, q, z, a]

    
def rungeKutta4th_2ord(init_vals, coeffs, rhs, init_pt, end_pt, N):
    '''
    p(x)y'' + q(x)y' + r(x)y = f(x); y(x0) = y0; y'(x0) = p0 solve for y(x1)->
    init_vals = [y(x0), y'(x0)]
    coeffs = [r(x), q(x), p(x)]
    rhs = f(x)
    init_pt = x0
    end_pt = x1
    
    '''
    step = (end_pt - init_pt) / N
    p = (lambda x : coeffs[-1](x)) if callable(coeffs[-1]) else lambda x : coeffs[-1]
    q = (lambda x : coeffs[1](x)) if callable(coeffs[1]) else lambda x : coeffs[1]
    r = (lambda x : coeffs[0](x)) if callable(coeffs[0]) else lambda x : coeffs[0]
    x_n = init_pt
    y_n = init_vals[0]
    z_n = init_vals[1]
    f2 = lambda x, y, z : (rhs(x) - r(x) * y - q(x) * z) / (p(x))
    f1 = lambda z : z
    
    while x_n <= end_pt:
        k1 = step * f1(z_n)
        l1 = step * f2(x_n, y_n, z_n)
        
        k2 = step * f1(z_n + 0.5 * l1)
        l2 = step * f2(x_n + 0.5 * step, y_n + 0.5 * k1, z_n + 0.5 * l1)
        
        k3 = step * f1(z_n + 0.5 * l2)
        l3 = step * f2(x_n + 0.5 * step, y_n + 0.5 * k2, z_n + 0.5 * l2)
        
        k4 = step * f1(z_n + k3)
        l4 = step * f2(x_n + step, y_n + k4, z_n + l3)
        
        y_n += (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        z_n += (1/6) * (l1 + 2 * l2 + 2 * l3 + l4)
        x_n += step
    
    return y_n

def random_diff_eq_2_poly(nranges=[1, 10], mdeg_coeffs=1, max_deg=2, n=2):
    coeffs = [poly.rand(random.randint(0, mdeg_coeffs), nranges) for i in range(3)]
    ppr = [[], [], []]
    for i in range(len(coeffs) - 1, -1, -1):
        ppr = connect(ppr, [[" "], ["+"], [" "]])
        ppr = connect(ppr[:], connect([[" "], ["("], [" "]], coeffs[i].pprint() if hasattr(coeffs[i], 'pprint') else poly([coeffs[i]]).pprint()))
        ppr = connect(ppr[:], [["  "] + ["'" for j in range(i)], [")y"]+[" " for j in range(i)], [" " for j in range(i + 2)]])
    ppr = connect(ppr[:], [["   "], [" = "], ["   "]])
    f, pprint_s = randFunction(nranges=nranges[:], n=n, max_deg=max_deg) if random.randint(0, 1) else (PowSeries(lambda n : 1 if n == 0 else 0), [[], [], []])
    h = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    s = h.pprint()
    fin_ppr = connect(ppr[:], connect(pprint_s[:], connect([[" "], ["("], [" "]], connect(s, [[" "], [")"], [" "]]))))
    init_vals = [random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])]
    n = f * h
    fin_func = lambda x : rungeKutta4th_2ord(init_vals, coeffs, n, 0, x, 100 * x)
    return fin_func, strpprint(fin_ppr), init_vals

def pdeSolveSepVar(coeffs, boundary):
    # Works on the first try !
    '''
    Initial version solves pde of the form
        Au_xx + Bu_yy + Cu_x + Du_y + Eu = 0
    
    coeffs = [A, B, C, D, E]
    where the boundary conditions are :
    
    u(0, y) = u(L, y) = 0
    
    and 
    
    u(x, 0) = g(x) and u_y(x, 0) = h(x)
    
    thus boundary = [L, g(x), h(x)] 
    
    h(x) is of the form p * exp(-Cx/(2A)) sin (n * pi * x / L) thus is denoted by [p, n]
    and so is g(x).
    thus boundary = [L, n, p_h, p_g]
    '''
    a, b, c, d, e = coeffs[:]
    L, n, p_h, p_g = boundary[:]
    l = e - (4*a**2*n**2*math.pi**2+c**2*L**2)/(4 * a * L**2)
    disc = d ** 2 - 4 * b * l
    function_x = lambda x: math.exp(-c/(2*a) * x) * math.sin(n * math.pi * x / L)
    if disc > 0:
        r1, r2 = (-d + math.sqrt(d**2 - 4*b*l)) / (2*b), (-d - math.sqrt(d**2 - 4*b*l)) / (2*b)
        alpha_1 = (r2*p_g - p_h)/(r2 - r1)
        beta_1 = (p_h - r1 * p_g) / (r2 - r1)
        function_y = lambda y : alpha_1 * math.exp(r1*y) + beta_1 * math.exp(r2*y)
        fin_func = lambda x, y : function_x(x) * function_y(y)
        return [fin_func, alpha_1, beta_1]
    
    elif disc < 0:
        alpha_1 = p_g
        q = math.sqrt(-disc)/(2 * b)
        beta_1 = p_h / q
        function_y = lambda y : math.exp(-d/(2*b) * y) * (alpha_1 * math.cos(q*y) + beta_1 * math.sin(q*y))
        fin_func = lambda x, y : function_x(x) * function_y(y)
        return [fin_func, alpha_1, beta_1]
    
def repCol(array, ind, col):
    new_array = [[array[i][j] if j != ind else col[i] for j in range(len(array[i]))] for i in range(len(array))]
    return new_array

def solveLineq(coeffs_array, solutions_array):
    A = matrix(coeffs_array[:])
    copy_array = coeffs_array[:]
    sol_det = A.det()
    arr_i = []
    for i in range(len(coeffs_array)):
        narr = [[j for j in i] for i in coeffs_array]
        for j in range(len(coeffs_array)):
            narr[j][i] = solutions_array[j]
        
        arr_i.append(matrix(narr).det() / sol_det)
    
    return arr_i

def pdeFDiffD(coeffs_array, boudary_conditions, step_x = 0.05, step_y=0.05):
    '''
    for the below equation we have : 
    
    A(x)u_xx + Bu_xy + C(y)u_yy + D(x)u_x + E(y)u_y + (F(x) + G(y))u = 0
    
    with boundary conditions : 
    u(x, 0) = R1(x)
    u(x, L) = R2(x)
    
    u(0, y) = S1(y)
    u(L, y) = S2(y)
    
    coeffs_array = [A, B, C, D, E, F, G]
    and
    boundary_conditions = [L, R1(x), R2(x), S1(y), S2(y)]
    
    '''
    A, B, C, D, E, F, G = coeffs_array[:]
    h, k = step_x, step_y
    grid = []
    L, R1, R2, S1, S2 = boudary_conditions[:]
    # grid[j][i] = u(i * step_x, j * step_y)
    for i in range(int(L / step_y) + 1):
        new_arr = []
        for j in range(int(L / step_x) + 1):
            if j == 0:
                new_arr.append(S1(i * step_y))
            elif int(j * step_x) == L:
                new_arr.append(S2(i * step_y))
            elif i == 0:
                new_arr.append(R1(j * step_x))
            elif int(i * step_y) == L:
                new_arr.append(R2(j*step_x))
            
            else:
                new_arr.append(0)
        
        grid.append(new_arr)
    
    for _ in range(1000):     
        for j in range(1, int(L/step_y)):
            for i in range(1, int(L/step_x)):
                x = i * step_x
                y = j * step_y
                a, b, c, d, e, f, = A(x), B(x, y), C(y), D(x), E(y), F(x) + G(y)
                
                p = [
                    (a/h**2+d/(2*h)) * grid[j][i+1],
                    (a/h**2-d/(2*h)) * grid[j][i-1],
                    (c/k**2 + e/(2*k)) * grid[j+1][i],
                    (c/k**2-e/(2*k)) * grid[j-1][i],
                    (b/(4*h*k)) * grid[j+1][i+1],
                    (-b/(4*h*k)) * grid[j-1][i+1],
                    (-b/(4*h*k)) * grid[j+1][i-1],
                    (b/(4*h*k)) * grid[j-1][i-1]
                ]
                if (2*a/h**2+2*c/k**2-f) == 0:
                    continue
                grid[j][i] = (1/(2*a/h**2+2*c/k**2-f)) * sum(p)
        
    def solution(x, y):
        if not (0 <= x <= L and 0 <= y <= L):
            print("Input not in domain !")
            return 0
        
        else:
            new_x, new_y = int(x / step_x), int(y / step_y)
            return grid[new_y][new_x]
    
    return solution

def randomPDEconst(nranges, L_ranges, sep=0):
    arr = []
    arr_str = []
    for i in range(4):
        z, s = rndF() if random.randint(0, 1) else [lambda x : 0, [[" "], ["0"], [" "]]]
        arr.append(z)
        arr_str.append(strpprint(s))
        
    r1, r2, s1, s2 = arr[:]
    arr_coeffs = []
    arr_coeffs_num = []
    for i in range(7):
        rand = random.randint(nranges[0], nranges[1]) if not (sep and i == 1) else 0
        if i != 1:
            arr_coeffs.append(lambda t : rand)
        if i == 1:
            arr_coeffs.append(lambda t, u : rand)
        arr_coeffs_num.append(rand)

    arr_coeffs_num[-2] += arr_coeffs_num[-1]
    arr_coeffs_num.pop(-1)
    l = random.randint(L_ranges[0], L_ranges[1])
    bnd_arr = [l, r1, r2, s1, s2]
    solution = pdeFDiffD(arr_coeffs, bnd_arr, step_x=0.1, step_y=0.1)
    string = [[], [], []]
    for i in range(6) : 
        if arr_coeffs_num[i] == 0:
            continue
        if i == 0:
            substr = ["x", "x"]
        elif i == 1:
            substr = ["x", "y"]
        elif i == 2:
            substr = ["y", "y"]
        elif i == 3:
            substr = ["x", " "]
        elif i == 4:
            substr = ["y", " "]
        else:
            substr = [" ", " "]
        nstr = connect(string, [["   "], [" + " if arr_coeffs_num[i] > 0 else " - "], ["   "]])[:]
        if abs(arr_coeffs_num[i]) == 1:
            new_string = connect(nstr, [[" " for j in range(3)],  ["u", " ", " "], [" "] + substr])[:]        
        
        new_string = connect(nstr, [[" " for j in range(len(str(abs(arr_coeffs_num[i]))) + 3)], [j for j in str(abs(arr_coeffs_num[i]))] + ["u", " ", " "], [" " for j in range(len(str(abs(arr_coeffs_num[i]))) + 1)] + substr])[:]        
        string = new_string[:]
    new_string = connect(string, [[" ", " ", " ", " "], [" ", "=", " ", "0"], [" ", " ", " ", " "]])[:]
    finstr = strpprint(new_string)
    return [solution, finstr, l, z, arr_str]

def specialPDE(nranges, L_ranges):
    arr = []
    arr_str = []
    for i in range(4):
        z, s = rndF() if random.randint(0, 1) else [lambda x : 0, [[" "], ["0"], [" "]]]
        arr.append(z)
        arr_str.append(strpprint(s))
        
    r1, r2, s1, s2 = arr[:]
    
    rand = random.randint(nranges[0], nranges[1])
    wave_eq_num = [abs(rand), 0, -1, 0, 0, 0, 0]
    heat_eq_num = [abs(rand), 0, 0, 0, -1, 0, 0]
    laplace_eq_num = [1, 1, 0, 0, 0, 0, 0]
    
    wave_eq = [lambda x : abs(rand), lambda x, y : 0, lambda x : -1, lambda x : 0, lambda x : 0, lambda x : 0, lambda x : 0]
    wave_eq_str = [[" " for _ in str(rand)] + [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
                   [i for i in str(rand)] + ["u", " ", " ", " ", "-", " ", "u", " ", " ", " ", "=", " ", "0"],
                   [" " for _ in str(rand)] + [" ", "x", "x", " ", " ", " ", " ", "t", "t", " ", " ", " ", " "]]
    heat_eq = [lambda x : abs(rand), lambda x, y : 0, lambda x : 0, lambda x : 0, lambda x : -1, lambda x : 0, lambda x : 0]
    heat_eq_str = [[" " for _ in str(rand)] + [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
                   [i for i in str(rand)] + ["u", " ", " ", " ", "-", " ", "u", " ", " ", " ", "=", " ", "0"],
                   [" " for _ in str(rand)] + [" ", "x", "x", " ", " ", " ", " ", "t", " ", " ", " ", " ", " "]]
    laplace_eq = [lambda x : 1, lambda x, y : 0, lambda x : 1, lambda x : 0, lambda x : 0, lambda x : 0, lambda x : 0]
    laplace_eq_str = [[" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
                   ["u", " ", " ", " ", "+", " ", "u", " ", " ", " ", "=", " ", "0"],
                   [" ", "x", "x", " ", " ", " ", " ", "y", "y", " ", " ", " ", " "]]
    polar_laplace = [lambda x : x**2, lambda x, y : 0, lambda x : 1, lambda x : x, lambda x : 0, lambda x : 0, lambda x : 0]
    polar_eq_str = [[" ", "2", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "],
                    ["x", " ", "u", " ", " ", "+", "x", "u", " ", "+", "u", " ", " ", " ", "=", " ", "0"],
                    [" ", " ", " ", "x", "x", " ", " ", " ", "x", " ", " ", "y", "y", " ", " ", " ", " "]]
    #tricomi_eq = [lambda x : 1, lambda x, y : 0, lambda x : 1/x, lambda x : 0, lambda x : 0, lambda x: 0, lambda x : 0]
    
    equations = [wave_eq, heat_eq, laplace_eq, polar_laplace]#, tricomi_eq]
    strings = [wave_eq_str, heat_eq_str, laplace_eq_str, polar_eq_str]
    ind = random.randint(0, 3)
    function = equations[ind]
    function_string = strings[ind]
    l = random.randint(L_ranges[0], L_ranges[1])
    bnd_arr = [l, r1, r2, s1, s2]
    solution = pdeFDiffD(equations[ind], bnd_arr, step_x=0.05, step_y=0.05)
    finstr = strpprint(function_string)
    return [solution, finstr, l, z, arr_str]

def randCauchyEuler(nranges=[1, 10], max_deg=2, n=2):
    p = poly.rand(1, nranges)
    coeffs = [p ** i if i != 0 else poly.rand(0, nranges) for i in range(3)]
    ppr = [[], [], []]
    for i in range(len(coeffs) - 1, -1, -1):
        ppr = connect(ppr, [[" "], ["+"], [" "]])
        ppr = connect(ppr[:], connect([[" "], ["("], [" "]], coeffs[i].pprint() if hasattr(coeffs[i], 'pprint') else poly([coeffs[i]]).pprint()))
        ppr = connect(ppr[:], [["  "] + ["'" for j in range(i)], [")y"]+[" " for j in range(i)], [" " for j in range(i + 2)]])
    ppr = connect(ppr[:], [["   "], [" = "], ["   "]])
    f, pprint_s = randFunction(nranges=nranges[:], n=n, max_deg=max_deg) if random.randint(0, 1) else (PowSeries(lambda n : 1 if n == 0 else 0), [[], [], []])
    h = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    s = h.pprint()
    fin_ppr = connect(ppr[:], connect(pprint_s[:], connect([[" "], ["("], [" "]], connect(s, [[" "], [")"], [" "]]))))
    init_vals = [random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])]
    n = f * h
    fin_func = lambda x : rungeKutta4th_2ord(init_vals, coeffs, n, 0, x, 100 * x)
    return fin_func, strpprint(fin_ppr), init_vals

def random_diff_eq_2_mixed(nranges=[1, 10], n=2, max_deg=2):
    cauchy_cond = random.randint(0, 1)
    if cauchy_cond :
        return randCauchyEuler(nranges=nranges, max_deg=max_deg, n=n)
    
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

def random_diff_eq_ord(order=3, nranges=[1, 10], n=2, max_deg=2):
    coeffs = [random.randint(nranges[0], nranges[1]) * (-1)**random.randint(0, 1) for i in range(order+1)]
    ppr = [[], [], []]
    for i in range(len(coeffs) - 1, -1, -1):
        if coeffs[i] != 0:
            if abs(coeffs[i]) != 1:
                ppr = connect(ppr[:], [[" " for j in range(len(str(abs(coeffs[i]))) + 2)]+["'" for j in range(i)],["+" if coeffs[i] > 0 else "-"]+[j for j in str(abs(coeffs[i]))] + ["y"] + [" " for j in range(i)], [" " for j in range(len(str(abs(coeffs[i]))) + i + 2)]])[:]
            else:
                ppr = connect(ppr[:], [[" " for j in range(order+1)]+["'" for j in range(i)],["+" if coeffs[i] > 0 else "-"]+ ["y"] + [" " for j in range(i)], [" " for j in range(i + order+1)]])[:]  
    ppr = connect(ppr[:], [["   "], [" = "], ["   "]])
    f, pprint_s = randFunction(nranges=nranges[:], n=n, max_deg=max_deg) if random.randint(0, 1) else (PowSeries(lambda n : 1 if n == 0 else 0), [[], [], []])
    h = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    s = h.pprint()
    fin_ppr = connect(ppr[:], connect(pprint_s[:], connect([[" "], ["("], [" "]], connect(s, [[" "], [")"], [" "]]))))
    init_vals = [random.randint(nranges[0], nranges[1]) for i in range(order)]
    n = f * h
    fin_func = poly(solveDEseries(coeffs, n, init_vals, 100))
    return fin_func, strpprint(fin_ppr), init_vals

def random_diff_eq_ord_mixed(order=3, nranges=[1, 10], n=2, max_deg=2):
    cauchy_cond = random.randint(0, 1)
    if cauchy_cond :
        return randCauchyEuler(nranges=nranges, max_deg=max_deg, n=n)
    
    return random_diff_eq_ord(order=order, nranges=nranges, n=n, max_deg=max_deg)

def random_cmplx_function(nrange=[1, 10], max_deg=2, n=2, repeat=False, mrep=0):
    p = poly.rand(max_deg, coeff_range=nrange[:])
    q_arr = []
    roots = []
    for i in range(int(max_deg / 2)):
        x, y = random.randint(nrange[0], nrange[1]), random.randint(nrange[0], nrange[1])
        a = random.randint(nrange[0], nrange[1])
        q_arr.append(poly([complex(x, y), a]))
        q_arr.append(poly([complex(x, -y), a]))
        if repeat:
            for j in range(random.randint(0, mrep)):
                q_arr.append(poly([complex(x, y), a]))
                q_arr.append(poly([complex(x, -y), a]))
        roots.append(-complex(x, y) / a)
        roots.append(-complex(x, -y) / a)
    
    a = poly([1])
    for i in q_arr:
        a *= i
    
    if max_deg % 2:
        m = poly.rand(1, coeff_range=nrange[:])
        a *= m
        roots.append(-m.coeffs[0] / m.coeffs[1])
        if repeat:
            for j in range(random.randint(0, mrep)):
                q_arr.append(m)
                a *= m
    
    q = poly([int(i.real) for i in a.coeffs[:]])
    p.variable = "z"
    q.variable = "z"
    f_array = []
    f_ppr_array = []
    for i in range(n):
        f, ppr = CrndF(nranges=[-5, 5])
        f_array.append(f)
        f_ppr_array.append(ppr)
    
    def function(z):
        w = p(z)
        for i in f_array:
            w *= i(z)
        
        return w
    
    str1 = connect([[" "], ["("], [" "]], connect(p.pprint(), [[" "], [")"], [" "]]))
    for i in f_ppr_array:
        nstr = connect(str1[:], connect([[" "], [" "], [" "]], i))
        str1 = nstr[:]
    
    str3 = q.pprint()
    l = max(len(str1[0]), len(str3[0]))
    str2 = ["-" for i in range(l)]
    finstr = "\n".join(["".join(i) for i in str1])+ "\n" + "".join(str2) + "\n" + "\n".join(["".join(i) for i in str3])
    fin_f = lambda z : function(z) / q(z)
    return fin_f, finstr, roots

def random_f_c_integrate(nranges=[1, 10], max_deg=2, n=2, repeat=False, mrep=0, clsd=True, boundary_ranges=[1, 10]):
    funct, fstring, roots = random_cmplx_function(nrange=nranges[:], max_deg=max_deg, n=n, repeat=repeat, mrep=mrep)
    center_root = sum(roots[:]) / len(roots)
    max_length = round(max([math.sqrt(abs(center_root - i)) for i in roots])) + 1
    if clsd:
        pcurve_real = lambda t : center_root.real + max_length * math.cos(t)
        pcurve_im = lambda t : center_root.imag + max_length * math.sin(t)
        path = lambda t : complex(pcurve_real(t), pcurve_im(t))
        res = cnint(funct, path, 0, 2*cmath.pi, dt=0.0001)
        ppr_path = [
            [" " for i in range(len(str(center_root)) + len(str(max_length)) + 6)] + ["i", "t"],
            [i for i in str(center_root)] + [" ", "+", " "] + [i for i in str(max_length)] + [" ", "e", " "],
            [" " for i in range(len(str(center_root)) + len(str(max_length)) + 8)]
        ]
        ppr_str = strpprint(ppr_path)
        return fstring, ppr_str, res
    
    else:
        pcurve_vect = pcurve.rand(max_deg=2, nranges=nranges)
        path = lambda t : complex(pcurve_vect.array[0](t), pcurve_vect.array[1](t))
        start, end = random.randint(boundary_ranges[0], boundary_ranges[1]), random.randint(boundary_ranges[0], boundary_ranges[1])
        res = cnint(funct, path, start, end)
        string = strpprint(connect([[" ", " ", " ", " "], ["x", " ", "=", " "], [" ", " ", " ", " "]], pcurve_vect.array[0].pprint()))
        string2 = strpprint(connect([[" ", " ", " ", " "], ["y", " ", "=", " "], [" ", " ", " ", " "]], pcurve_vect.array[1].pprint()))
        return fstring, string+"\n"+string2+"\n", res, start, end

def generate_integral_cmplx_rat(nranges = [1, 10], max_deg=2):
    p = poly.rand(random.randint(0, max_deg - 2), coeff_range=nranges[:])
    q_arr = []
    roots = []
    for i in range(int(max_deg / 2)):
        x, y = random.randint(nranges[0], nranges[1]) * (-1)**random.randint(0, 1), random.randint(nranges[0], nranges[1]) * (-1)**random.randint(0, 1)
        q_arr.append(poly([complex(x, y), 1]))
        q_arr.append(poly([complex(x, -y), 1]))
        roots.append(-complex(x, y) / 1)
        roots.append(-complex(x, -y) / 1)
    
    a = poly([1])
    for i in q_arr:
        a *= i
    
    if max_deg % 2:
        m = poly.rand(1, coeff_range=nranges[:])
        a *= m
        roots.append(-m.coeffs[0] / m.coeffs[1])
    
    q = poly([int(i.real) for i in a.coeffs[:]])
    d, e = random.randint(0, 10), random.randint(0, 10) 
    arr = [(lambda x : math.sin(d * x), [[" ", " ", " ", " "] + [" " for i in str(d)] + [ " ", " "], ["s", "i", "n", "("] + [i for i in str(d)] + ["x", ")"], [" ", " ", " ", " "] + [" " for i in str(d)] + [ " ", " "]]), 
                 (lambda x : math.cos(e * x), [[" ", " ", " ", " "] + [" " for i in str(e)] + [ " ", " "], ["c", "o", "s", "("] + [i for i in str(e)] + ["x", ")"], [" ", " ", " ", " "] + [" " for i in str(e)] + [ " ", " "]])]
    
    cond = random.randint(0, 1)
    if cond:
        f, fppr = arr[random.randint(0, 1)]
        function = lambda x : p(x) * f(x) / q(x)
        string = connect([[" "], ["("], [" "]], connect(p.pprint(), connect([[" "], [")"], [" "]], fppr)))
    else:
        function = lambda x : p(x)/ q(x)   
        string = p.pprint()[:]
    
    string3 = q.pprint()[:]
    string2 = ["-" for i in range(max(len(string[1]), len(string3[1])))]
    fstring = strpprint(string)+"\n"+"".join(string2)+"\n"+strpprint(string3)
    res = numericIntegration(lambda x : function(x/(1-x**2)) * (1+x**2)/(1-x**2)**2, -1, 1, dx=0.00001)
    
    return function, res, fstring

def generate_trig_cmplx(nranges=[1, 10]):
    a, s, c = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    c2 = random.randint(nranges[0], nranges[1])
    a2 = random.randint(c2, nranges[1])
    n = random.randint(1, 9)
    p = lambda x : a + s * math.sin(n*x)
    q = lambda x : a2 + c2 * math.cos(n*x)
    p_str = "%d + %dsin(%sx)" % (a, s, "" if n == 1 else str(n))
    q_str = "%d + %dcos(%sx)" % (a2, c2, "" if n == 1 else str(n))
    string = p_str + "\n" + "".join(["-" for i in range(max(len(p_str), len(q_str)))]) + "\n" + q_str
    res = numericIntegration(lambda x : p(x) / q(x), 0, 2*math.pi/n)
    return res, string, n

def maclaurin_series(function, n):
    return nDiff(function, n, 0, dx=0.00001) / math.factorial(n)

def generate_rand_mseries(nranges , max_deg):
    p = poly.rand(max_deg, coeff_range=nranges[:])
    q = poly.rand(max_deg, coeff_range=nranges[:])
    string = p.pprint()[:]
    string3 = q.pprint()[:]
    string2 = ["-" for i in range(max(len(string[1]), len(string3[1])))]
    fstring = strpprint(string)+"\n"+"".join(string2)+"\n"+strpprint(string3)
    function = lambda x : p(x) / q(x)
    res = lambda n : maclaurin_series(function, n)
    return fstring, res

def generate_rand_func_arr(ndigits=5, n=3):
    #syms = ["x", "y", "z", "w", "m", "n", "p", "q", "r", "s"]
    nums = [int(random.random() * 10 ** ndigits) / 10 ** ndigits for i in range(n)]
    while 0 in nums:
        nums[nums.index(0)] = int(random.random() * 10 ** ndigits) / 10 ** ndigits

    syms = [str(i) for i in nums]
    func = [rndFeval(symbol=syms[i]) for i in range(n)]
    fstr = [func[i][1]for i in range(n)]
    fres = [func[i][0](nums[i]) for i in range(n)]
    return fstr, fres, nums

def generate_matrix_str_ent(fstr, fres, dim, calc_ndigits=3):
    res_mat = [[int(fres[j * dim + i]*10**calc_ndigits)/(10**calc_ndigits) for i in range(dim)] for j in range(dim)]
    array = [[fstr[j * dim + i] for i in range(dim)] for j in range(dim)]
    res_det = det(res_mat[:])
    tot_cells = []
    for i in array:
        cells = []
        for j in i:
            lines = [[], [], []]
            lines = connect(lines, j)              
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
    
    string = matrixpprint(tot_lines[:])
    return res_det, string

def generate_matrix_item(ndigits=3, dim=3, calc_ndigits=3):
    fstr, fres, nums = generate_rand_func_arr(ndigits=ndigits, n=dim**2)
    syms = ["x", "y", "z", "w", "m", "n", "p", "q", "r", "s"]
    narray = [syms[i] + " = " + str(nums[i]) for i in range(dim**2)]
    return generate_matrix_str_ent(fstr, fres, dim, calc_ndigits=calc_ndigits)#, narray

def generate_function_item(ndigits=3, calc_ndigits=3, n=2):
    p = 1
    fstr, fres, nums = generate_rand_func_arr(ndigits=ndigits, n=n)
    syms = ["x", "y", "z", "w", "m", "n", "p", "q", "r", "s"]
    newstr = [[], [], []]
    for i in fstr:
        newstr = connect(newstr, i)
        
    for i in fres:
        p *= int(i*10**calc_ndigits)/(10**calc_ndigits)
    
    return strpprint(newstr), p

def rand_pfd_prop(nranges=[1, 10], max_deg=3):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    p2 = poly([1])
    curr_deg = 0
    while curr_deg < max_deg:
        if max_deg - curr_deg == 1:
            p2 *= poly.rand(1, coeff_range=nranges[:])
            curr_deg += 1
        
        elif max_deg - curr_deg == 2:
            p2 *= poly.rand(2, coeff_range=nranges[:])
            curr_deg += 2
        
        else:
            cdeg = random.randint(1, max_deg)
            p2 *= poly.rand(cdeg, coeff_range=nranges[:])
            curr_deg += cdeg
    
    return Div([p1, p2])

def rand_rat_prop(nranges=[1, 10], max_deg=3):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    p2 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    return Div([p1, p2])

def rand_int_part_i(nranges=[1, 10], max_deg=3, n=2, max_deg_comp=2):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    func = [exp(), sin(), cos(), p1]
    arr = []
    for i in range(n):
        z = random.randint(0, len(func) - 1)
        if func[z] == p1:
            arr += [func[z]]
            func.remove(p1)
            continue
        if isinstance(func[z], exp):
            arr += [Comp([poly.rand(1, coeff_range=nranges[:]), func[z]])]
            func.pop(z)
            continue
        arr += [Comp([poly.rand(1, coeff_range=nranges[:]), func[z]])]
        
    p2 = Prod(arr[:])
    return p2

def rand_int_part_ii(nranges=[1, 10], max_deg=3, n=2, max_deg_comp=2, ):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    func = [exp(), sin(), cos(), p1]#, atan(), asin()]
    arr = []
    for i in range(n):
        z = random.randint(0, len(func) - 1)
        if func[z] == p1:
            arr += [func[z]]
            func.remove(p1)
            continue
        if isinstance(func[z], exp):
            arr += [Comp([poly.rand(1, coeff_range=nranges[:]), func[z], poly.rand(random.randint(0, max_deg_comp), coeff_range=nranges[:])])]
            func.pop(z)
            continue
        arr += [Comp([poly.rand(1, coeff_range=nranges[:]), func[z], poly.rand(random.randint(0, max_deg_comp), coeff_range=nranges[:])])]
        
    p2 = Prod(arr[:])
    return p2

def rand_sqrt_type_i(nranges=[1, 10], max_deg=3, deg_i=2, n_range=[1, 4]):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    p2 = Comp([poly.rand(deg_i, coeff_range=nranges[:]), sqrt()])
    return Prod([p1, p2])

def rand_sqrt_type_ii(nranges=[1, 10], max_deg=3, deg_i=2, n_range=[1, 4]):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    p2 = Comp([poly.rand(deg_i, coeff_range=nranges[:]), sqrt()])
    return Div([p1, p2])

def rand_sqrt_type_iii(nranges=[1, 10], max_deg=3, deg_i=2, n_range=[1, 4]):
    p1 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    p2 = poly.rand(random.randint(0, max_deg), coeff_range=nranges[:])
    p3 = Comp([poly.rand(deg_i, coeff_range=nranges[:]), sqrt()])
    return Div([p1, Prod([p2, p3])])

def rand_rat_tan(nranges=[1, 10], n_range=[1, 4], m_range=[1, 4]):
    n = random.randint(n_range[0], n_range[1])
    p1 = poly([0 for i in range(n)] + [1])
    p2 = []
    for i in range(n):
        p2 += [poly([abs(random.randint(nranges[0], nranges[1])), 0, 1])]
    return Div([p1, Prod(p2[:])])

def rand_rat_cos(nranges=[1, 10], n_range=[1, 4], m_range=[1, 4]):
    n = random.randint(n_range[0], n_range[1])
    m = random.randint(m_range[0], m_range[1])
    p1 = poly([0 for i in range(m)] + [1])
    p2 = poly([random.randint(nranges[0], nranges[1]) ** (2*n)] + [0 for i in range(n - 1)] + [-abs(random.randint(nranges[0], nranges[1]))] + [0 for i in range(n - 1)] + [1])
    return Div([p1, p2])

def rand_sqrt_iv(nranges=[1, 10], max_deg=3, deg_i=2, n_range=[1, 4]):
    n = random.randint(n_range[0], n_range[1])
    m = random.randint(nranges[0], nranges[1])
    p1 = poly([0 for i in range(2 * n)] + [1])
    p2 = Comp([Prod([poly([1, 0, -1]), poly([1, 0, -m**2])]), sqrt()])
    return Div([p1, p2])

def rand_sqrt_v(nranges=[1, 10], max_deg=3, deg_i=2, n_range=[1, 4]):
    n = random.randint(n_range[0], n_range[1])
    m = random.randint(nranges[0], nranges[1])
    a = random.randint(nranges[0], nranges[1])
    p1 = poly([0 for i in range(2 * n)] + [1])
    p2 = Comp([Prod([poly([1, 0, -1]), poly([1, 0, -m**2])]), sqrt()])
    p3 = Comp([poly([1, 0, a]), p1])
    return Div([poly([1]), Prod([p3, p2])])

def rand_trig_i(nranges=[1, 10]):
    p = random.randint(nranges[0], nranges[1])
    q = random.randint(nranges[0], nranges[1])
    p1 = Comp([sin(), poly([0 for i in range(p)] + [1])])
    p2 = Comp([cos(), poly([0 for i in range(q)] + [1])])
    return Prod([p1, p2])

def rand_poly_i(nranges=[1, 10], m_range = [1, 10]):
    m = random.randint(m_range[0], m_range[1])
    p = random.randint(m_range[0], m_range[1])
    q = poly([0 for i in range(m)] + [1])
    x = poly.rand(2, coeff_range=nranges[:])
    xq = Comp([x, poly([0 for i in range(p)] + [1])])
    return Prod([q, xq])

def rand_sqrt_vi(nranges=[1, 10], max_deg=3, deg_i=2, n_range=[1, 10]):
    n = random.randint(n_range[0], n_range[1])
    q = poly([0 for i in range(n)] + [1])
    a, b, c = random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1]), random.randint(nranges[0], nranges[1])
    p = poly([a, 0, b, 0, c])
    return Div([q, Comp([p, sqrt()])])

def generate_general_int(function, args, boundary_ranges = [-10, 10]):
    lb = random.randint(boundary_ranges[0], boundary_ranges[1] - 1)
    hb = random.randint(lb + 1, boundary_ranges[1])
    while True:
        try:
            p = function(*args)
            result = numericIntegration(p, lb, hb)
            int_ppr = [[" ", " ", "/"], 
                    [" ", "/", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "|", " "],
                    [" ", "/", " "],
                    ["/", " ", " "]]
            for i in range(max(len(str(lb)), len(str(hb)))):
                curr = [[str(hb)[i] if i < len(str(hb)) else " "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [" "],
                        [str(lb)[i] if i < len(str(lb)) else " "]]
                int_ppr[:] = connect(int_ppr[:], curr[:])[:]
            x = p.npprint()
            r = [" " for i in x[0]]
            dx = [[" ", " "],
                [" ", " "],
                [" ", " "],
                [" ", " "],
                ["d", "x"],
                [" ", " "],
                [" ", " "],
                [" ", " "],
                [" ", " "]]
            int_ppr[:] = connect(int_ppr[:], [r] + x + [r])[:]
            int_ppr[:] = connect(int_ppr[:], dx[:])[:]
            string = strpprint(int_ppr)[:]
            return result, string, lb, hb
        except:
            continue

rand_prop_generators = [rand_pfd_prop, 
                        rand_rat_prop,
                        rand_int_part_i,
                        rand_int_part_ii,
                        rand_sqrt_type_i,
                        rand_sqrt_type_ii,
                        rand_sqrt_type_iii,
                        rand_sqrt_iv,
                        rand_sqrt_v,
                        rand_sqrt_vi,
                        rand_rat_tan,
                        rand_rat_cos,
                        rand_trig_i,
                        rand_poly_i,]
def ndiff(n, function):
    if n == 0:
        return function
    if n == 1:
        return function.diff() if hasattr(function, 'diff') else 0
    else:
        return ndiff(n - 1, function.diff() if hasattr(function, 'diff') else 0)

def solve_first_deg_ode_sys(equations, init_conds, target, n=10):
    '''
    equations = [
        f1(x, y1, y2, ..., ym),
        f2(x, y1, y2, ..., ym),
        .
        .
        .
        fm(x, y1, y2, ..., ym)
    ]
    init_conds = [y1(0), y2(0), ..., ym(0)]
    
    is equivalent to :
    y1' = f1(x, y1, y2, ..., ym)
    y2' = f2(x, y1, y2, ..., ym)
    .
    .
    .
    ym' = fm(x, y1, y2, ..., ym)
    '''
    
    y_array = init_conds[:]
    func_array = [j for j in equations]
    h = target / n
    x = 0
    i = 0
    while i <= n:
        inp_array_k1 = [x] + [i for i in y_array]
        array_k1 = [h * f(*inp_array_k1) for f in func_array]
        inp_array_k2 = [x + h/2] + [y_array[i] + array_k1[i]/2 for i in range(len(y_array))]
        array_k2 = [h * f(*inp_array_k2) for f in func_array]
        inp_array_k3 = [x + h/2] + [y_array[i] + array_k2[i]/2 for i in range(len(y_array))]
        array_k3 = [h * f(*inp_array_k3) for f in func_array]
        inp_array_k4 = [x + h] + [y_array[i] + array_k3[i] for i in range(len(y_array))]
        array_k4 = [h * f(*inp_array_k4) for f in func_array]
        
        new_y = [y_array[i] + (array_k1[i] + 2*array_k2[i] + 2*array_k3[i] + array_k4[i]) / 6  for i in range(len(y_array))]
        y_array[:] = new_y[:]
        
        i += 1
        x += h
    
    return y_array[:]
def rat_diff(numerator, denominator):
    if hasattr(numerator, 'diff'):
        if hasattr(denominator, 'diff'):
            return numerator.diff() * denominator - denominator.diff() * numerator, denominator ** 2
        else:
            return numerator.diff() * denominator, denominator ** 2
    else:
        if hasattr(denominator, 'diff'):
            return - denominator.diff() * numerator, denominator ** 2
        else:
            return poly([0]), denominator ** 2
    

def rat_ndiff(numerator, denominator, n):
    if n == 0:
        return numerator, denominator
    elif n == 1:
        a, b = rat_diff(numerator, denominator)
        return a, b
    else:
        nu, d = rat_diff(numerator, denominator)
        return rat_ndiff(nu, d, n - 1)

def rat_add(array):
    n = 0
    d = 1
    for i in range(len(array)):
        if i != len(array) - 1:
            p = array[i][0]
            for nj, pj in array[:i] + array[i+1:]:
                p *= pj
        
        else:
            p = array[i][0]
            for nj, pj in array[:i]:
                p *= pj
        
        n += p
    
    for i, j in array:
        d *= j
    
    return n, d

def approx_in(arr, e, item):
    for i in range(len(arr)):
        if abs(arr[i] - item) <= e:
            return i
    return -1

def cmplx_rem(arr, item):
    for i in range(len(arr)):
        if abs(arr[i] - item) == 0:
            arr.pop(i)
            return 


def rat_simplify(p, q):
    p_z = p.roots()[:]
    q_z = q.roots()[:]
    np = p_z[:]
    nq = q_z[:]

    for i in range(len(p_z)):

        ind = approx_in(nq[:], 0.05, p_z[i])
        if ind > -1:

            cmplx_rem(np, p_z[i])
            cmplx_rem(nq, nq[ind])
    s = p.coeffs[-1]
    r = q.coeffs[-1]
    for i in range(len(nq)):
        if nq[i].imag < 10 ** (-7):
            nq[i] = nq[i].real
        if nq[i].real < 10 ** (-7):
            nq[i] = complex(0, nq[i].imag)
            
    for i in range(len(np)):
        if np[i].imag < 10 ** (-7):
            np[i] = np[i].real  
        if np[i].real < 10 ** (-7):
            np[i] = complex(0, np[i].imag) 
                 
    for i in np:
        
        s *= poly([-i, 1])
        
    for i in nq:
        r *= poly([-i, 1])
    
    return s, r

def inv_laplace_tr_rat(numerator, denominator, t):
    roots = denominator.roots()
    lead_coeff = denominator.coeffs[-1]
    if denominator.coeffs[:] == [0 for i in denominator.coeffs[:]]:
        raise Exception(ValueError, "Denominator is zero.")
    i = -1
    while lead_coeff == 0:
        i -= 1
        lead_coeff = denominator.coeffs[i]

    mod_roots = []
    while len(roots) > 1:
        r_arr = [roots[0]]
        for i in range(1, len(roots)):
            if abs(roots[i] - roots[0]) < 0.01:
                r_arr.append(roots[i])
        
        mod_roots.append((sum(r_arr) / len(r_arr), len(r_arr)))
        for i in r_arr:
            roots.remove(i)
    
    mroots = mod_roots[:] + [(roots[0], 1)] if len(roots) > 0 else mod_roots[:]
    
    s = complex(0, 0)
    for i in range(len(mroots)):
        if i < len(mroots) - 1:
            other_terms = mroots[:i] + mroots[i+1:]
        else:
            other_terms = mroots[:i]
        p1 = poly([lead_coeff])
        for j in range(len(other_terms)):
            p1 *= poly([-other_terms[j][0], 1]) ** other_terms[j][1]
        
        root, power = mroots[i]
        new_f = Prod([Div([numerator, p1]), Comp([poly([0, t]), exp()])])
        #p = Div([numerator, p1])
        #n = power - 1
        #S = Sum([Prod([(poly([0, 1]) ** i)(t), rat_ndiff(numerator, p1, n - i)]) for i in range(n + 1)])
        #D = Prod([Comp([poly([0, t]), exp()]), S])
        r = (1 / math.factorial(power - 1)) * ndiff(power - 1, new_f)(root)
        #r = (1 / math.factorial(power - 1)) * D(root)
        s += r
    return s  

def create_poly_pow(arr_r, arr_pow):
    s = 1
    for i in range(len(arr_r)):
        s *= poly([-arr_r[i], 1]) ** arr_pow[i]
    
    return s

def partial_frac_decomp(p, q):
    adj_p_arr = p.coeffs[:]
    while len(adj_p_arr):
        if adj_p_arr[-1] == 0:
            adj_p_arr.pop(-1)
        else:
            break
    if len(adj_p_arr) == 0:
        return 0
    
    adj_q_arr = q.coeffs[:]
    while len(adj_q_arr):
        if adj_q_arr[-1] == 0:
            adj_q_arr.pop(-1)
        else:
            break
    if len(adj_q_arr) == 0:
        raise Exception(ValueError, 'denominator zero.')
    
    adj_p, adj_q = poly(adj_p_arr[:]), poly(adj_q_arr[:]) #rat_simplify(poly(adj_p_arr[:]), poly(adj_q_arr[:]))
    lead_q = adj_q_arr[-1]
    
    q_roots = adj_q.roots()[:]
    q_root_keys = []
    q_root_count = []
    for i in q_roots:
        ind = approx_in(q_root_keys, 0.05, i)
        if ind > -1:
            q_root_count[ind] += 1
        else:
            q_root_keys.append(i)
            q_root_count.append(1)
    
    coeffs = []
    for i in range(len(q_root_keys)):
        arr = [0] * q_root_count[i]
        new_arr_r = q_root_keys[:i] + q_root_keys[i + 1:] if i < len(q_root_keys) - 1 else q_root_keys[:-1]
        new_arr_p = q_root_count[:i] + q_root_count[i + 1:] if i < len(q_root_count) - 1 else q_root_count[:-1]
        nq = create_poly_pow(new_arr_r, new_arr_p)
        for j in range(q_root_count[i]):
            k = j + 1
            t, u = rat_ndiff(adj_p, nq, j)
            if not callable(u):
                u = poly([u])
            if not callable(t):
                t = poly([t])
            d = t(q_root_keys[i]) /  u(q_root_keys[i]) 
            arr[q_root_count[i] - k] = (1/math.factorial(k - 1)) * d / lead_q
        
        coeffs.append((arr[:], q_root_keys[i]))
    return coeffs[:]

def sym_inv_lap_rat(p, q):
    pfd_arr = partial_frac_decomp(p, q)
    fin_func_arr = []
    for coeff, root in pfd_arr:
        z = poly([coeff[i]/math.factorial(i) for i in range(len(coeff))])
        fin_func_arr.append(Prod([z, Comp([poly([0, root]), exp()])]))
    
    return Sum(fin_func_arr[:])


   

        
    
def laplace_tr(f):
    
    if isinstance(f, poly):
        s = []
        for i in range(len(f.coeffs)):
            s.append(Div([f.coeffs[i] * math.factorial(i), poly([0, 1])**(i+1)]))
        
        return Sum(s)
    
    elif isinstance(f, sin):
        return Div([poly([0, 1]), poly([1, 0, 1])]) 
    
    elif isinstance(f, cos):
        return Div([1, poly([1, 0, 1])])  
    
    elif isinstance(f, exp):
        return Div([1, poly([-1, 1])])
    
def sym_laplace_tr(f):
    if isinstance(f, poly):
        s = Div([0, 1])
        for i in range(len(f.coeffs)):
            s = s + Div([f.coeffs[i] * math.factorial(i), poly([0, 1])**(i+1)])
        return s
        
        
    
    elif isinstance(f, sin):
        return Div([poly([0, 1]), poly([1, 0, 1])]) 
    
    elif isinstance(f, cos):
        return Div([1, poly([1, 0, 1])])  
    
    elif isinstance(f, exp):
        return Div([1, poly([-1, 1])])


def solve_ndeg_ode_sys(equations, rhs, init_conds, t):
    '''
    equations=[
        [p11(D), p12(D), ..., p1m(D)],
        [p21(D), p22(D), ..., p2m(D)],
        .
        .
        .
        [pm1(D), pm2(D), ..., pmm(D)]
    ]
    init_conds = [
        [y1(0), Dy1(0), ..., D^(m-1) y1(0)],
        [y2(0), Dy2(0), ..., D^(m-1) y2(0)],
        .
        .
        .
        [ym(0), Dym(0), ..., D^(m-1) ym(0)]
    ]
    rhs = [n1, n2, ..., nm]
    with n_i as real numbers.
    is equivalent to:
    p11(D)y1 + p12(D)y2 + p13(D)y3 + ... + p1m(D)ym = n1
    p21(D)y1 + p22(D)y2 + p23(D)y3 + ... + p2m(D)ym = n2
    .
    .
    .
    pm1(D)y1 + pm2(D)y2 + pm3(D)y3 + ... + pmm(D)ym = nm     
    '''
    laplacian_matrix = matrix(equations[:])
    mod_rhs = []
    for i in range(len(rhs)):
        p = poly([0])
        for j in range(len(equations[i])):
            for k in range(len(equations[i][j].coeffs)-1):
                narr = [init_conds[j][l] for l in range(k)]
            p += poly(narr[:])
        mod_rhs.append((Div([laplace_tr(rhs[i]), poly([0,1])])+p).simplify())
    
    for i in solveLineq(equations, mod_rhs):
        print(i)
    ans = [i.simplify() for i in solveLineq(equations, mod_rhs)]
    
    functions = [inv_laplace_tr_rat(i.arr[0], i.arr[1], t) for i in ans]
    
    return functions[:]

def solve_ndeg_ode_sys_func(equations, rhs, init_conds):
    return lambda t : solve_ndeg_ode_sys(equations, rhs, init_conds, t)

def numeric_inverse_laplace_transform(function, t, order=2, dt=0.001):
    k = order-t
    if k < 0:
        k = 0
    return cnint(lambda x : function(x) * cmath.exp(t * x), lambda t:complex(order+1, t),-100 * k -10, 100, dt=dt) / complex(0, 2*cmath.pi)


def generate_invlaplace_transform_problem(nranges=[-10, 10], max_deg=2, diff_int=0, delay_deg=0):
    p1 = poly.rand(random.randint(1, max_deg), coeff_range=nranges[:])
    p2 = poly.rand(random.randint(0, p1.deg-1), coeff_range=nranges[:])

    rat = Div([p2, p1])
    new_func = rat
    
        
    
    #fin_func = Prod([delay_arr, new_func])
    fin_func = new_func
    if diff_int:
        p1 = poly.rand(random.randint(1, max_deg), coeff_range=nranges[:])
        p2 = poly.rand(random.randint(1, max_deg), coeff_range=nranges[:])
        while p2.deg == p1.deg:
            p2 = poly.rand(random.randint(1, max_deg), coeff_range=nranges[:])
        fin_func = Div([p1, p2])
        alg_diff_arr = [log(), atan()]
        z = alg_diff_arr[random.randint(0, len(alg_diff_arr) - 1)]
        new_func = Comp([fin_func, z])
        fin_func = new_func
        inv_l = lambda t : numeric_inverse_laplace_transform(fin_func, t, order=1)
        return inv_l, fin_func
    
    else:
        inv_l_obj = lambda t : inv_laplace_tr_rat(p2, p1, t)
        
        return inv_l_obj, fin_func

def generate_mult_arithm_item(num_ranges = [1, 10], rat_range=[1, 10], number_of_parts = 1, number_of_variables=3, pure_arithm = True, fun_ranges = [0, 1], inp_ndigits=1, res_ndigits=4):
    poly_arr = []
    arr = [[[0 for k in range(3)] for j in range(3)] for i in range(3)]
    arr[1][1][1] = 1
    for i in range(number_of_parts):
        empty_ppr = [[], [], []]
        ppr_arr = [empty_ppr[:] for j in range(3)]
        sub_arr_n = [1 for j in range(3)]
        rand_arr = [(integer.rand, num_ranges), (lambda arr : rational.rand(arr[:]), rat_range)]
        if not pure_arithm:
            func_arr = [(math.tan, "tan"), (math.sin, "sin"), (math.cos, "cos"), (math.log, "log"), (math.exp, "exp")]
            f, string = random.choice(func_arr)
            rand_arr.append((lambda nranges: DummyFunc(f, string, random.randint(nranges[0], nranges[1]) + round(random.random(), ndigits=inp_ndigits)), fun_ranges[:]))
            
        for j in range(number_of_variables):
            ind = random.randint(0, len(rand_arr) - 1)
            a, b = rand_arr[ind]
            k = a(b[:])
            #print(k)
            sub_arr_n[j] = k.n
            ppr_arr[j] = connect(k.pprint()[:], [[" "], ["*"], [" "]]) if j < number_of_variables - 1 else k.pprint()[:]
        
        p = polymvar(arr[:])
        p.x_ppr, p.y_ppr, p.z_ppr = ppr_arr[0][:], ppr_arr[1][:], ppr_arr[2][:]  
        poly_arr.append([p, sub_arr_n[:]])
    
    ans = 0
    ppr = [[], [], []]
    for p, s in poly_arr:
        #ans += p(*s[:])
        ans += p(*[truncate(i, res_ndigits) for i in s])
        if ppr != [[], [], []]:
            ppr = connect(ppr[:], connect([[" "], ["+"], [" "]], p.pprint()[:]))[:]
        else:
            ppr = connect(ppr[:], p.pprint()[:])[:]
        
    #return truncate(ans, res_ndigits), strpprint(ppr[:])
    return ans, strpprint(ppr[:])

def diffeq_nppr(q, symbol):
    p = q
    prev_ppr=[[" "], [" "], [" "], [symbol], [" "], [" "], [" "]]
    new_array = p.coeffs[:]
    new_array.reverse()
    lines = [[], [], [], [], [], [], []]
    right_pr = [[" "], [" "], [" "], ["("], [" "], [" "], [" "]]
    left_pr = [[" "], [" "], [" "], [")"], [" "], [" "], [" "]]
    nppr = prev_ppr[:]
    if len(prev_ppr[0]) > 1:
        nppr = connect(right_pr, connect(prev_ppr[:], left_pr))
    
    for i in range(len(new_array)):
        pow, coeff_abs, sgn_ppr = p.deg - i, abs(new_array[i]), [[" "], [" "], [" "], [("+" if i != 0 else " ") if sgn(new_array[i]) else "-"], [" "], [" "], [" "]]
        coeff_abs_ppr = [[" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [j for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)]]
        pow_ppr = [[" " for j in "(" + str(pow) + ")"],
                    [" " for j in "(" + str(pow) + ")"],
                    [j for j in "(" + str(pow) + ")"],
                    [" " for j in "(" + str(pow) + ")"],
                    [" " for j in "(" + str(pow) + ")"],
                    [" " for j in "(" + str(pow) + ")"],
                    [" " for j in "(" + str(pow) + ")"]]
        if pow == 1:
            pow_ppr = [[" "], ["'"], [" "], [" "], [" "], [" "], [" "]]
        elif pow == 2:
            pow_ppr = [["  "], ["''"], ["  "], ["  "], ["  "], ["  "], ["  "]]
        elif pow == 3:
            pow_ppr = [["   "], ["'''"], ["   "], ["   "], ["   "], ["   "], ["   "]]
        elif pow == 0:
            pow_ppr = [[], [], [], [], [], [], []]
        if coeff_abs == 1 and pow != 0:
            coeff_abs_ppr = [[], [], [], [], [], [], []]
        if coeff_abs == 0:
            continue
        else:
            if pow != 0:
                lines = connect(lines[:], connect(sgn_ppr[:], connect(coeff_abs_ppr[:], connect(nppr[:], pow_ppr[:]))))[:]
            else:
                lines = connect(lines[:], connect(sgn_ppr[:], connect(coeff_abs_ppr[:], connect(nppr[:], pow_ppr[:]))))[:]
                
    return lines[:]

def npprint_diffeq(array, syms, rhs_nppr):
    arr = []
    plus = [[" "], [" "], [" "], ["+"], [" "], [" "], [" "]]
    eq = [[" "], [" "], [" "], ["="], [" "], [" "], [" "]]
    for i in range(len(array)):
        sub_arr = [[], [], [], [], [], [], []]
        for j in range(len(array[i])):
            symbol = syms[j]
            sub_arr = connect(sub_arr[:], connect(plus, diffeq_nppr(array[i][j], symbol)))[:]
        
        arr.append(strpprint(connect(sub_arr[:], connect(eq, rhs_nppr[i]))))
    
    return arr[:]
def rndFLapDiffeqSys(symbol="x"):
    functions = [sin(), cos(), exp()]
    return functions[random.randint(0, len(functions) - 1)]
def generate_sys_diffeq(n, mdeg, nranges=[-10, 10], rhs_mdeg=2, m=2, fweights=[1, 1, 1, 1, 1, 1, 1, 1, 1], wweights=[1, 1, 1, 1]):
    syms = ["y", "z", "w", "p", "q", "r", "s", "u", "v"]
    sys_arr = [[poly.rand(mdeg, coeff_range=nranges[:]) for i in range(n)] for j in range(n)]
    init_conds = [[random.randint(nranges[0], nranges[1]) for i in range(mdeg)] for j in range(n)]
    rhs_arr = [rndFLapDiffeqSys() for i in range(n)]#[rand_func_iii(nranges=nranges[:], max_deg=rhs_mdeg, n=m, fweights=fweights[:], wweights=wweights[:]) for i in range(n)]

    sols = solve_ndeg_ode_sys_func(sys_arr[:], rhs_arr[:], init_conds[:])
    return "\n".join(npprint_diffeq(sys_arr, syms, [i.npprint()[:] for i in rhs_arr])), init_conds[:], sols


def generate_lap_matrix(dim=3, nranges=[-10, 10], mdeg=2):
    arr = []
    for i in range(dim):
        sub = []
        for j in range(dim):
            p = poly.rand(random.randint(0, mdeg), coeff_range=nranges[:])
            p.variable = "s"
            sub.append(p)
        arr.append(sub[:])

    lap_mat = matrix(arr[:])
    return lap_mat, lap_mat.inverse()

def generate_sys_lap_problem(nranges=[1, 10], dim=3, mdeg=2, mdeg_rhs=2):
    l, l_inv = generate_lap_matrix(dim=dim, nranges=nranges, mdeg=mdeg)
    rhs_arr = []
    for i in range(dim):
        p1 = poly.rand(random.randint(0, mdeg_rhs), coeff_range=nranges[:])
        p2 = poly.rand(random.randint(0, mdeg_rhs), coeff_range=nranges[:])
        z = Div([p1, p2])
        z.arr[0].variable = "s"
        z.arr[1].variable = "s"
        rhs_arr.append([z])


    rhs = matrix(rhs_arr)
    ans = matrix([[i[0].simplify()] for i in (l_inv * rhs).array[:]])
    ans_time_domain = lambda t : matrix([[inv_laplace_tr_rat(i[0].arr[0], i[0].arr[1], t)] for i in ans.array[:]])
    return l, rhs, ans, ans_time_domain


def solve_diffeq_sym(coeffs, rhs_lap_arr, init_vals):
    # coeffs = [1, 2, 3, 4] -> y + 2y' + 3y'' + 4y''
    # init_vals = [a, b, c, ...] -> y(0) = a, y'(0) = b, y''(0) = c, ...
    lhs_lap = poly(coeffs[:])
    rhs_init_poly = 0
    s = poly([0, 1])
    for i in range(len(coeffs)):
        p = 0
        for j in range(1, i + 1):
            p += init_vals[j - 1] * s ** (i - j)
        rhs_init_poly += p * coeffs[i]
    
    p, q = rhs_lap_arr[0] + rhs_init_poly * rhs_lap_arr[1], rhs_lap_arr[1] * lhs_lap
    
    return sym_inv_lap_rat(p, q)

def rand_poly_nice_roots(root_ranges, deg, all_real=True):
    
    if all_real:
        p = poly([1])
        for i in range(deg):
            r = random.randint(root_ranges[0], root_ranges[1])
            p = p * poly([-r, 1])

        return p
    
    else:
        p = poly([1])
        if deg % 2:
            r = random.randint(root_ranges[0], root_ranges[1])
            p = p * poly([-r, 1])
            deg -= 1
        
        for i in range(int(deg / 2)):
            s = random.randint(0, 1)
            a, b =  random.randint(root_ranges[0], root_ranges[1]), random.randint(root_ranges[0], root_ranges[1])
            if s:
                
                p = p * poly([a ** 2 + b ** 2, -2*a, 1])
            else:
                p = p * poly([a * b, -(a+b), 1])
        
        return p
        
        
def rand_diffeq_sym(nranges, deg, mdeg):
    coeffs = rand_poly_nice_roots(nranges[:], deg, all_real=False)
    p, q = rand_poly_nice_roots(nranges[:], mdeg - 1, all_real=False), rand_poly_nice_roots(nranges[:], mdeg, all_real=False)
    arr = poly.rand(deg - 1, coeff_range=nranges[:])
    s = sym_inv_lap_rat(p, q)
    
    ppr = coeffs.npprint(prev_ppr=[[" "], [" "], [" "], ["y"], [" "], [" "], [" "]])
    
    new_array = coeffs.coeffs[:]
    new_array.reverse()
    lines = [[], [], [], [], [], [], []]
    
    nppr = [[" "], [" "], [" "], ["y"], [" "], [" "], [" "]]
    
    for i in range(len(new_array)):
        pow, coeff_abs, sgn_ppr = deg - i, abs(new_array[i].real), [[" "], [" "], [" "], [("+" if i != 0 else " ") if sgn(new_array[i].real) else "-"], [" "], [" "], [" "]]
        if new_array[i].imag != 0:
            coeff_abs = new_array[i]
            sgn_ppr =  [[" "], [" "], [" "], ["+"], [" "], [" "], [" "]]
        coeff_abs_ppr = [[" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [j for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)],
                            [" " for j in str(coeff_abs)]]
        pow_ppr = [[" " for j in range(pow)],
                    [" " for j in range(pow)],
                    ["'" for j in range(pow)],
                    [" " for j in range(pow)],
                    [" " for j in range(pow)],
                    [" " for j in range(pow)],
                    [" " for j in range(pow)]]
        
        if coeff_abs == 1 and pow != 0:
            coeff_abs_ppr = [[], [], [], [], [], [], []]
        if coeff_abs == 0:
            continue
        else:
            
            lines = connect(lines[:], connect(sgn_ppr[:], connect(coeff_abs_ppr[:], connect(nppr[:], pow_ppr[:]))))[:]
            
    #return lines[:]
    ppr = connect(lines[:], connect([["   "], ["   "], ["   "], [" = "], ["   "], ["   "], ["   "]], s.npprint()))
    
    return solve_diffeq_sym(coeffs.coeffs[:], [p, q], arr.coeffs[:]), strpprint(ppr[:]), arr.coeffs[:]

def rand_diffeq_sym_tex(nranges, deg, mdeg):
    coeffs = rand_poly_nice_roots(nranges[:], deg, all_real=False)
    p, q = rand_poly_nice_roots(nranges[:], mdeg - 1, all_real=False), rand_poly_nice_roots(nranges[:], mdeg, all_real=False)
    arr = poly.rand(deg - 1, coeff_range=nranges[:])
    s = sym_inv_lap_rat(p, q)
    
    ppr = coeffs.texify(prev_tex='y')
    
    new_array = coeffs.coeffs[:]
    new_array.reverse()
    lines = [[], [], [], [], [], [], []]
    
    nppr = [[" "], [" "], [" "], ["y"], [" "], [" "], [" "]]
    
    for i in range(len(new_array)):
        pow, coeff_abs, sgn_ppr = deg - i, abs(new_array[i].real), ("+" if i != 0 else " ") if sgn(new_array[i].real) else "-"
        if new_array[i].imag != 0:
            coeff_abs = new_array[i]
            sgn_ppr =  '+'
        coeff_abs_ppr = str(coeff_abs)
        pow_ppr = "".join(["'" for j in range(pow)])
        
        if coeff_abs == 1 and pow != 0:
            coeff_abs_ppr = ""
        if coeff_abs == 0:
            continue
        else:
            
            lines = connect(lines[:], connect(sgn_ppr[:], connect(coeff_abs_ppr[:], connect(nppr[:], pow_ppr[:]))))[:]
            
    #return lines[:]
    ppr = connect(lines[:], connect([["   "], ["   "], ["   "], [" = "], ["   "], ["   "], ["   "]], s.npprint()))
    
    return solve_diffeq_sym(coeffs.coeffs[:], [p, q], arr.coeffs[:]), strpprint(ppr[:]), arr.coeffs[:]

def diff_det(nranges, dim, mat_deg, mdeg):
    mat = matrix.randpoly([dim, dim], mat_deg, coeff_range=nranges[:])
    init_vals =[0 for i in range(len(mat.det().coeffs[:]) - 1)]
    coeffs = mat.det().coeffs[:]
    p, q = rand_poly_nice_roots([-10, 10], mdeg - 1, all_real=False), rand_poly_nice_roots([-10, 10], mdeg, all_real=False)
    s = sym_inv_lap_rat(p, q)
    f = solve_diffeq_sym(coeffs[:], [p, q], init_vals[:])
    return f, s, mat


def find_extrema(func):
    if isinstance(func, list):
        return (func[0].diff()*func[1]-func[1].diff()*func[0]).roots()
    
    else:
        f = func.diff()
        return [f.newtonsmethod()]

def solve_eq(rhs, lhs):
    z = Sum([rhs, Prod([-1, lhs])])
    return z.newtonsmethod()

def tangent_line(function, x):
    return poly([function(x) - function.diff()(x) * x, function.diff()(x)])

def arithmetic_elems(nranges, n):
    result = random.randint(nranges[0], nranges[1])
    curr_str = str(result)
    res_str = str(result)
    c = 0
    for i in range(n):
        op = random.randint(1, 4)
        n2 = random.randint(nranges[0], nranges[1])
        while n2 == 0:
            n2 = random.randint(nranges[0], nranges[1])
    
        if op == 1: # +
            curr_str = (curr_str[:] + "+" + str(n2))[:]
            res_str = (res_str[:] + "+" + str(n2))[:]
            result += n2
        
        elif op == 2: # -
            curr_str = (curr_str[:] + "-" + str(n2))[:]
            res_str = (res_str[:] + "-" + str(n2))[:]
            result -= n2
        
        elif op == 3: # *
            curr_str = ( curr_str[:] + "*" + str(n2))[:]
            res_str = (res_str[:] + "*" + str(n2))[:]
            result *= n2
        
        elif op == 4: # /
            l = max(len(curr_str), len(str(n2)))
            k = min(len(curr_str), len(str(n2)))
            curr_str = curr_str + "\n" + "".join(['-' for j in range(l)]) + "\n"+ "".join([" " for j in range(int((abs(l - k)) / 2)-1)]) + str(n2)
            res_str = ("(" + res_str[:] + ")" + "/" + "(" +str(n2))[:]
            c += 1
    for i in range(c):
        res_str = res_str + ")"
    return curr_str, eval(res_str)
