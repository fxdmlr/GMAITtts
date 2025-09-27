from utils import *
max_deg = 2
nranges = [-10, 10]
q = polymvar.rand_n(d=4, nrange=nranges[:])
print(q)
p = q.array[:]
s = Sum([])
for i in range(len(p)):
    for j in range(len(p[i])):
        for k in range(len(p[i][j])):

            a, b, c = Comp([sin(), poly([0, 1])**i]) , Comp([cos(), poly([0, 1])**j]) , Comp([exp(), poly([0, 1])**k])
            e = 1
            if i != 0:
                e *= a
            if j != 0:
                e *= b
            if k != 0:
                e *= c
            s +=  e * p[i][j][k]

print(s)