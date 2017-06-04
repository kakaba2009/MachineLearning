import functools
S = list(input())

A = list(filter(lambda x : x.isalpha(), S))
D = list(filter(lambda x : x.isdigit(), S))
L = list(filter(lambda x : x.islower(), A))
U = list(filter(lambda x : x.isupper(), A))
O = list(filter(lambda x : int(x) % 2 == 1, D))
E = list(filter(lambda x : int(x) % 2 == 0, D))
R = list(map(lambda x : x.sort(), [L,U,O,E]))
if len(L) > 0:
    L = functools.reduce(lambda x,y : x+y, L)
else:
    L = ""
if len(U) > 0:
    U = functools.reduce(lambda x,y : x+y, U)
else:
    U = ""
if len(O) > 0:
    O = functools.reduce(lambda x,y : x+y, O)
else:
    O = ""
if len(E) > 0:
    E = functools.reduce(lambda x,y : x+y, E)
else:
    E = ""
R = L + U + O + E
print(R)
