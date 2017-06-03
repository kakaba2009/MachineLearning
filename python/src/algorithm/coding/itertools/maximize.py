import itertools

K, M = list(map(int, input().split(' ')))
L = []
for i in range(K):
    t = list(map(int, input().split(' ')))
    t = t[1:]
    t = [x**2 for x in t]
    L.append(t)

P = itertools.product(*L)

R = map(lambda x: (sum(x) % M), P)

print(max(R))
