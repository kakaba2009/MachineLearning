import itertools

S, r = input().split(' ')

S = list(S)
S.sort()
r = int(r)

C = itertools.combinations_with_replacement(S, r)
for i in C:
    print(*i, sep='')
