import itertools

S, r = input().split(" ")

S = list(S)
S.sort()

R = int(r)

for r in range(1, R+1):
    C = list(itertools.combinations(S, r))
    for i in C:
        print(*i, sep='')

