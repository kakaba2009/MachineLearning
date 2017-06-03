import itertools

S = input()
S = list(S)
S = list(map(int, S))
result = []
for k, g in itertools.groupby(S):
    result.append(tuple((len(list(g)), k)))

print(*result, sep=' ')
