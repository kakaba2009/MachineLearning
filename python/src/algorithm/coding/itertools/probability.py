import itertools

N = int(input())
S = input().split(' ')
K = int(input())

S = list(S)
C = list(itertools.combinations(S, K))
T = len(C)

a = list(filter(lambda x: 'a' in x, C))

print(len(a)/T)

