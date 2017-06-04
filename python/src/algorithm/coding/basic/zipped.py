n, x = list(map(int, input().split()))

S = []

for i in range(x):
    S.append(list(map(float, input().split())))
Z = zip(*S)

[print(sum(t)/x) for t in Z]
