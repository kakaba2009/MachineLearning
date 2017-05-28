n, m = map(int, input().split())

N = list(map(int, input().split()))
A = list(map(int, input().split()))
A = set(A)
B = list(map(int, input().split()))
B = set(B)

score = 0

for i in N:
    if i in A:
        score += 1
    if i in B:
        score -= 1

print(score)
