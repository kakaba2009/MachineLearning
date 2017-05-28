n = int(input())
s = set(map(int, input().split()))

N = int(input())

for i in range(N):
    cmd = input()
    B = set(map(int, input().split()))
    if "symmetric_difference_update" in cmd:
        s.symmetric_difference_update(B)
    elif "intersection_update" in cmd:
        s.intersection_update(B)
    elif "difference_update" in cmd:
        s.difference_update(B)
    elif "update" in cmd:
        s.update(B)

print(sum(s))
