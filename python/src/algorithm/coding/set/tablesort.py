
table = []

n, m = map(int, input().split())

for i in range(n):
    row = list(map(int, input().split()))
    table.append(row)

k = int(input())

table.sort(key=lambda x: x[k])

for row in table:
    print(*row)
