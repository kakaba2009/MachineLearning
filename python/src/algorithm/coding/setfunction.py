n = int(input())
s = set(map(int, input().split()))

N = int(input())

for i in range(N):
    cmd = input()
    if "pop" in cmd:
        s.pop()
    elif "remove" in cmd:
        cmd, element = cmd.split()
        element = int(element)
        if element in s:
            s.remove(element)
    elif "discard" in cmd:
        cmd, element = cmd.split()
        element = int(element)
        if element in s:
            s.discard(element)

print(sum(s))
