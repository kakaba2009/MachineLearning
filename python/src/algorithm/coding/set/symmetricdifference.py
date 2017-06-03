m = int(input())
mlist = list(map(int, input().split()))
mset = set(mlist)

n = int(input())
nlist = list(map(int, input().split()))
nset = set(nlist)

sd = mset.symmetric_difference(nset)

sl = list(sd)

sl.sort()

for s in sl:
    print(s)
