import cmath

cn = input().strip()

num = complex(cn)
print(abs(num))
print(cmath.phase(num))
