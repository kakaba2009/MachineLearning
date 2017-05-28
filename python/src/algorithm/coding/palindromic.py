
n = int(input())

slist = input().split()
ilist = list(map(int, slist))

if all(x > 0 for x in ilist):
    if any(x == ''.join(letter for letter in reversed(x)) for x in slist):
        print("True")
    else:
        print("False")
else:
    print("False")

