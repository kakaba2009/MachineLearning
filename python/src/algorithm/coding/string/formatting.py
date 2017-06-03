def print_formatted(number):
    # your code goes here
    w = len(str(bin(number)).replace('0b',''))
    for i in range(1, number+1):
        b = bin(int(i)).replace('0b','').rjust(w, ' ')
        o = oct(int(i)).replace('0o','', 1).rjust(w, ' ')
        h = hex(int(i)).replace('0x','').upper().rjust(w, ' ')
        d = str(i).rjust(w, ' ')
        print(d, o, h, b)

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

