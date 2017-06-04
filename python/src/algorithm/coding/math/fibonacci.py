cube = lambda x: x ** 3  # complete the lambda function
memo = {}

def fibonacci1(n):
    if n == 0 or n == 1:
        return n

    if n in memo:
        return memo[n]
    else:
        fib = fibonacci1(n - 1) + fibonacci1(n - 2)
        memo[n] = fib
        return fib


def fibonacci(n):
    r = []
    for i in range(n):
        r.append(fibonacci1(i))
    return r

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
