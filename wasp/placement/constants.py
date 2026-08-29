KB = 1024
MB = 1024 * KB
GB = 1024 * MB
TB = 1024 * GB

SECOND = 1
MINUTE = 60 * SECOND
MILLISECOND = 1e-3
MICROSECOND = 1e-6

BYTE_SIZE = 2

M = 1e15

def get_divisors(n):
    divisors = []
    for i in range(2, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors
