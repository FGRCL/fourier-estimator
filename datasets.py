from numpy import (
    linspace,
    pi,
    sign,
    sin,
)

def generate_square_wave_dataset(min, max, num):
    X = linspace(min, max, num)
    Y = square(X, 1)
    return X, Y

def square(x, period):
    return sign(sin((2 * pi * x) / period))

def generate_line_dataset(min, max, num, a=1, b=0):
    X = linspace(min, max, num)
    Y = line(X, a, b)
    return X, Y

def line(x, a, b):
    return a*x+b

