from numpy import (
    arange,
    array,
    cos,
    linspace,
    pi,
    sign,
    sin,
    sum,
    append,
    random,
    power,
)
from random import gauss
from math import sqrt


def main():
    train(10)


def train(n_params):
    Xs = linspace(-10, 10, num=100)
    Y_true = square(Xs, 1)

    a0 = gauss(0, 1)
    an = random.normal(0, n_params, 1)
    bn = random.normal(0, n_params, 1)
    L = gauss(0, 1)
    learning_rate = 0.0001

    errors = []
    for i in range(2000):
        for x, y_true in zip(Xs, Y_true):
            a0, an, bn, L = descent(x, a0, L, an, bn, y_true, learning_rate)

        Y_pred = predict(Xs, a0, L, an, bn)
        errors.append(mse(Y_true, Y_pred))

    Y_pred = predict(Xs, a0, L, an, bn)
    return Xs, Y_true, Y_pred, errors


def square(x, period):
    return sign(sin((2 * pi * x) / period))


def fourier(x, a0, L, an, bn):
    na = arange(1, len(an) + 1)
    nb = arange(1, len(bn) + 1)
    return a0 + sum(an * cos(na * x * (pi / L))) + sum(bn * sin(nb * x * (pi / L)))


def fourier_derivative(x, a0, L, an, bn, y, y_pred):
    n = arange(1, len(an) + 1)
    a0_d = -2 * (y - y_pred)
    an_d = -2 * (y - y_pred) * cos(n * x * pi / L)
    bn_d = -2 * (y - y_pred) * sin(n * x * pi / L)
    L_d = (
        -2
        * (y - y_pred)
        * (
            sum(an * sin(n * x * pi / L) * n * x * pi / L**2)
            - sum(bn * cos(n * x * pi / L) * n * x * pi / L**2)
        )
    )
    return a0_d, an_d, bn_d, L_d


def descent(x, a0, L, an, bn, y, learning_rate):
    y_pred = fourier(x, a0, L, an, bn)
    a0_d, an_d, bn_d, L_d = fourier_derivative(x, a0, L, an, bn, y, y_pred)

    a0 = a0 - learning_rate * a0_d
    an = an - learning_rate * an_d
    bn = bn - learning_rate * bn_d
    L = L - learning_rate * L_d

    return a0, an, bn, L


def mse(true, pred):
    return sum(power(true - pred, 2)) / len(true)


def predict(Xs, a0, L, an, bn):
    Y_pred = array([])
    for x in Xs:
        y_pred = fourier(x, a0, L, an, bn)
        Y_pred = append(Y_pred, y_pred)

    return Y_pred


def squareParamtest(n, h):
    p = array([])
    for i in range(1, n + 1):
        v = (4 * h) / (i * pi)
        p = append(p, v)
    return p


if __name__ == "__main__":
    main()
