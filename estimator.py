from numpy import (
    arange,
    cos,
    pi,
    reshape,
    sin,
    sum,
    random,
    power,
    average
)
from tqdm import tqdm

class FourierEstimator():
    def __init__(self, n_params, learning_rate, training_steps):
        self.n_params = n_params
        self.a0 = random.normal(0, 1, 1)
        self.an = random.normal(0, 1, n_params)
        self.bn = random.normal(0, 1, n_params)
        self.L = random.normal(0, 1, 1)
        self.n = arange(1, n_params+1)
        self.learning_rate = learning_rate
        self.training_steps = training_steps

        self.errors = []

    def fit(self, X, Y):
        self.errors = []
        trange = tqdm(range(self.training_steps), total=self.training_steps, leave=True)
        for _ in trange:
            Y_pred = self.predict(X)
            self.descent(X, Y, Y_pred)
            error = self.mse(Y, Y_pred)
            self.errors.append(error)
            trange.set_description(f"error: {error}")

    def mse(self, true, pred):
        true = reshape(true, (len(true), 1))
        return sum(power(true - pred, 2)) / len(true)

    def descent(self, x, y, y_pred):
        x = reshape(x, (len(x), 1))
        y = reshape(y, (len(y), 1))
        a0_D, an_D, bn_D, L_D = self.fourier_derivative(x, y, y_pred) 

        a0_d = average(a0_D, 0)
        an_d = average(an_D, 0)
        bn_d = average(bn_D, 0)
        L_d = average(L_D, 0)

        self.a0 = self.a0 - self.learning_rate * a0_d
        self.an = self.an - self.learning_rate * an_d
        self.bn = self.bn - self.learning_rate * bn_d
        self.L = self.L - self.learning_rate * L_d

    def fourier_derivative(self, x, y, y_pred):
        a0_d = -(y - y_pred)
        an_d = -2 * (y - y_pred) * cos(self.n * x * (pi / self.L))
        bn_d = -2 * (y - y_pred) * sin(self.n * x * pi / self.L)
        L_d = (
            -2
            * (y - y_pred)
            * (
                sum(self.an * sin(self.n * x * pi / self.L) * self.n * x * pi / self.L**2, axis=1, keepdims=True)
                - sum(self.bn * cos(self.n * x * pi / self.L) * self.n * x * pi / self.L**2, axis=1, keepdims=True)
            )
        )
        return a0_d, an_d, bn_d, L_d

    def fourier(self, x):
        return (self.a0/2) \
            + sum(self.an * cos(self.n * x * (pi / self.L)), axis=1) \
            + sum(self.bn * sin(self.n * x * (pi / self.L)), axis=1)

    def predict(self, X):
        X = reshape(X, (len(X), 1))
        Y = self.fourier(X)
        Y = reshape(Y, (len(Y), 1))
        return Y

