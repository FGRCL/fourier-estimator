from keras.layers import Layer
from numpy import (
    arange,
    pi
)
from tensorflow import (
    cast,
    float32,
    reshape,
    reduce_sum,
    sin,
    cos,
)



class FourierLayer(Layer):
    def __init__(self, n_params=32):
        super().__init__()
        self.a0 = self.add_weight(
            shape=((1,)),
            initializer="random_normal",
            trainable=True,
        )
        self.an = self.add_weight(
            shape=((n_params,)),
            initializer="random_normal",
            trainable=True,
        )

        self.bn = self.add_weight(
            shape=((n_params,)),
            initializer="random_normal",
            trainable=True,
        )
        self.n = cast(reshape(arange(1, n_params+1), (n_params, 1)), float32)
        self.L = self.add_weight(
            shape=((1,)),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        a = (self.a0/2)
        b = reduce_sum(self.an * cos(self.n * x * (pi / self.L)), axis=1)
        c = reduce_sum(self.bn * sin(self.n * x * (pi / self.L)), axis=1)
        result = a+b+c
        return result
