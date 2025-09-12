from keras import Sequential
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from numpy import float32, reshape
from datasets import generate_line_dataset
from FourierLayer import FourierLayer

def main():
    X, Y = generate_line_dataset(-10, 10, 256)
    X = reshape(X, (256, 1, 1)).astype(float32)

    model = Sequential([
        FourierLayer(10),
    ])

    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=1e-3),
    )

    model.fit(X, Y, 256, 1000)
 

if __name__ == "__main__":
    main()
