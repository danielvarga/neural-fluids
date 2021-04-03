import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split


def read_data(filename):
    a = np.load(open(filename, "rb"))
    x = a[:-1]
    y = a[1:]
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    return x, y


x, y = read_data(sys.argv[1])

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, shuffle=True)

input_shape = x.shape[1:]
print("input_shape", input_shape)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same"),
        layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same"),
        layers.Conv2D(5, kernel_size=(5, 5), activation="linear", padding="same")
    ]
)

model.summary()

batch_size = 128
epochs = 100

model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["mean_squared_error"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

model.save("model.mdl")
