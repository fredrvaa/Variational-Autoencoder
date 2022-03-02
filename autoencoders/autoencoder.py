from tensorflow import keras
from keras import Sequential, layers, losses
from keras.losses import binary_crossentropy

class Autoencoder:
    def __init__(self):
        self.encoder = Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
        ])
        self.decoder = Sequential([
            layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(784, activation='relu'),
            layers.Reshape((28, 28, 1))
        ])

        self.model = Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(loss=losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=.0001))
        self.model.build((None, 28, 28, 1))
        self.model.summary()
