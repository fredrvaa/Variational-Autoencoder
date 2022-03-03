import numpy as np
from tensorflow import keras
from keras import Sequential, layers
from keras.losses import binary_crossentropy

from utils.loss_history import LossHistory


class Autoencoder:
    def __init__(self,
                 encoding_dim: int = 8,
                 file_name: str = "./models/autoencoder/autoencoder"
                 ):
        self.encoding_dim: int = encoding_dim
        self.file_name: str = file_name

        self.encoder = Sequential([
            layers.InputLayer((28, 28, 1)),
            layers.Conv2D(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(encoding_dim),
        ], name='Encoder')

        self.decoder = Sequential([
            layers.InputLayer((encoding_dim,)),
            layers.Dense(16*32, activation='relu'),
            layers.Reshape(target_shape=(16, 1, 32)),
            layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same'),
            layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='same'),
            layers.Flatten(),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ], name='Decoder')

        self.model = Sequential(name='Autoencoder')
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(loss=binary_crossentropy, optimizer=keras.optimizers.Adam())
        #self.model.build((None, 28, 28, 1))

        self.done_training: bool = self.load_weights()

    def train(self, x_train: np.ndarray, epochs: int = 10, batch_size: int = 256, force_relearn: bool = False) -> None:
        if force_relearn or self.done_training is False:
            # Stack channels into grayscale images
            x_train = x_train.transpose(3,0,1,2).reshape(-1, x_train.shape[1], x_train.shape[2])
            self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=2)
            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training
    
    def load_weights(self) -> bool:
            # noinspection PyBroadException
            try:
                self.model.load_weights(filepath=self.file_name)
                # print(f"Read model from file, so I do not retrain")
                done_training = True
            except Exception:
                print(f"Could not read weights for autoencoder from file. Must retrain...")
                done_training = False

            return done_training

    def encode(self, x: np.ndarray) -> np.ndarray:
        n_channels = x.shape[-1]
        encoded = np.zeros((x.shape[0], self.encoding_dim, n_channels))
        for n in range(n_channels):
            encoded[:, :, n] = self.encoder.predict(x[:, :, :, n])
        return encoded

    def decode(self, z: np.ndarray) -> np.ndarray:
        n_channels = z.shape[-1]
        image_shape = self.decoder.layers[-1].output_shape[1:]
        decoded = np.zeros((z.shape[0],) + image_shape + (n_channels, ))
        for n in range(n_channels):
            decoded[:, :, :, n] = self.decoder.predict(z[:, :, n])
        return decoded

    def __call__(self, x) -> np.ndarray:
        return self.decode(self.encode(x))
        #return self.model.predict(x)

    def reconstruction_loss(self, x: np.ndarray) -> np.ndarray:
        loss_history = LossHistory()
        self.model.evaluate(x, x, batch_size=1, callbacks=[loss_history])
        return np.array(loss_history.losses)

    def summary(self) -> None:
        self.encoder.summary()
        self.decoder.summary()