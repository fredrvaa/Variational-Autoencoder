import numpy as np
import tensorflow as tf

tfk = tf.keras

class Autoencoder:
    def __init__(self,
                 encoding_dim: int = 8,
                 image_dim: tuple[int, int] = (28, 28),
                 file_name: str = "./models/autoencoder/autoencoder"
                 ):
        self.encoding_dim: int = encoding_dim
        self.image_dim: tuple[int, int] = image_dim
        self.file_name: str = file_name

        self.encoder = tfk.Sequential([
            tfk.layers.InputLayer((*self.image_dim, 1)),
            tfk.layers.Conv2D(16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(32, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(32, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(64, kernel_size=7, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(64, kernel_size=7, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Flatten(),
            tfk.layers.Dense(encoding_dim, activation=tf.nn.sigmoid),
        ], name='Encoder')

        self.decoder = tfk.Sequential([
            tfk.layers.InputLayer((encoding_dim,)),
            tfk.layers.Reshape((1, 1, encoding_dim)),
            tfk.layers.Conv2DTranspose(32, kernel_size=7, strides=1, activation=tf.nn.leaky_relu, padding='valid'),
            tfk.layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(32, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(16, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(16, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(1, kernel_size=5, strides=1, activation=tf.nn.sigmoid, padding='same'),
            tfk.layers.Reshape(self.image_dim)
        ], name='Decoder')

        self.loss = tfk.losses.binary_crossentropy

        self.model = tfk.Model(name='VariationalAutoencoder', inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs))
        self.model.compile(loss=self.loss, optimizer=tfk.optimizers.Adam(learning_rate=1e-3))

        self.done_training: bool = self.load_weights()

    def train(self, x_train: np.ndarray, epochs: int = 10, batch_size: int = 256, force_relearn: bool = False) -> None:
        if force_relearn or self.done_training is False:
            # Stack channels into grayscale images
            if len(x_train.shape) == 4 and x_train.shape[-1] > 1:
                x_train = x_train.transpose(3,0,1,2).reshape(-1, x_train.shape[1], x_train.shape[2], 1)
            x_train = np.squeeze(x_train)

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
        encoded = tf.Variable(tf.zeros((x.shape[0], self.encoding_dim, n_channels)))
        for n in range(n_channels):
            encoded[:, :, n].assign(self.encoder(np.expand_dims(x[:, :, :, n], axis=-1)))
        return encoded

    def decode(self, z: np.ndarray) -> np.ndarray:
        n_channels = z.shape[-1]
        decoded = np.zeros((z.shape[0], *self.image_dim, n_channels))
        for n in range(n_channels):
            decoded[:, :, :, n] = self.decoder(z[:, :, n])
        return decoded

    def __call__(self, x) -> np.ndarray:
        return self.decode(self.encode(x))

    def reconstruction_loss(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self.loss(x, self(x)).numpy(), axis=(1,2))
        # loss_history = LossHistory()
        # self.model.evaluate(x, x, batch_size=1, callbacks=[loss_history])
        # return np.array(loss_history.losses)

    def summary(self) -> None:
        self.encoder.summary()
        self.decoder.summary()