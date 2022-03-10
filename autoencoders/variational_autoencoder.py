from enum import Enum
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tfk = tf.keras
tfd = tfp.distributions


class Measure(Enum):
    mean = tfd.Distribution.mean
    mode = tfd.Distribution.mode
    sample = tfd.Distribution.sample

def negative_log_likelihood(x, rx):
    return -rx.log_prob(x)


class VariationalAutoencoder:
    def __init__(self,
                prior_distribution: tfd.Distribution,
                encoding_dim: int = 8,
                image_dim: tuple[int, int] = (28, 28),
                file_name: str = "./models/variational_autoencoder/autoencoder",
                ):
        self.prior_distribution: tfd.Distribution = prior_distribution
        self.encoding_dim: int = encoding_dim
        self.image_dim: tuple[int, int] = image_dim
        self.file_name: str = file_name

        self.encoder = tfk.Sequential([
            tfk.layers.InputLayer((*self.image_dim, 1)),
            tfk.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
            tfk.layers.Conv2D(16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(32, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(32, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(64, kernel_size=7, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Flatten(),
            tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(encoding_dim), activation=None),
            tfp.layers.MultivariateNormalTriL(encoding_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior_distribution, weight=1.0))
        ], name='Encoder')

        self.decoder = tfk.Sequential([
            tfk.layers.InputLayer((encoding_dim,)),
            tfk.layers.Reshape((1, 1, encoding_dim)),
            tfk.layers.Conv2DTranspose(32, kernel_size=7, strides=1, activation=tf.nn.leaky_relu, padding='valid'),
            tfk.layers.Conv2DTranspose(32, kernel_size=7, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(32, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(16, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(16, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2DTranspose(16, kernel_size=5, strides=1, activation=tf.nn.leaky_relu, padding='same'),
            tfk.layers.Conv2D(1, kernel_size=5, strides=1, activation=None, padding='same'),
            tfk.layers.Flatten(),
            tfp.layers.IndependentBernoulli(self.image_dim, tfd.Bernoulli.logits),
        ], name='Decoder')

        self.loss = negative_log_likelihood #tfk.losses.binary_crossentropy#lambda x, rv_x: -rv_x.log_prob(x) # Negative likelihood

        self.model = tfk.Model(name='VariationalAutoencoder', inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs[0]))
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

    def encode(self, x: np.ndarray, output_measure: Measure = Measure.mean) -> np.ndarray:
        n_channels = x.shape[-1]
        encoded = tf.Variable(tf.zeros((x.shape[0], self.encoding_dim, n_channels)))
        for n in range(n_channels):
            encoded[:, :, n].assign(output_measure(self.encoder(x[:, :, :, n])))
        return encoded

    def decode(self, z: np.ndarray, output_measure: Measure = Measure.mean) -> np.ndarray:
        n_channels = z.shape[-1]
        decoded = tf.Variable(tf.zeros((z.shape[0], *self.image_dim, n_channels)))
        for n in range(n_channels):
            decoded[:, :, :, n].assign(output_measure(self.decoder(z[:, :, n])))
        return decoded

    def __call__(self, x, output_measure: Measure = Measure.mean) -> np.ndarray:
        return self.decode(self.encode(x), output_measure=output_measure)

    def reconstruction_loss(self, x: np.ndarray) -> np.ndarray:
        return self.loss(np.squeeze(x), self.decoder(self.encoder(x))).numpy() 

    def sampled_loss(self, x: np.ndarray, N: int = 10000) -> np.ndarray:
        n_channels = x.shape[-1]
        z = tf.reshape(self.prior_distribution.sample(N * n_channels), (N, self.encoding_dim, n_channels))
        loss = np.zeros((x.shape[0], n_channels))
        for n in range(n_channels):
            print(f'Channel {n+1}')
            x_hat = self.decoder(z[:, :, n])
            x_c = x[:, :, :, n]
            for i, example in enumerate(x_c):
                if i % int(x.shape[0] / 10) == 0:
                    print(f'{i} / {x.shape[0]}')
                tiled = np.repeat(example[np.newaxis, :, :], N, axis=0)
                loss[i][n] = np.mean(self.loss(tiled, x_hat))
            print(f'{x.shape[0]} / {x.shape[0]}')
        return np.mean(loss, axis=1)

    def summary(self) -> None:
        self.encoder.summary()
        self.decoder.summary()