from keras import Sequential, layers, Model


class Autoencoder:
    def __init__(self):
        self.encoder = Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(16, activation='relu')
        ], name='encoder')
        self.decoder = Sequential([
            layers.Dense(784, activation='relu'),
            layers.Reshape((28, 28, 1))
        ], name='decoder')
        self.model = Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)
        self.model.compile(loss='mse', optimizer='Adam')
        self.model.build((None, 28, 28, 1))
        self.model.summary()
