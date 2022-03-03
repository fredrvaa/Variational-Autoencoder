from subprocess import call
from keras.callbacks import Callback

class LossHistory(Callback):
    def on_test_begin(self, logs=None):
        self.losses = []

    def on_test_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))