from enum import auto
import matplotlib.pyplot as plt
from autoencoders.autoencoder import Autoencoder
from utils.stacked_mnist import StackedMNISTData, DataMode

gen = StackedMNISTData(
    mode=DataMode.MONO_BINARY_COMPLETE,
    default_batch_size=9)
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

autoencoder = Autoencoder()
autoencoder.model.fit(x_train, x_train, epochs=3)

ex = x_test[10]
plt.imshow(ex, cmap='gray')
plt.show()
encoded = autoencoder.encoder.predict(ex.reshape(-1, 28, 28, 1))[0]
plt.imshow(encoded.reshape(4, 4), cmap='gray')
plt.show()
decoded = autoencoder.model.predict(ex.reshape(-1, 28, 28, 1))[0]
plt.imshow(decoded, cmap='gray')
plt.show()