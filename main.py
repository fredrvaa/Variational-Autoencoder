import matplotlib.pyplot as plt
from autoencoders.autoencoder import Autoencoder
from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet

gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE)
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

autoencoder = Autoencoder()
autoencoder.model.fit(x_train, x_train, epochs=20, batch_size=256)

# Example
ex = x_test[10]
plt.imshow(ex, cmap='gray')
plt.show()
# encoded = autoencoder.encoder.predict(ex.reshape(-1, 28, 28, 1))[0]
# plt.imshow(encoded.reshape(3, 3), cmap='gray')
# plt.show()
decoded = autoencoder.model.predict(ex.reshape(-1, 28, 28, 1))[0]
plt.imshow(decoded, cmap='gray')
plt.show()
print(x_test.shape)
x_pred = autoencoder.model.predict(x_test)

# Train verification net
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=5)
# I have no data generator (VAE or whatever) here, so just use a sampled set
cov = net.check_class_coverage(data=x_pred, tolerance=.8)
pred, acc = net.check_predictability(data=x_pred, correct_labels=y_test)
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")
