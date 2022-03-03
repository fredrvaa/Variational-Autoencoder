import argparse
import matplotlib.pyplot as plt
from autoencoders.autoencoder import Autoencoder
from utils.stacked_mnist import StackedMNISTData, DataMode
from utils.verification_net import VerificationNet

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--encoding_dim', help='Encoding/latent dimensionality in autoencoder', default=8, type=int)
args = parser.parse_args()

# Get mnist data
gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE)
x_train, y_train = gen.get_full_data_set(training=True)
x_test, y_test = gen.get_full_data_set(training=False)

# Fit autoencoder
autoencoder = Autoencoder(args.encoding_dim)
autoencoder.train(x_train, epochs=20, batch_size=256)

# Fit verification net
net = VerificationNet(force_learn=False)
net.train(generator=gen, epochs=10)

# Check coverage, predictability, and accuracy
x_pred = autoencoder(x_test)
cov = net.check_class_coverage(data=x_pred, tolerance=.8)
pred, acc = net.check_predictability(data=x_pred, correct_labels=y_test)
print(f"Coverage: {100*cov:.2f}%")
print(f"Predictability: {100*pred:.2f}%")
print(f"Accuracy: {100 * acc:.2f}%")

# Visualize example
ex = x_test[10]

fig, ax = plt.subplots(1, 3)
ax[0].imshow(ex, cmap='gray')
ax[0].set_title('Example')
encoded = autoencoder.encoder.predict(ex.reshape(-1, 28, 28, 1))[0]
ax[1].imshow(encoded.reshape(args.encoding_dim, 1), cmap='gray')
ax[1].set_title('Encoded')
decoded = autoencoder.model.predict(ex.reshape(-1, 28, 28, 1))[0]
ax[2].imshow(decoded, cmap='gray')
ax[2].set_title('Decoded')
plt.show()

