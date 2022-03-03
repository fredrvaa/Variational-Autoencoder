import argparse

import numpy as np
from matplotlib import pyplot as plt

from autoencoders.autoencoder import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('-n', help='Number of examples to generate', default=5, type=int)
args = parser.parse_args()

autoencoder = Autoencoder()
z = np.random.uniform(size=(args.n, autoencoder.encoding_dim))
decoded = autoencoder.decode(z)

# Visualize
fig, ax = plt.subplots(2, args.n, figsize=(12, 6))
for n in range(args.n):
    ax[0][n].imshow(z[n].reshape(autoencoder.encoding_dim, 1), cmap='gray')
    ax[1][n].imshow(decoded[n], cmap='gray')
    ax[0][n].set_xticks([])
ax[0][0].set_ylabel('Random encoding')
ax[1][0].set_ylabel('Generated')
fig.align_ylabels()
plt.show()