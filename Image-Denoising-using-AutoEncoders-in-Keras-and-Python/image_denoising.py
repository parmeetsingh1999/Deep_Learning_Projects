# Importing libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Load dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Visualize a sample image

plt.imshow(x_train[0], cmap = 'gray')

# Checking the shape of the training and testing data

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Perform the data visualization

i = random.randint(1,60000)
plt.imshow(x_train[i], cmap = 'gray')
label = y_train[i]
print(label)

# Viewing image in grid format

W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
axes = axes.ravel()
n_training = len(x_train)

for i in np.arange(0, W_grid * L_grid):
  index= np.random.randint(0, n_training)
  axes[i].imshow(x_train[index])
  axes[i].axis('off')

# Perform data preprocessing
#Normalize data

x_train = x_train/255
x_test = x_test/255
print(x_train)

#Adding noise

noise_factor = 0.3
noise_dataset = []

for img in x_train:
  noisy_img = img+noise_factor * np.random.randn(*img.shape)
  noisy_img = np.clip(noisy_img, 0, 1)
  noise_dataset.append(noisy_img)

plt.imshow(noise_dataset[22], cmap = 'gray')

noise_dataset = np.array(noise_dataset)
plt.imshow(noise_dataset[22], cmap = 'gray')

noise_factor = 0.1
noise_test_set = []

for img in x_test:
  noisy_img = img + noise_factor * np.random.randn(*img.shape)
  noisy_img = np.clip(noisy_img, 0, 1)
  noise_test_set.append(noisy_img)

noise_test_set = np.array(noise_test_set)
plt.imshow(noise_test_set[20], cmap = 'gray')

# Build and train autoencoder deep learning model

autoencoder = tf.keras.models.Sequential()

#Encoder

autoencoder.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, strides = 2, padding = 'same', input_shape = (28,28,1)))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, strides = 1, padding = 'same'))

#Decoder

autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size = 3, strides = 2, activation = 'sigmoid', padding = 'same'))

#Compile

autoencoder.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.001))
autoencoder.summary(line_length = None, positions = None, print_fn = None)
autoencoder.fit(noise_dataset.reshape(-1,28,28,1), x_train.reshape(-1,28,28,1), epochs = 10, batch_size = 200, validation_data = (noise_test_set.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1)))

# Evaluate trained model performance

evaluation = autoencoder.evaluate(noise_test_set.reshape(-1,28,28,1), x_test.reshape(-1,28,28,1))
print("Test loss: ", evaluation)

predicted = autoencoder.predict(noise_test_set[:10].reshape(-1,28,28,1))

fig, axes = plt.subplots(nrows = 2, ncols = 10, sharex = True, sharey = True, figsize = (20,4))
for images, rows in zip([noise_test_set[:10], predicted], axes):
  for img, ax in zip(images, rows):
    ax.imshow(img.reshape((28,28)), cmap = 'gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
