# Importing libraries

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
%matplotlib inline

# Checking the tensorflow version

print("Using TensorFlow version", tf.__version__)

# Loading MNIST Dataset

(x_train,y_train), (x_test,y_test) = mnist.load_data()

# Shape of imported arrays

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# Plot an image example

plt.imshow(x_train[0], cmap = 'binary')
plt.show()
y_train[0]
print(set(y_train))

# One Hot Encoding

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Validating Shapes

print('y_train_encoded shape:', y_train_encoded.shape)
print('y_test_encoded shape:', y_test_encoded.shape)
print(y_train_encoded[0])

# Unrolling n-dimensional arrays to vectors

x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))
print(set(x_train_reshaped[0]))

# Data Normalization

x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)
epsilon = 1e-10
x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

# Create a model

model = Sequential([
                   Dense(128, activation = 'relu', input_shape = (784,)),
                   Dense(128, activation = 'relu'),
                   Dense(10, activation = 'softmax')
])

# Compiling the model

model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model

# Training the model

model.fit(x_train_norm, y_train_encoded, epochs = 3)

# Evaluating the model

_, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print("Test set accuracy:",accuracy)

# Predictions on test set

preds = model.predict(x_test_norm)
print("Shape of preds:",preds.shape)

# Plotting the results

plt.figure(figsize = (12,12))
start_index = 0
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  pred = np.argmax(preds[start_index+i])
  ground_truth = y_test[start_index+i]
  col = 'g'
  if pred != ground_truth:
    col = 'r'
  plt.xlabel('i = {}, pred = {}, gt = {}'.format(start_index+i, pred, ground_truth), color = col)
  plt.imshow(x_test[start_index+i], cmap = 'binary')
plt.show()

# Incase of a wrong prediction

plt.plot(preds[8])
plt.show()
