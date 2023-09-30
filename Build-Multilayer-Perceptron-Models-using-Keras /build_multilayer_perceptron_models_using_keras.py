# Importing modules

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras import Sequential
from keras.layers import Activation, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

np.random.seed(0)
print("Tensorflow version: ", tf.__version__)

# Load the reuters dataset

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 10000, test_split = 0.2)
print(len(x_train), 'training examples')
print(len(x_test), 'test examples')
num_classes = np.max(y_train) + 1
print(num_classes,'classes')

# Vectorize Sequence Data and One-hot Encode Class Labels

tokenizer = Tokenizer(num_words = 10000)
x_train = tokenizer.sequences_to_matrix(x_train, mode = 'binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode = 'binary')
x_train.shape, x_test.shape

y_train = tf.keras.utils.to_categorical(x_train, num_classes)
y_test = tf.keras.utils.to_categorical(x_test, num_classes)
y_train.shape, y_test.shape

# Build Multilayer Perceptron Model

model = Sequential([Dense(512, input_shape = (10000,)), Activation('relu'), Dropout(0.5), Dense(num_classes), Activation('softmax')])
model.summary()

# Train Model

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1, mode = 'min')
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_split = 0.1, callbacks = [es])

# Evaluate Model on Test Data

model.evaluate(x_test, y_test, batch_size = 32, verbose = 1)
plt.plot(history.hostory['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabels('Epochs')
plt.ylabels('Loss')
plt.legend()
plt.show()

plt.plot(history.hostory['accuracy'], label = 'Training accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabels('Epochs')
plt.ylabels('Loss')
plt.legend()
plt.show()
