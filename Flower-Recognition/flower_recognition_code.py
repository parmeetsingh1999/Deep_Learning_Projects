# Import libraries

import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

# Steps of preprocessing and generating image from segregated dataset

data = list()
label = list()
IMG_SIZE = 128
for i in os.listdir("../input/flowers-recognition/flowers/daisy"):
    try:
        path = "../input/flowers-recognition/flowers/daisy/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        data.append(img)
        label.append(0)
    except:
        None
for i in os.listdir("../input/flowers-recognition/flowers/dandelion"):
    try:
        path = "../input/flowers-recognition/flowers/dandelion/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        data.append(img)
        label.append(1)
    except:
        None
for i in os.listdir("../input/flowers-recognition/flowers/rose"):
    try:
        path = "../input/flowers-recognition/flowers/rose/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        data.append(img)
        label.append(2)
    except:
        None
for i in os.listdir("../input/flowers-recognition/flowers/sunflower"):
    try:
        path = "../input/flowers-recognition/flowers/sunflower/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        data.append(img)
        label.append(3)
    except:
        None
for i in os.listdir("../input/flowers-recognition/flowers/tulip"):
    try:
        path = "../input/flowers-recognition/flowers/tulip/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        data.append(img)
        label.append(4)
    except:
        None
        
set(label)
data = np.array(data)
print(data.shape)
x_train,x_test,y_train,y_test = tts(data,label,test_size = 0.15,random_state = 0)
x_train_val,x_test_val,y_train_val,y_test_val = tts(x_train,y_train,test_size = 0.15,random_state = 0)

# A small glips of generated dataset with label

plt.figure(figsize=(20,20))
for i in range(10):
    img = x_train[2*i]
    plt.subplot(1,10,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(y_train[2*i])
    
# Applying Data Augumentation on x_train for more data generation

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  
    samplewise_center=False,  
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    rotation_range=60, 
    zoom_range = 0.1,  
    width_shift_range=0.1,  
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode = "reflect"
    ) 
   
datagen.fit(x_train)

plt.figure(figsize=(20,20))
for i in range(10):
    img = x_train[2*i]
    plt.subplot(1,10,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(y_train[2*i])
    
# Build the CNN

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (128, 128, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(5, activation = "softmax"))

model.compile(optimizer = 
tf.keras.optimizers.Adam(lr = 0.001), loss = tf.keras.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])
model.summary()

epoch = 50

history = model.fit(np.array(x_train),np.array(y_train),epochs= epoch,validation_data=(np.array(x_test_val),np.array(y_test_val)))

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(train_acc,label = "Training")
plt.plot(val_acc,label = 'Validation/Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss,label = 'Training')
plt.plot(val_loss,label = 'Validation/Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

cnn_pred = model.predict_classes(x_test)
print(classification_report(y_test,cnn_pred))

# Saving CNN

model.save('CNN.h5',)

# Applying transfer learning

process_unit = tf.keras.applications.resnet50.preprocess_input
base_model = tf.keras.applications.ResNet50(input_shape=(128,128,3),include_top=False,weights='imagenet')
base_model.trainable = False
inputs = tf.keras.Input(shape=(128,128,3))
x = base_model(inputs,training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024,activation='relu')(x)
outputs = tf.keras.layers.Dense(5,activation='softmax')(x)
n_model = tf.keras.Model(inputs,outputs)
n_model.summary()
n_model.compile(optimizer=tf.keras.optimizers.Nadam(),loss =tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'] )
hist = n_model.fit(np.array(x_train),np.array(y_train),epochs=10,validation_data=(np.array(x_test_val),np.array(y_test_val)))

train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
plt.plot(train_acc,label = "Training")
plt.plot(val_acc,label = 'Validation/Test')
plt.title('training graph')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss,label = 'Training')
plt.plot(val_loss,label = 'Validation/Test')
plt.title('loss graph')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

n_model.save('transfer_learning_model.h5')
