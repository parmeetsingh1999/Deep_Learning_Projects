#Installing required packages and libraries
pip install -U tensorflow-addons

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_addons as tfa
​
from tensorflow import keras
from tensorflow.keras import layers

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Prepare the data
add Codeadd Markdown
num_classes = 100
input_shape = (32, 32, 3)
​
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
​
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

#Configure the hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72 
patch_size = 6
num_patches = (image_size // patch_size) ** 2 
projection_dim = 64
num_heads = 4
#Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
#Size of the dense layers of the final classifier
mlp_head_units = [2048, 1024]

#Data Augmentation
add Codeadd Markdown
data_aug = keras.Sequential(
[
    layers.Normalization(),
    layers.Resizing(image_size, image_size),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor = 0.02),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2)
], 
    name = 'data_aug',
)
data_aug.layers[0].adapt(x_train)

#Implement multilayer Perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation = tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

plt.figure(figsize = (4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")
​
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size = (image_size, image_size)
)
​
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
​
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize = (4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i+1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype('uint8'))
    plt.axis('off')

#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units = projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim = num_patches,
            output_dim = projection_dim
        )
    def call(self, patch):
        positions = tf.range(start = 0, limit = self.num_patches, delta = 1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

#Build the ViT Model
def create_vit_classifier():
    inputs = layers.Input(shape = input_shape)
    augmented = data_aug(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        #Layer normalization 1
        x1 = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
        #Multihead attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = projection_dim,
            dropout = 0.1
        )(x1, x1)
        #Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        #Layer Normalization 2
        x3 = layers.LayerNormalization(epsilon = 1e-6)(x2)
        #MLP
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate = 0.1)
        #Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    #Creating a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon = 1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    #Add MLP
    features = mlp(representation, hidden_units = mlp_head_units, dropout_rate = 0.5)
    
    #Classify outputs
    logits = layers.Dense(num_classes)(features)
    
    model = keras.Model(inputs = inputs, outputs = logits)
    return model

#Compile, Train, and Evaluate the Model
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate = learning_rate,
        weight_decay = weight_decay
    )
    model.compile(
        optimizer = optimizer,
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name = 'accuracy'),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name = 'top-5-accuracy'),
        ],
    )
    checkpoint_filepath = '/tmp/checkpoint'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor = 'val_accuracy',
        save_best_only = True,
        save_weights_only = True,
    )
    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = batch_size,
        epochs = num_epochs,
        validation_split = 0.1,
        callbacks = [checkpoint_callback],
    )
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    return history
​
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
