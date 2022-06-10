# Import Libraries
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl

tf.__version__

# Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Normalize
x_train = x_train / 255
x_test = x_test / 255

# Expected by the NN
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)

# Creating the CNN 
cnn = km.Sequential()


# Convolution Layer 1
cnn.add(kl.Conv2D(
    filters = 32,
    kernel_size = 3,
    activation = 'relu',
    input_shape=[28,28,1]
))
# Pooling
cnn.add(kl.MaxPool2D(
    pool_size=(2,2),
    strides = 2
))

# Dropout layer to help with over fitting
cnn.add(kl.Dropout(.5))

# Convolution Layer 2
cnn.add(kl.Conv2D(
    filters = 64,
    kernel_size = 3,
    activation = 'relu',
    input_shape=[28,28,1]
))
cnn.add(kl.MaxPool2D(
    pool_size=(2,2),
    strides = 2
))
 
cnn.add(kl.Dropout(.5))

# Flatten Layer
cnn.add(kl.Flatten())

# Hidden Layers
cnn.add(kl.Dense(
    units = 128,
    activation = 'relu'
))

cnn.add(kl.Dense(
    units = 256,
    activation = 'relu'
))

# Output layers
cnn.add(kl.Dense(
    units=10,
    activation = 'softmax'
))

# Compile
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Train
cnn.fit(x_train, y_train, epochs = 5)

# Evaluate against test set
cnn.evaluate(x_test,y_test)

# Save
cnn.save('handwritten.model')
