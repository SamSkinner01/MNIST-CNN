# Imports
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Load Model
model = tf.keras.models.load_model('handwritten.model')

# Starting # in directory
image_number = 1

# Load image and show prediction
while os.path.isfile(f"my_images/digits{image_number}.png"):
    img = cv2.imread(f"my_images/digits{image_number}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"This digit is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap = plt.cm.binary)
    plt.show()
    image_number += 1