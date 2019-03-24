import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

categories = ["dogs", "cats"]

def prepare(filepath):
	img_size = 50
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array, (img_size, img_size))
	return new_array.reshape(-1, img_size, img_size, 1)

model = tf.keras.models.load_model('dogcat.model')

prediction = model.predict([prepare(sys.argv[1])])
print(categories[int(prediction[0][0])])