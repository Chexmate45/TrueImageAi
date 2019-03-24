import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('first_num_reader.model')
model.fit(x_train, y_train, epochs=30)

model.save('first_num_reader.model') """
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = tf.keras.models.load_model('first_num_reader.model')

predictions = model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_train[0], cmap= plt.cm.binary)
plt.show()