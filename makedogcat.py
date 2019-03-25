import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import sys


dataDir = str(sys.argv[1])
categories = ["dogs", "cats"]

img_size = 50

training_data = []

def create_training_data():
	for category in categories:
		path = os.path.join(dataDir, category)
		class_num = categories.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (img_size, img_size))
				training_data.append([new_array, class_num])
			except:
				pass
create_training_data()
print(len(training_data))
#plt.imshow(new_array, cmap="gray")
#plt.show()
random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
	X.append(features)
	Y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)