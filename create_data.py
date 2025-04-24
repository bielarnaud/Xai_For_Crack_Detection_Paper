import cv2
import numpy as np
import os
import random
import h5py

data_directory = "data_path"
img_size = 128
categories = ["Positive", "Negative"]
data = []

def create_data():
    for category in categories:
        path = os.path.join(data_directory, category)
        class_num = categories.index(category)

        i = 0
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            data.append([new_array, class_num])
            i += 1
            if i % 500 == 0:
                print(f"{i} images processed for category {category}")

print("Creating data...")
create_data()
print("The data successfully created!!")

print("Shuffling data...")
random.shuffle(data)
print("The data successfully shuffled!!")

X_data = []
y = []

i = 0
for features, label in data:
    X_data.append(features)
    y.append(label)
    i += 1
    if i % 500 == 0:
        print(f"{i} images processed")

print("X and y data successfully created!!")

print("Reshaping X data...")
X = np.array(X_data).reshape(len(X_data), img_size, img_size, 1)
print("X data successfully reshaped!!")

print("Saving the data...")
hf = h5py.File("save_path/concrete_crack_image_data.h5", "w")
hf.create_dataset("X_concrete", data = X, compression = "gzip")
hf.create_dataset("y_concrete", data = y, compression = "gzip")
hf.close()
print("Data successfully saved!!")