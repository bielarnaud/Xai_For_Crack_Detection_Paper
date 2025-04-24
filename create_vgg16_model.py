import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py

print("Setting the image size...")
img_size = 128
print("image size = 128")

print("Loading the data...")
hf = h5py.File('saved_path/concrete_crack_image_data.h5', 'r')
X = np.array(hf.get('X_concrete'))
y = np.array(hf.get("y_concrete"))
hf.close()
print("Data successfully loaded!")

print("Scaling the data...!")
X = X / 255
print("Data successfully scaled!")

print("Converting grayscale images to RGB...")
if X.shape[-1] != 3: 
    X = np.repeat(X, 3, axis=-1)
print("Shape of X after conversion:", X.shape)
print("Images successfully converted to RGB!")

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Compiling the model...")
model.compile(loss="binary_crossentropy", metrics=["accuracy"])
print("Model successfully compiled!!")

print("Fitting the model...")
model.fit(X, y, batch_size=8, epochs=10, validation_split=.3)
print("Model successfully fitted!!")

print("Evaluating on training data...")
train_loss, train_accuracy = model.evaluate(X, y, verbose=0)
print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")

print("Saving the model...")
model.save("save_path/Concrete_Crack_Classification_VGG16.h5")
print("Model successfully saved!!")