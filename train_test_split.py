import h5py
from sklearn.model_selection import train_test_split
import numpy as np

with h5py.File('saved_path/concrete_crack_image_data.h5', 'r') as f:
    images = f['X_concrete'][:]
    labels = f['y_concrete'][:]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

with h5py.File('save_path/train_data.h5', 'w') as f:
    f.create_dataset('X_concrete', data=X_train)
    f.create_dataset('y', data=y_train)
    f.close()

with h5py.File('save_path/test_data.h5', 'w') as f:
    f.create_dataset('X_concrete', data=X_test)
    f.create_dataset('y_concrete', data=y_test)
    f.close()

print("Data successfully saved!!")