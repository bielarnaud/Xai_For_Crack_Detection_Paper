import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

model_path = "saved_path/Concrete_Crack_Classification_VGG19.h5"
model = load_model(model_path)

test_data_path = 'saved_path/test_data.h5'

with h5py.File(test_data_path, 'r') as hf:
    X_test = np.array(hf['X_concrete'])
    y_test = np.array(hf['y_concrete'])

X_test = X_test / 255.0 

if X_test.shape[-1] != 3:
    X_test = np.repeat(X_test, 3, axis=-1)

y_test = y_test.ravel()  

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nAcurácia no teste: {test_accuracy:.4f}")
print(f"Loss no teste: {test_loss:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int) 

print(classification_report(y_test, y_pred_classes))

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Crack', 'Crack'],
            yticklabels=['No Crack', 'Crack'])
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Pred')
plt.show()

plt.figure(figsize=(15, 7))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].squeeze(), cmap='gray' if X_test.shape[-1]==1 else None)
    plt.title(f"True: {y_test[i]}\nPred: {y_pred_classes[i][0]}")
    plt.axis('off')
plt.tight_layout()
plt.show()