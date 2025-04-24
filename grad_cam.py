print("Importing libraries...")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import cv2
import h5py
import os
import pandas as pd

def carregar_e_redimensionar_imagem(img_path, tamanho=(128, 128)):
    
    imagem = Image.open(img_path)
    imagem = imagem.resize(tamanho)
    return imagem

def converter_para_array_e_preparar_dados(imagem):
    
    imagem_array = np.array(imagem)
    dados = imagem_array.reshape((-1, 3))  # Redimensionar para (n_pixels, 3)
    return dados

def aplicar_kmeans(dados, n_clusters=2):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(dados)
    return kmeans

def criar_imagem_binaria(rotulos, shape):
    
    imagem_binaria = np.where(rotulos == 0, 0, 255).reshape(shape)
    return imagem_binaria

def exibir_imagens(imagem_original, imagem_binaria):
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagem_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagem_binaria, cmap='gray')
    plt.axis('off')

    plt.show()

def salvar_imagens(imagem_original, imagem_binaria, caminho_salvar):
    
    print(caminho_salvar)
    diretorio = os.path.dirname(caminho_salvar)

    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(imagem_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imagem_binaria, cmap='gray')
    plt.axis('off')

    plt.savefig(caminho_salvar, bbox_inches='tight', pad_inches=0)
    plt.close() 

def seguimentar_imagem(img, save_path, tamanho=(128, 128), n_clusters=2):
    
    img = img[0]
    print(img.shape)

    dados = converter_para_array_e_preparar_dados(img)

    kmeans = aplicar_kmeans(dados, n_clusters)

    imagem_binaria = criar_imagem_binaria(kmeans.labels_, img.shape[:2][::-1])

    save_path = save_path.replace('gradcam_block1_conv1', 'seg')
    salvar_imagens(img, imagem_binaria, save_path)

    return imagem_binaria

def calcular_area_comum(imagem1, imagem2):
    array1 = np.array(imagem1)  # CAN
    array2 = np.array(imagem2)  # SEG

    area_vp = np.sum((array1 == 255) & (array2 == 255))

    area_vn = np.sum((array1 == 0) & (array2 == 0))

    area_fp = np.sum((array1 == 255) & (array2 == 0))

    area_fn = np.sum((array1 == 0) & (array2 == 255))

    return area_vp, area_vn, area_fp, area_fn

def calcular_metricas(area_vp, area_vn, area_fp, area_fn):

    area_total = area_vp + area_vn + area_fp + area_fn
    metric = area_vp / (area_vp + area_fp)
    return area_total, metric

def calcular_f1_score(area_vp, area_vn, area_fp, area_fn):
    precisao = area_vp / (area_vp + area_fp)

    recall = area_vp / (area_vp + area_fn)

    if (precisao + recall) == 0: 
        f1_score = 0
    else:
        f1_score = 2 * (precisao * recall) / (precisao + recall)

    return f1_score

def create_mask(image: np.ndarray, lower_bound: tuple, upper_bound: tuple) -> np.ndarray:

    return cv2.inRange(image, lower_bound, upper_bound)

def dilate_mask(mask: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)

def load_and_preprocess_image(img_path: str, size: tuple = (128, 128)) -> np.ndarray:

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def processar_imagens_e_calcular_metrica(img, save_path, gradcam_img, size=(128, 128)):
    
    seg_img = seguimentar_imagem(img, save_path)  
    seg_img = dilate_mask(seg_img)

    print('Creating GradCAM mask...')
    lower_grad = (0.0784, 0, 0)
    upper_grad = (1, 1, 1)

    mask_gradcam = create_mask(gradcam_img, lower_grad, upper_grad)

    salvar_imagens(gradcam_img, mask_gradcam, save_path)

    area_vp, area_vn, area_fp, area_fn = calcular_area_comum(seg_img, mask_gradcam)

    metric = calcular_f1_score(area_vp, area_vn, area_fp, area_fn)

    return metric

def load_model_and_extract_layers(model_path):
  model = load_model(model_path)

  camadas_convolucionais = [layer.name for layer in model.layers if 'Conv2D' in str(layer)]
  camadas_densas = [layer.name for layer in model.layers if 'Dense' in str(layer)]

  return model, camadas_convolucionais, camadas_densas

def load_preprocess_normalize_img_to_model(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array

def generate_cam(model, img_array, layer_name):
    print(f"Generating CAM for layer: {layer_name}...")
    conv_layer = model.get_layer(layer_name)
    cam_model = tf.keras.models.Model(inputs=model.inputs, outputs=(conv_layer.output, model.output))

    with tf.GradientTape() as tape:
        conv_output, predictions = cam_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
        print(f"Predicted class index: {class_idx}")

    grads = tape.gradient(loss, conv_output)[0]
    output = conv_output[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, index]

    cam = cv2.resize(cam.numpy(), (img_array.shape[2], img_array.shape[1]))
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())  
    return cam

def generate_grad_cam(model, img_array, layer_name):
    print(f"Generating Grad-CAM for layer: {layer_name}...")
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=(model.get_layer(layer_name).output, model.output))

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
        print(f"Predicted class index: {class_idx}")

    grads = tape.gradient(loss, conv_output)[0]
    output = conv_output[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    grad_cam = np.zeros(output.shape[0:2], dtype=np.float32)

    for index, w in enumerate(weights):
        grad_cam += w * output[:, :, index]

    grad_cam = cv2.resize(grad_cam.numpy(), (img_array.shape[2], img_array.shape[1]))
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  
    return grad_cam, class_idx.numpy() 

def f1_can(model_path, img, save_path, size=(128, 128)):
    model, camadas_convolucionais, camadas_densas = load_model_and_extract_layers(model_path)
    img_array = img

    metrics_gradcam = []
    predicted_class = None

    for layer_name in camadas_convolucionais:
        grad_cam, pred_class = generate_grad_cam(model, img_array, layer_name)
        predicted_class = pred_class 

        grad_cam_color = plt.cm.jet(grad_cam)
        grad_cam_color = grad_cam_color[..., :3]

        save_path_img = os.path.join(save_path, f'gradcam_{layer_name}.png')
        metrics_gradcam.append(processar_imagens_e_calcular_metrica(img, save_path_img, grad_cam_color))

    print('\n'*2)
    best_gradcam_layer = camadas_convolucionais[metrics_gradcam.index(max(metrics_gradcam))]
    print(f'Best Metric: {max(metrics_gradcam)}')
    print(f"Best Grad-CAM layer: {best_gradcam_layer}")

    return best_gradcam_layer, max(metrics_gradcam), predicted_class

def get_true_and_predicted_classes(img_array, model, true_class=None):

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()

    prediction_probabilities = predictions[0].tolist()

    return predicted_class, prediction_probabilities


if __name__ == "__main__":
    model_path = 'saved_path/Concrete_Crack_Classification_VGG19_3.h5'
    imgs_path = 'saved_path/test_data.h5'
    save_path = 'save_path'
    size = (128, 128)

    with h5py.File(imgs_path, 'r') as f:
        if 'X_concrete' in f:
            images = f['X_concrete'][:]  
            print("Formato das imagens:", images.shape)
        else:
            print("Dataset 'images' não encontrado.")

        if 'y_concrete' in f:
            labels = f['y_concrete'][:]  
            print("Formato dos rótulos:", labels.shape)  
        else:
            print("Dataset 'labels' não encontrado.")

    best_layers = []
    best_metrics = []
    predicted_classes = []
    true_classes = []

    model = load_model(model_path)

    for i, img in enumerate(images):
        
        print(f'Image {i}')
        save_path_img = os.path.join(save_path, f'img_{i}')

        img_normalized = img/255
        img_normalized = np.expand_dims(img_normalized, axis=0)
        img_normalized_rgb = np.repeat(img_normalized, 3, axis=-1)

        best_layer, best_metric, pred_class = f1_can(model_path, img_normalized_rgb, save_path_img)

        predicted_class, prediction_probabilities = get_true_and_predicted_classes(img_normalized_rgb, model, labels[i])
        print(f"Predicted class: {predicted_class}")
        print(f"Prediction probabilities: {prediction_probabilities}")

        best_layers.append(best_layer)
        best_metrics.append(best_metric)
        predicted_classes.append(pred_class)
        true_classes.append(labels[i])

        if (i % 10) == 0:
            print(f'Image {i} processed')
            print(best_layers)
            print(best_metrics)
            print(predicted_classes)

            data = {
                'best_layer': best_layers,
                'best_metric': best_metrics,
                'predicted_class': predicted_classes,
                'true_class': true_classes
            }

            df_bm = pd.DataFrame(data)

            print(df_bm.describe())
            df_bm.to_csv('best_metrics.csv')
