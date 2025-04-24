import numpy as np
import shap
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import load_model
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd

model_path = 'saved_path/Concrete_Crack_Classification_VGG19_2.h5'
h5_path = 'saved_path/test_data.h5'
output_base_dir = 'save_path/shap_results'

def main():
    model = load_model(model_path)

    with h5py.File(h5_path, 'r') as f:
        X = f['X_concrete'][:]
        y = f['y_concrete'][:]  
        num_images = len(X)

    metricas = []

    for i in range(num_images):
      output_dir = os.path.join(output_base_dir, f"imagem_{i}")
      os.makedirs(output_dir, exist_ok=True)

      original_image = X[i]
      processed_image = preprocess_image(original_image)

      masker = shap.maskers.Image("inpaint_telea", processed_image.shape)

      explainer = shap.Explainer(model, masker, output_names=["Prediction"])

      shap_values = explainer(np.expand_dims(processed_image, axis=0), max_evals=200, batch_size=50)

      display_shap_values(shap_values, processed_image, output_dir)
      shap_values_norm = print_shap_values(shap_values, output_dir)

      lower_shap = (185, 185, 185)
      upper_shap = (255, 255, 255)

      mask_shap = create_mask(shap_values_norm * 255, lower_shap, upper_shap)
      mask_shap_dilated = dilate_mask(mask_shap)

      salvar_imagens(mask_shap_dilated, os.path.join(output_dir, "mask_shap_dilated.png"))

      seg_img = seguimentar_imagem(processed_image, output_dir)

      area_vp, area_vn, area_fp, area_fn = calcular_area_comum(mask_shap_dilated, seg_img)
      metric = calcular_f1_score(area_vp, area_vn, area_fp, area_fn)

      metricas.append({
          "indice_imagem": i,
          "label_original": int(y[i]),
          "f1_score": metric })

      print(f"imagem {i} - Processada")

    df_resultados = pd.DataFrame(metricas)
    df_resultados.to_csv(os.path.join(output_base_dir, "metricas_final.csv"), index=False)
    return df_resultados

def calcular_area_comum(imagem1, imagem2):
    array1 = np.array(imagem1)  
    array2 = np.array(imagem2)  
    area_vp = np.sum((array1 == 255) & (array2 == 255))

    area_vn = np.sum((array1 == 0) & (array2 == 0))

    area_fp = np.sum((array1 == 255) & (array2 == 0))

    area_fn = np.sum((array1 == 0) & (array2 == 255))

    return area_vp, area_vn, area_fp, area_fn

def calcular_f1_score(area_vp, area_vn, area_fp, area_fn):
    precisao = area_vp / (area_vp + area_fp)

    recall = area_vp / (area_vp + area_fn)

    if (precisao + recall) == 0:  
        f1_score = 0
    else:
        f1_score = 2 * (precisao * recall) / (precisao + recall)

    return f1_score

def aplicar_kmeans(dados, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(dados)
    return kmeans

def criar_imagem_binaria(rotulos, shape):
    if rotulos.size != np.prod(shape):
        raise ValueError(f"Shape {shape} requer {np.prod(shape)} elementos, mas 'rotulos' tem {rotulos.size}")
    imagem_binaria = np.where(rotulos == 0, 0, 255).reshape(shape)
    return imagem_binaria

def salvar_imagens(imagem_binaria, caminho_salvar):
    diretorio = os.path.dirname(caminho_salvar)

    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

    plt.figure(figsize=(5, 5))
    plt.imshow(imagem_binaria, cmap='gray')
    plt.axis('off')
    plt.savefig(caminho_salvar, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def seguimentar_imagem(img, save_path, tamanho=(128, 128), n_clusters=2):
    dados = converter_para_array_e_preparar_dados(img)

    kmeans = aplicar_kmeans(dados, n_clusters)

    altura, largura = img.shape[0], img.shape[1]
    imagem_binaria = criar_imagem_binaria(kmeans.labels_, (altura, largura))

    caminho_salvar = os.path.join(save_path, "segmentacao_resultado.png")
    salvar_imagens(imagem_binaria, caminho_salvar)

    return imagem_binaria

def create_mask(image: np.ndarray, lower_bound: tuple, upper_bound: tuple) -> np.ndarray:
    return cv2.inRange(image, lower_bound, upper_bound)

def dilate_mask(mask: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)

def converter_para_array_e_preparar_dados(imagem):
    imagem_array = np.array(imagem)
    dados = imagem_array.reshape((-1, 3)) 
    return dados

def preprocess_image(img_array):
    if img_array.size == 128 * 128:
        img_array = img_array.reshape((128, 128))
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    processed = img_array.astype(np.float32)
    return processed

def save_shap_results(shap_values, images):
    os.makedirs(output_dir, exist_ok=True)

    try:
        plt.figure(figsize=(10, 5))
        shap.image_plot(
            shap_values,
            show=False
        )
        plt.savefig(
            os.path.join(output_dir, 'shap_explanation.png'),
            bbox_inches='tight',
            dpi=150
        )
        plt.close()

        print("Resultados salvos em:", os.listdir(output_dir))

    except Exception as e:
        print(f"Erro ao salvar: {str(e)}")

def display_shap_values(shap_values, images, output_dir):
    images = (images - images.min()) / (images.max() - images.min())

    fig, ax = plt.subplots()

    shap.image_plot(shap_values[0], images, width = 20, aspect = 0.2, hspace = 0.2, show=False)

    output_path = os.path.join(output_dir, 'shap_values.png')
    plt.savefig(output_path)
    plt.close()

def print_shap_values(shap_values, output_dir):
    shap_image = shap_values[0].values[..., 0]

    shap_image = (shap_image - shap_image.min()) / (shap_image.max() - shap_image.min())

    output_path = os.path.join(output_dir, 'shap_values_1.png')

    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    return shap_image

if __name__ == "__main__":
    try:
        df_resultados = main()

    except Exception as e:
        print(f"Erro crítico: {str(e)}")
        raise