import lime
from lime import lime_image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic
import cv2
import os
import h5py
import pandas as pd
from sklearn.cluster import KMeans

# Configurações
model_path = ''
h5_path = ''
output_base_dir = ''
metrics_file = os.path.join(output_base_dir, 'metricas_lime.csv')

# Parâmetros do LIME
n_segments = 500
num_features = 30
num_samples = 1000

# Função para garantir que a imagem tenha 3 canais (RGB)
def convert_to_rgb(image):
    if image.ndim == 3 and image.shape[-1] == 1:
        # Replicar a imagem para 3 canais (RGB)
        image = np.repeat(image, 3, axis=-1)
    return image

def main():
    # Carregar modelo
    model = load_model(model_path)
    
    # Função wrapper para compatibilidade com LIME
    def predict_fn(images):
        return model.predict(images / 255.0)
    
    # Carregar dados de teste
    with h5py.File(h5_path, 'r') as f:
        X = f['X_concrete'][:]
        y = f['y_concrete'][:]
        num_images = len(X)

    # Verificar se o arquivo de métricas existe e carregar métricas anteriores
    if os.path.exists(metrics_file):
        df_metricas = pd.read_csv(metrics_file)
        print("Métricas anteriores carregadas.")
    else:
        df_metricas = pd.DataFrame(columns=["indice_imagem", "label_original", "f1_score", "VP", "VN", "FP", "FN"])
        print("Nenhuma métrica anterior encontrada. Iniciando um novo arquivo.")
    
    # Criar diretório base para resultados
    os.makedirs(output_base_dir, exist_ok=True)

    # Lista para armazenar métricas durante o processamento
    metricas_parciais = []

    for i in range(len(X)): 
        # Criar diretório para a imagem
        output_dir = os.path.join(output_base_dir, f"imagem_{i}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Obter e pré-processar imagem
            img_orig = X[i].astype('double')
            
            # Garantir que a imagem tenha 3 canais (RGB)
            img_orig_rgb = convert_to_rgb(img_orig)
            
            # Criar explicador LIME
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_orig_rgb,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=num_samples,
                segmentation_fn=lambda x: slic(x, n_segments=n_segments, compactness=10, sigma=1)
            )
            
            # Obter máscara das regiões importantes
            _, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=num_features,
                hide_rest=False
            )
            
            # Processar máscara
            mask_binary = (mask > 0).astype(np.uint8) * 255
            mask_dilated = dilate_mask(mask_binary)
            
            # Salvar máscara
            plt.imsave(os.path.join(output_dir, 'lime_mask.png'), mask_binary, cmap='gray')
            plt.imsave(os.path.join(output_dir, 'lime_mask_dilated.png'), mask_dilated, cmap='gray')
            
            # Segmentar imagem original com K-means
            seg_img = segmentar_imagem(img_orig_rgb.astype(np.uint8), output_dir)
            
            # Calcular métricas
            area_vp, area_vn, area_fp, area_fn = calcular_area_comum(mask_dilated, seg_img)
            f1_score = calcular_f1_score(area_vp, area_vn, area_fp, area_fn)
            
            # Armazenar resultados parciais
            metricas_parciais.append({
                "indice_imagem": i,
                "label_original": int(y[i]),
                "f1_score": f1_score,
                "VP": area_vp,
                "VN": area_vn,
                "FP": area_fp,
                "FN": area_fn
            })
            
            print(f"Imagem {i} processada - F1: {f1_score:.4f}")
            
            # Salvar métricas a cada 10 imagens processadas
            if (i + 1) % 10 == 0:
                df_metricas_parciais = pd.DataFrame(metricas_parciais)
                df_metricas = pd.concat([df_metricas, df_metricas_parciais], ignore_index=True)
                df_metricas.to_csv(metrics_file, index=False)
                metricas_parciais = []  # Limpar métricas parciais para a próxima rodada

        except Exception as e:
            print(f"Erro na imagem {i}: {str(e)}")
            metricas_parciais.append({
                "indice_imagem": i,
                "label_original": int(y[i]),
                "f1_score": -1,
                "VP": -1,
                "VN": -1,
                "FP": -1,
                "FN": -1
            })
    
    # Salvar as métricas finais
    if metricas_parciais:
        df_metricas_parciais = pd.DataFrame(metricas_parciais)
        df_metricas = pd.concat([df_metricas, df_metricas_parciais], ignore_index=True)
        df_metricas.to_csv(metrics_file, index=False)

    print("Processamento concluído! Métricas salvas em:", metrics_file)
    return df_metricas

# ===== FUNÇÕES AUXILIARES =====
def dilate_mask(mask, kernel_size=3, iterations=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)

def segmentar_imagem(img, save_path):
    """Segmenta imagem usando K-means e salva resultado"""
    # Converter para array 2D
    pixels = img.reshape((-1, 3)).astype(np.float32)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Criar imagem binária
    segmented_img = labels.reshape(img.shape[:2])
    segmented_img = (segmented_img == segmented_img.max()).astype(np.uint8) * 255
    
    # Salvar resultado
    plt.imsave(os.path.join(save_path, 'segmentacao.png'), segmented_img, cmap='gray')
    return segmented_img

def calcular_area_comum(mask1, mask2):
    """Calcula áreas de verdadeiro positivo, negativo, falso positivo e negativo"""
    mask1_bin = (mask1 > 127).astype(int)
    mask2_bin = (mask2 > 127).astype(int)
    
    vp = np.sum((mask1_bin == 1) & (mask2_bin == 1))
    vn = np.sum((mask1_bin == 0) & (mask2_bin == 0))
    fp = np.sum((mask1_bin == 1) & (mask2_bin == 0))
    fn = np.sum((mask1_bin == 0) & (mask2_bin == 1))
    
    return vp, vn, fp, fn

def calcular_f1_score(vp, vn, fp, fn):
    """Calcula F1-score a partir da matriz de confusão"""
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    
    if (precisao + recall) > 0:
        return 2 * (precisao * recall) / (precisao + recall)
    return 0

if __name__ == "__main__":
    df_metricas = main()
    print("Processamento concluído! Métricas salvas em:", output_base_dir)
