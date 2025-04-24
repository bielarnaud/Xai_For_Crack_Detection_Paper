import lime
from lime import lime_image
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import time
import tracemalloc
import os

def main():
    tracemalloc.start()
    tpi = time.time()

    model_path ='saved_path/Concrete_Crack_Classification_VGG19_2.h5'
    img_path = 'saved_path/img_path.jpg'

    size = (128, 128)

    output_dir =''

    model = load_model(model_path)

    img_orig = load_and_preprocess_image(img_path, size)

    n_segments = 500
    num_features = 30

    segments_slic = slic(img_orig, n_segments=n_segments, compactness=10, sigma=1, start_label=1)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_orig.astype('double'),
        model.predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=slic
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=True
    )

    ind = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap_orig = np.vectorize(dict_heatmap.get)(explanation.segments)

    heatmap = apply_heatmap(mask, img_orig)

    masked_img = img_orig.copy()
    masked_img[np.where(mask == 0)] = 0

    tpf = time.time()
    tpt = tpf - tpi
    print("Memória utilizada: ", tracemalloc.get_traced_memory())
    print("Tempo de processamento: ", tpt)

    tracemalloc.stop()

    plot_results(img_orig, segments_slic, temp, mask, heatmap, heatmap_orig, output_dir)

def load_and_preprocess_image(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return img

def apply_heatmap(mask, img_orig):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    heatmap[np.where(mask == 0)] = 0
    heatmap = cv2.addWeighted(img_orig, 0.5, heatmap, 0.5, 0)
    return heatmap

def plot_results(img, segments_slic, temp, mask, heatmap, heatmap_orig, output_dir):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].set_title("Entrada")
    axs[0, 0].imshow(img)
    axs[0, 0].axis('off')

    axs[0, 1].set_title("SLIC")
    axs[0, 1].imshow(mark_boundaries(img, segments_slic))
    axs[0, 1].axis('off')

    axs[0, 2].set_title('LIME')
    axs[0, 2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    axs[0, 2].axis('off')

    axs[1, 0].set_title('Regiões Importantes')
    axs[1, 0].imshow(heatmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Heatmap')
    heat = axs[1, 1].imshow(heatmap_orig, cmap="RdBu", vmin=-heatmap_orig.max(), vmax=heatmap_orig.max())
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(heat, cax=cax)
    axs[1, 1].axis('off')

    axs[1, 2].axis('off')  

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(output_dir, 'lime_result.png'))

if __name__ == "__main__":
    main()
