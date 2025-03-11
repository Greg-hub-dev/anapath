import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path


def load_tif_image(file_path):
    """Charge une image TIF (mono ou multi-canal)."""
    try:
        image = misc.imread(file_path)
        return image
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path}: {e}")
        return None


def preprocess_image(image, target_size=(256, 256), normalize=True):
    """Redimensionne et normalise l'image."""
    if image is None:
        return None

    # Convertir en objet PIL si ce n'est pas déjà le cas
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Redimensionner
    image = image.resize(target_size, Image.ANTIALIAS)

    # Convertir en tableau NumPy
    image_array = np.array(image)

    # Normaliser (optionnel)
    if normalize:
        image_array = image_array.astype(np.float32) / 255.0

    return image_array


def save_image(image_array, output_path):
    """Sauvegarde l'image sous format TIF."""
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    image.save(output_path, format='TIFF')


def display_image(image_path):
    """Affiche une image TIF."""
    image = load_tif_image(image_path)
    if image is not None:
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()


def detect_contours(image_path):
    """Détecte les contours d'une image TIF."""
    image = load_tif_image(image_path)
    if image is not None:
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image

        # Détection des contours avec Canny
        edges = cv2.Canny(gray_image, 100, 200)

        # Afficher les contours
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plt.show()


def process_folder(input_folder, output_folder, target_size=(256, 256), normalize=True):
    """Automatise le prétraitement d'un dossier d'images TIF."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            print(f"Processing: {input_path}")
            image = load_tif_image(input_path)
            processed_image = preprocess_image(image, target_size, normalize)

            if processed_image is not None:
                save_image(processed_image, output_path)
                print(f"Saved processed image to: {output_path}")
