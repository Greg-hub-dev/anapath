import os
from PIL import Image
import numpy as np
import openslide
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import csv
import re
import random
import shutil
import tifffile as tiff


Min_SIZE_MB = 5 # Taille min par tuile (en Mo)
tile_size_l = 4096
tile_size_h = 2048  # Taille des tuiles
level = 0  # Niveau de zoom OpenSlide (0 = max résolution)
train_tumor_path = "/mnt/c/Users/grego/Documents/Projet_ML/Data/Dataset/train/tumor"
train_normal_path = "/mnt/c/Users/grego/Documents/Projet_ML/Data/Dataset/train/normal"
treated_tumor_path='/mnt/c/Users/grego/Documents/Projet_ML/Data/TREATED/tumor'
treated_normal_path='/mnt/c/Users/grego/Documents/Projet_ML/Data/TREATED/normal'
totreat_tumor_path = '/mnt/c/Users/grego/Documents/Projet_ML/Data/TOTREAT/tumor'
totreat_normal_path = '/mnt/c/Users/grego/Documents/Projet_ML/Data/TOTREAT/normal'

os.environ["OMP_NUM_THREADS"] = "16"  # Ajustez selon votre nombre de cœurs
os.environ["MKL_NUM_THREADS"] = "16"
input_path = "/mnt/c/Users/grego/Documents/Projet_ML/Data/TOTREAT/tumor" # / a la fin
val_path='/mnt/c/Users/grego/Documents/Projet_ML/Data/Dataset/val'
train_path='/mnt/c/Users/grego/Documents/Projet_ML/Data/Dataset/train'


def load_slide(file, source):
    ''' Fonction de chargement du fichier mrxs
        Création du thumbnail utilisé pour définir les contours de l'échantillon

    '''
    filename = f"{file}.mrxs"
    path_to_slide = os.path.join(source,filename)
    print(path_to_slide)
    #with open(path_to_slide, "r") as f:
    #   contenu = f.read()
    slide = openslide.OpenSlide(path_to_slide)
    # Récupérer une miniature pour analyse (taille réduite pour traitement rapide)
    thumbnail = slide.get_thumbnail((1024, 2048))  # Ajuster la taille selon l’image
    # Convertir en format OpenCV
    thumbnail_np = np.array(thumbnail.convert("RGB"))
    return slide, thumbnail_np


def contour_cells(thumbnail_np):
    '''
    ## FONCTION DE DEFINITION DES CONTOURS
    Création du mask pour isoler les parties interessantes
    définition des contours
    retourne l'image détourée
    '''
    # Convertir l’image en HSV
    hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)

    # Définir les plages de couleur pour détecter le rose
    lower_pink = np.array([110,0 , 0])   # Valeurs min HSV
    upper_pink = np.array([180, 255, 255])  # Valeurs max HSV

    # Masque des zones contenant du tissu
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Trouver les contours des zones roses
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l’image originale
    contour_image = thumbnail_np.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Afficher l’image avec les contours détectés
    image_to_show = Image.fromarray(contour_image)

    return image_to_show, mask, contours

def slide_cut(slide, thumbnail_np, contours, filename,
              output_path, level=0, tile_size_l=tile_size_l, tile_size_h=tile_size_h,
              min_size_mb=Min_SIZE_MB, max_workers=8, global_csv_path="all_tiles_coordinates.csv"):
    """
    Découpe une lame (slide) en tuiles selon les contours détectés et sauvegarde leurs coordonnées.

    Args:
        slide: Objet OpenSlide contenant l'image de la lame
        thumbnail_np: Miniature de la lame sous forme de tableau numpy
        contours: Liste des contours détectés dans la miniature
        filename: Nom du fichier source pour nommer les tuiles
        output_path: Chemin de sortie pour sauvegarder les tuiles
        level: Niveau de résolution à utiliser (défaut: 0, résolution maximale)
        tile_size_l: Largeur des tuiles en pixels (défaut: 256)
        tile_size_h: Hauteur des tuiles en pixels (défaut: 256)
        min_size_mb: Taille minimale des tuiles en Mo (défaut: 0.01)
        max_workers: Nombre de threads pour le traitement parallèle (défaut: 4)

    Returns:
        tuple: (nombre_tuiles, temps_execution)
    """
    # Assurer que le dossier de sortie existe
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # S'assurer que le nom de fichier est complet (avec le chemin d'accès)
    # Stocker le nom de fichier complet (avec l'extension) pour la recherche ultérieure
    full_filename = os.path.basename(filename)

    # Vérifier si le fichier CSV global existe, sinon le créer avec des en-têtes
    global_csv_path = os.path.join(output_path, global_csv_path)
    csv_file_exists = os.path.isfile(global_csv_path)

    # Préparer le fichier CSV avec les en-têtes
    if not csv_file_exists:
        with open(global_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['slide_filename', 'tile_filename', 'contour_idx', 'x_original', 'y_original',
                             'width', 'height', 'x_thumbnail', 'y_thumbnail',
                             'w_thumbnail', 'h_thumbnail'])

    # Calculer le facteur d'échelle entre la miniature et l'image originale
    scale_x = slide.dimensions[0] / thumbnail_np.shape[1]
    scale_y = slide.dimensions[1] / thumbnail_np.shape[0]

    print(f"Début du traitement du fichier {full_filename}...")
    start_time = time.time()  # Début du chronomètre

    # Liste pour stocker les tâches de découpage
    tiles_to_process = []

    # Préparer la liste des tuiles à extraire
    for contour_idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)  # Récupérer les coordonnées dans la miniature

        # Convertir en coordonnées de l'image OpenSlide (grande taille)
        x_slide = int(x * scale_x)
        y_slide = int(y * scale_y)
        w_slide = int(w * scale_x)
        h_slide = int(h * scale_y)

        # Récupérer les coordonnées de toutes les tuiles dans cette zone
        for i in range(x_slide, x_slide + w_slide, tile_size_l):
            for j in range(y_slide, y_slide + h_slide, tile_size_h):
                # Ajouter cette tuile à la liste des tuiles à traiter
                tile_info = {
                    'position': (i, j),
                    'contour_idx': contour_idx,
                    'row': (i - x_slide) // tile_size_l,
                    'col': (j - y_slide) // tile_size_h,
                    'thumbnail_coords': (x, y, w, h)  # Coordonnées dans la miniature
                }
                tiles_to_process.append(tile_info)

    def process_tile(tile_info):
        """Fonction qui traite une tuile individuelle"""
        try:
            i, j = tile_info['position']
            contour_idx = tile_info['contour_idx']
            row, col = tile_info['row'], tile_info['col']
            x_thumb, y_thumb, w_thumb, h_thumb = tile_info['thumbnail_coords']

            # Extraire la tuile
            tile = slide.read_region((i, j), level, (tile_size_l, tile_size_h))

            # Vérifier la taille avant sauvegarde
            buffer = BytesIO()
            tile.save(buffer, format="PNG")
            file_size_mb = len(buffer.getvalue()) / (1024 * 1024)

            if file_size_mb > min_size_mb:
                # Créer un nom de fichier plus informatif
                tile_filename = f"tile_{full_filename}_c{contour_idx}_r{row}_c{col}.png"
                tile_path = os.path.join(output_path, tile_filename)
                tile.save(tile_path)

                # Ajouter les coordonnées à la liste qui sera écrite dans le CSV
                return [
                    full_filename,            # Nom du fichier original
                    tile_filename,        # Nom du fichier de la tuile
                    contour_idx,          # Indice du contour
                    i,                    # Position X dans l'image originale
                    j,                    # Position Y dans l'image originale
                    tile_size_l,          # Largeur de la tuile
                    tile_size_h,          # Hauteur de la tuile
                    x_thumb,              # Position X dans la miniature
                    y_thumb,              # Position Y dans la miniature
                    w_thumb,              # Largeur du contour dans la miniature
                    h_thumb               # Hauteur du contour dans la miniature
                ]

            return None  # Indique que la tuile n'a pas été sauvegardée car trop petite

        except Exception as e:
            print(f"Erreur lors du traitement d'une tuile: {e}")
            return False

    # Traiter les tuiles en parallèle
    saved_tiles_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_tile, tiles_to_process))
        # Filtrer les résultats None (tuiles non sauvegardées)
        saved_tiles_data = [result for result in results if result is not None]

    # Écrire toutes les coordonnées dans le fichier CSV global
    if saved_tiles_data:
        with open(global_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(saved_tiles_data)

    # Calculer le temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Fin du traitement du fichier {full_filename} en {execution_time:.2f} secondes")
    print(f"{len(saved_tiles_data)} tuiles générées à partir du fichier {full_filename}")
    print(f"Coordonnées ajoutées au fichier global: {global_csv_path}")

    return len(saved_tiles_data), execution_time, global_csv_path

def select_random_files_by_index(directory, num_files=15):
    """
    Sélectionne aléatoirement un certain nombre de fichiers ayant le même indice.

    Args:
        directory (str): Chemin vers le répertoire contenant les fichiers
        num_files (int): Nombre de fichiers à sélectionner (défaut: 5)

    Returns:
        list: Liste des fichiers sélectionnés
    """
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Extraire les indices des fichiers
    indices = {}
    for filename in all_files:
        # Utiliser une expression régulière pour extraire l'indice
        match = re.search(r'tile_([a-zA-Z0-9]+)_', filename)
        if match:
            index = match.group(1)
            if index not in indices:
                indices[index] = []
            indices[index].append(filename)
    print(indices)
    # Dictionnaire pour stocker les résultats
    selected_files_by_index = {}

    # Pour chaque indice, sélectionner des fichiers aléatoires
    for index, files in indices.items():
        # Si suffisamment de fichiers sont disponibles pour cet indice
        if len(files) >= num_files:
            # Si suffisamment de fichiers sont disponibles, en sélectionner 5 aléatoirement
            selected_files_by_index[index] = random.sample(files, num_files)
        else:
            # Si moins de 5 fichiers sont disponibles, prendre tous les fichiers
            selected_files_by_index[index] = files

    return selected_files_by_index

def move_unselected_files(directory, selected_files_dict):
    """
    Déplace les fichiers non sélectionnés vers un répertoire 'noselect'.

    Args:
        directory (str): Chemin vers le répertoire contenant les fichiers
        selected_files_dict (dict): Dictionnaire des fichiers sélectionnés par indice

    Returns:
        int: Nombre de fichiers déplacés
    """
    # Créer une liste de tous les fichiers sélectionnés
    all_selected_files = []
    for files in selected_files_dict.values():
        all_selected_files.extend(files)

    # Liste tous les fichiers dans le répertoire
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Créer le répertoire 'noselect' s'il n'existe pas
    noselect_dir = os.path.join(directory, "noselect")
    if not os.path.exists(noselect_dir):
        os.makedirs(noselect_dir)

    # Déplacer les fichiers non sélectionnés vers le répertoire 'noselect'
    files_moved = 0
    for filename in all_files:
        # Vérifier si le fichier contient un indice (pour éviter de déplacer des fichiers non pertinents)
        if re.search(r'tile_(\d+)_', filename):
            if filename not in all_selected_files:
                source_path = os.path.join(directory, filename)
                target_path = os.path.join(noselect_dir, filename)
                shutil.move(source_path, target_path)
                files_moved += 1

    return files_moved




def load_tif_image(file_path):
    """Charge une image TIF (mono ou multi-canal)."""
    try:
        image = tiff.imread(file_path)
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