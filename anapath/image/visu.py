import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import openslide
import os
import glob

def visualize_tiles_from_global_csv(global_csv_path, slides_dir, output_dir):
    """
    Visualise les tuiles de plusieurs lames à partir d'un fichier CSV global.

    Args:
        global_csv_path (str): Chemin vers le fichier CSV global
        slides_dir (str): Répertoire contenant les lames originales
        output_dir (str, optional): Répertoire pour sauvegarder les images de visualisation
        scale_factor (float, optional): Facteur d'échelle pour réduire l'image originale

    Returns:
        dict: Dictionnaire des images générées par nom de fichier
    """
    # Créer le répertoire de sortie si nécessaire
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Charger le fichier CSV global
    df = pd.read_csv(global_csv_path, dtype={'slide_filename': str, 'tile_filename': str})

    # Obtenir la liste des fichiers de lames uniques
    unique_slides = df['slide_filename'].unique()

    # Dictionnaire pour stocker les images générées
    generated_images = {}

    # Traiter chaque lame
    for slide_filename in unique_slides:
        print(f"Visualisation des tuiles pour {slide_filename}...")

        # Filtrer les données pour cette lame
        slide_df = df[df['slide_filename'] == slide_filename]

        # Rechercher le fichier de lame dans le répertoire
        # D'abord, essayer avec le nom exact
        slide_filename2 = f"{slide_filename}.mrxs"
        slide_path = os.path.join(slides_dir, slide_filename2)
        print(slide_path)

        # Si le fichier n'existe pas avec ce nom exact, rechercher avec des motifs
        if not os.path.exists(slides_dir):
            print(f"Fichier {slide_path} non trouvé, recherche d'alternatives...")

            # Extraire le nom de base sans extension
            base_name = os.path.splitext(slide_filename)[0]

            # Rechercher tous les fichiers potentiels avec un nom similaire
            # 1. Essayer avec le nom exact mais différentes extensions
            potential_files = []
            for ext in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']:
                potential_files.extend(glob.glob(os.path.join(slides_dir, f"{base_name}{ext}")))

            # 2. Rechercher avec un 0 au début (pour corriger le problème mentionné)
            if not potential_files and not base_name.startswith('0'):
                for ext in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']:
                    potential_files.extend(glob.glob(os.path.join(slides_dir, f"0{base_name}{ext}")))

            # 3. Rechercher par motif partiel si toujours rien trouvé
            if not potential_files:
                potential_files = glob.glob(os.path.join(slides_dir, f"*{base_name}*"))

            if potential_files:
                slide_path = potential_files[0]
                print(f"Fichier trouvé: {slide_path}")
            else:
                print(f"Erreur: Aucun fichier correspondant à {slide_filename} trouvé dans {slides_dir}")
                continue

        try:
            # Charger la lame
            slide = openslide.OpenSlide(slide_path)
            print(slide.dimensions[0])
            # Créer une miniature pour la visualisation
            l=1024
            h=2048
            ratio=8
            thumbnail = slide.get_thumbnail((ratio*l,ratio*h))  # Ajuster la taille selon l’image
            # Convertir en format OpenCV
            #thumbnail_np = np.array(thumbnail.convert("RGB"))
            #thumbnail_size = (int(slide.dimensions[0] * scale_factor),
            #int(slide.dimensions[1] * scale_factor))
            #thumbnail = slide.get_thumbnail(thumbnail_size)
            visualization_img = np.array(thumbnail)

            # Calculer les facteurs d'échelle
            scale_x = thumbnail.width / slide.dimensions[0]
            scale_y = thumbnail.height / slide.dimensions[1]


            # Dessiner toutes les tuiles pour cette lame
            for _, row in slide_df.iterrows():
                # Coordonnées de la tuile dans l'image originale, converties à l'échelle de visualisation
                x, y = int(row['x_original'] * scale_x), int(row['y_original'] * scale_y)
                w, h = int(row['width'] * scale_x), int(row['height'] * scale_y)

                # Couleur basée sur l'indice du contour
                contour_idx = int(row['contour_idx'])
                #color = colors[contour_idx % len(colors)]

                # Dessiner un rectangle autour de la tuile

                cv2.rectangle(visualization_img, (x, y), (x + w, y + h), (0,0,0), thickness=2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(visualization_img, slide_filename2, (x, y), font, fontScale=0.5, color=(0,0,0),thickness=1 ) #, font_scale, text_color, thickness)
            # Sauvegarder l'image si demandé
            if output_dir:
                base_name = os.path.splitext(os.path.basename(slide_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_visualization.png")

                plt.figure(figsize=(12, 10))
                plt.imshow(visualization_img)
                plt.title(f"Tuiles extraites de {os.path.basename(slide_path)}")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Image sauvegardée: {output_path}")

            # Stocker l'image générée
            generated_images[slide_filename] = visualization_img

        except Exception as e:
            print(f"Erreur lors de la visualisation des tuiles pour {slide_filename}: {e}")

    return generated_images


def display_image(image: np.array):
    """ affiche une image au format np.array"""
    plt.imshow(image)
    plt.title("Affichage de l'image")
    plt.axis("off")
    plt.show()
    return None
