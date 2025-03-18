
from keras.utils import to_categorical
from anapath.params import *
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from anapath.params import *

def load_image_anapath(type='tumor',env='train', max_image=2):
    """Charge les images dans des X et y"""
    if type == 'tumor':
        data_path = tumor_test_path
        classes = {'normal':0, 'tumor':1}
    if type == 'cell':
        data_path = cell_test_path
        classes = {'low':0, 'medium':1, 'high':2}
    imgs = []
    labels = []
    for (cl, i) in classes.items():
        images_path = [elt for elt in os.listdir(os.path.join(data_path,env, cl)) if elt.find('.png')>0]
        for img in tqdm(images_path[:max_image]):
            path = os.path.join(data_path, env, cl, img)
            if os.path.exists(path):
                image = Image.open(path)
                image = image.resize((512, 1024))
                imgs.append(np.array(image))
                labels.append(i)

    X = np.array(imgs)
    num_classes = len(set(labels))
    y = to_categorical(labels, num_classes)

    return X, y, num_classes

def preprocess_image(image : np.array, target_size=(256, 256), normalize=True):
    """Redimensionne et normalise l'image."""
    if image is None:
        return None

    # Convertir en objet PIL si ce n'est pas déjà le cas
    #if not isinstance(image, Image.Image):
    #    image = Image.fromarray(image)

    # Redimensionner
    image = cv2.resize(image, target_size)

    # Convertir BGR en RGB
    image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normaliser (optionnel)
    if normalize:
        image_array = image_array.astype(np.float32) / 255.0

    return image_array

def datagenerator(type):
# Configuration du générateur d'augmentation

    if type == 'diag':
        datagen = ImageDataGenerator(
            rescale = 1/255,           #Pas sur que cela soit nécesssaire car déjà scaler
            rotation_range=20,        # Rotation aléatoire jusqu'à 20 degrés
            width_shift_range=0.2,    # Décalage horizontal jusqu'à 20% de la largeur
            height_shift_range=0.2,   # Décalage vertical jusqu'à 20% de la hauteur
            #shear_range=0.2,          # Cisaillement (déformation) jusqu'à 20%
            zoom_range=0.2,           # Zoom aléatoire entre 80% et 120%
            horizontal_flip=True,     # Retournement horizontal aléatoire
            fill_mode='nearest')      # Méthode pour remplir les pixels créés après transformation
        train_generator = datagen.flow_from_directory(
            train_path,  # Dossier parent contenant un sous-dossier par classe
            target_size=(512, 1024),        # Redimensionnement des images
            batch_size=1,                 # Nombre d'images par lot
            class_mode='categorical')      # Type d'encodage des étiquettes (one-hot pour multi-classes)
        validation_generator = datagen.flow_from_directory(
            val_path,  # Dossier parent contenant un sous-dossier par classe
            target_size=(512, 1024),        # Redimensionnement des images
            batch_size=1,                 # Nombre d'images par lot
            class_mode='categorical')      # Type d'encodage des étiquettes (one-hot pour multi-classes)
        return datagen, train_generator, validation_generator
    else:
        datagen = ImageDataGenerator(
            rescale = 1/255,           #Pas sur que cela soit nécesssaire car déjà scaler
            rotation_range=20,        # Rotation aléatoire jusqu'à 20 degrés
            width_shift_range=0.2,    # Décalage horizontal jusqu'à 20% de la largeur
            height_shift_range=0.2,   # Décalage vertical jusqu'à 20% de la hauteur
            #shear_range=0.2,          # Cisaillement (déformation) jusqu'à 20%
            zoom_range=0.2,           # Zoom aléatoire entre 80% et 120%
            horizontal_flip=True,     # Retournement horizontal aléatoire
            fill_mode='nearest')      # Méthode pour remplir les pixels créés après transformation
        train_generator = datagen.flow_from_directory(
            train_path,  # Dossier parent contenant un sous-dossier par classe
            target_size=(512, 1024),        # Redimensionnement des images
            batch_size=1,                 # Nombre d'images par lot
            class_mode='categorical')      # Type d'encodage des étiquettes (one-hot pour multi-classes)
        validation_generator = datagen.flow_from_directory(
            val_path,  # Dossier parent contenant un sous-dossier par classe
            target_size=(512, 1024),        # Redimensionnement des images
            batch_size=1,                 # Nombre d'images par lot
            class_mode='categorical')      # Type d'encodage des étiquettes (one-hot pour multi-classes)
    return datagen, train_generator, validation_generator
