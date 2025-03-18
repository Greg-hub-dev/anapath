import os
import shutil
import random

# Chemin vers le dossier parent Dataset => A MODIFIER AVEC VOTRE PATH
file_path = "/Users/annelisethomin/Docs/0_PROJET_ANAPATH/Data_test/Dataset"

# Chemins vers les sous-dossiers
train_tumor_path = os.path.join(file_path, "train", "tumor")
train_normal_path = os.path.join(file_path, "train", "normal")
val_tumor_path = os.path.join(file_path, "val", "tumor")
val_normal_path = os.path.join(file_path, "val", "normal")
test_tumor_path = os.path.join(file_path, "test", "tumor")
test_normal_path = os.path.join(file_path, "test", "normal")

# Fonction pour déplacer les fichiers
def move_files(source_dir, dest_dir, files):
    for file in files:
        shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

# Sélectionner et déplacer 15% des images tumor
tumor_images = os.listdir(train_tumor_path)
num_tumor = len(tumor_images)
num_val_tumor = int(0.15 * num_tumor)
num_test_tumor = int(0.15 * num_tumor)

val_tumor_images = random.sample(tumor_images, num_val_tumor)
test_tumor_images = random.sample([img for img in tumor_images if img not in val_tumor_images], num_test_tumor)

move_files(train_tumor_path, val_tumor_path, val_tumor_images)
move_files(train_tumor_path, test_tumor_path, test_tumor_images)

# Sélectionner et déplacer 15% des images normal
normal_images = os.listdir(train_normal_path)
num_normal = len(normal_images)
num_val_normal = int(0.15 * num_normal)
num_test_normal = int(0.15 * num_normal)

val_normal_images = random.sample(normal_images, num_val_normal)
test_normal_images = random.sample([img for img in normal_images if img not in val_normal_images], num_test_normal)

move_files(train_normal_path, val_normal_path, val_normal_images)
move_files(train_normal_path, test_normal_path, test_normal_images)

print("Les fichiers ont été déplacés avec succès.")
