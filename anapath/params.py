import os

min_size_mb = 13.5 # Taille min par tuile (en Mo)
tile_size_l = 4096
tile_size_h = 2048  # Taille des tuiles
level = 0  # Niveau de zoom OpenSlide (0 = max résolution)
path_to_data = os.environ.get("PATH_TO_DATA")

train_tumor_path = f"{path_to_data}/Dataset/train/tumor"
train_normal_path = f"{path_to_data}/Dataset/train/normal"
val_tumor_path = f"{path_to_data}/Dataset/val/tumor"
val_normal_path = f"{path_to_data}/Dataset/val/normal"
test_tumor_path = f"{path_to_data}/Dataset/test/tumor"
test_normal_path = f"{path_to_data}/Dataset/test/normal"

treated_tumor_path= f"{path_to_data}/TREATED/tumor"
treated_normal_path= f"{path_to_data}/TREATED/normal"
totreat_tumor_path = f"{path_to_data}/TOTREAT/tumor"
totreat_normal_path = f"{path_to_data}/TOTREAT/normal"
val_path= f"{path_to_data}/Dataset/val"
train_path= f"{path_to_data}/Dataset/train"

os.environ["OMP_NUM_THREADS"] = "16"  # Ajustez selon votre nombre de cœurs
os.environ["MKL_NUM_THREADS"] = "16"
