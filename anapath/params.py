import os

min_size_mb = 13.5 # Taille min par tuile (en Mo)
tile_size_l = 4096
tile_size_h = 2048  # Taille des tuiles
level = 0  # Niveau de zoom OpenSlide (0 = max résolution)
path_to_data = os.environ.get("CHUNK_SIZE")
train_tumor_path = f"{path_to_data}/Dataset/train/tumor"
train_normal_path = f"{path_to_data}/Dataset/train/normal"
treated_tumor_path= f"{path_to_data}/TREATED/tumor"
treated_normal_path= f"{path_to_data}/TREATED/normal"
totreat_tumor_path = f"{path_to_data}/TOTREAT/tumor"
totreat_normal_path = f"{path_to_data}/TOTREAT/normal"
input_path = "{path_to_data}/TOTREAT/tumor" # / a la fin
val_path= f"{path_to_data}/Dataset/val"
train_path= f"{path_to_data}/Dataset/train"

os.environ["OMP_NUM_THREADS"] = "16"  # Ajustez selon votre nombre de cœurs
os.environ["MKL_NUM_THREADS"] = "16"
