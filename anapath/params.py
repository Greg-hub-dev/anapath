import os

LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

min_size_mb = 13.5 # Taille min par tuile (en Mo)
tile_size_l = 4096
tile_size_h = 2048  # Taille des tuiles
level = 0  # Niveau de zoom OpenSlide (0 = max résolution)
path_to_data = os.environ.get("PATH_TO_DATA")

tumor_test_path = f"{path_to_data}/Dataset"
cell_test_path = f"{path_to_data}/Dataset2"

train_tumor_path = f"{tumor_test_path}/train/tumor"
train_normal_path = f"{tumor_test_path}/train/normal"
val_tumor_path = f"{tumor_test_path}/val/tumor"
val_normal_path = f"{tumor_test_path}/val/normal"
test_tumor_path = f"{tumor_test_path}/test/tumor"
test_normal_path = f"{tumor_test_path}/test/normal"


treated_tumor_path= f"{path_to_data}/TREATED/tumor"
treated_normal_path= f"{path_to_data}/TREATED/normal"
totreat_tumor_path = f"{path_to_data}/TOTREAT/tumor"
totreat_normal_path = f"{path_to_data}/TOTREAT/normal"
val_path= f"{path_to_data}/Dataset/val"
train_path= f"{path_to_data}/Dataset/train"
test_path= f"{path_to_data}/Dataset/test"

os.environ["OMP_NUM_THREADS"] = "16"  # Ajustez selon votre nombre de cœurs
os.environ["MKL_NUM_THREADS"] = "16"


'''
Dataset 2 : tumor low and high
'''
val_cell_path= f"{path_to_data}/Dataset2/val"
train_cell_path= f"{path_to_data}/Dataset2/train"
test_cell_path= f"{path_to_data}/Dataset2/test"
