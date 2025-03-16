
from keras.utils import to_categorical
from anapath.params import *
from tqdm import tqdm
import numpy as np
import os
from PIL import Image

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
