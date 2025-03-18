import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras import layers, Sequential, models

from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

from anapath.params import *


### A ADAPTER COMPLETEMENT



def initialize_model_tx(input_shape: tuple) -> Model:

    """
    Initialize the Neural Network with random weights
    """
    model = tf.keras.Sequential([
    # INPUT LAYER
    layers.Input(shape=(input_shape)),

    # CONV/HIDDEN LAYERS

    layers.Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),

    layers.MaxPool2D(pool_size=(2,2)),
    #layers.Dropout(0.2),
    #layers.Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    #layers.Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    ##layers.MaxPool2D(pool_size=(2,2)),
    #layers.Dropout(0.2),
    #layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    #layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    ##layers.MaxPool2D(pool_size=(2,2)),
    #layers.Dropout(0.2),

    ### Flattening
    layers.Flatten(),

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"

    layers.Dense(16, activation='relu',  kernel_initializer='he_uniform'),

    layers.Dropout(0.2),

    # PREDICITVE LAYER
    layers.Dense(1, activation='sigmoid')
])

    print("✅ Model initialized")

    return model

def compile_model_tx(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    adam_opt = optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer='adam',
              loss='binary_crossentropy',

              metrics=['accuracy','recall'])



    print("✅ Model compiled")

    return model




def train_model(
        model: Model,
        X,
        y,
        validation_data,
        epochs=10,
        batch_size=2,
        patience=10,

    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_recall",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history_cnn = model.fit(

        X,y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[es]
        )

    print(f"✅ Model trained on {len(X)} images with min val Accuracy: {round(np.min(history_cnn.history['val_accuracy']), 2)}")


    return model, history_cnn
