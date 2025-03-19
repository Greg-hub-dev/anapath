import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Sequential, models
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

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
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
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
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),

    # PREDICITVE LAYER
    layers.Dense(1, activation='sigmoid')
])

    print("✅ Model initialized")

    return model




initial_learning_rate = 0.0001

lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps = 1000,
        decay_rate = 0.5
    )

adam = Adam(learning_rate=lr_schedule)

def compile_model_tx(model: Model, optimizer) -> Model:
    """
    Compile the Neural Network
    """


    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

    print("✅ Model compiled")

    return model


def train_model_tx(
        model: Model,
        train_generator,
        val_generator,
        batch_size=16,
        patience=8,
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history_cnn = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=5,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[es]
        )

    print(f"✅ Model trained on {len(train_generator)} rows with min val Accuracy: {round(np.min(history_cnn.history['val_accuracy']), 2)}")

    return model, history_cnn
