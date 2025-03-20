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


def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = tf.keras.Sequential([
    # INPUT LAYER
    layers.Input(shape=(input_shape)),

   # CONV/HIDDEN LAYERS
    layers.Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    layers.Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    # CONV/HIDDEN LAYERS
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform', activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    layers.Dense(128, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    layers.Dense(256, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    layers.Dense(256, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    layers.Dense(128, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    layers.Dense(128, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    layers.Dense(64, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    layers.Dense(32, activation='relu',  kernel_initializer='he_uniform'),
    #layers.Dropout(0.2),
    # PREDICITVE LAYER
    layers.Dense(1, activation='sigmoid')
])

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    adam_opt = optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


    print("✅ Model compiled")

    return model


def evaluate_model2(
        model: Model
        ):
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None
    datagen_test = ImageDataGenerator(
        rescale = 1/255)
    test_generator = datagen_test.flow_from_directory(
        test_path,  # Dossier parent contenant un sous-dossier par classe
        target_size=(1024, 2048),        # Redimensionnement des images
        shuffle=True,
        batch_size=8,
        color_mode ='grayscale', # Nombre d'images par lot
        class_mode='binary')     # Type d'encodage des étiquettes

    metrics = model.evaluate(
        test_generator,
        batch_size=8,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)} and loss : {loss}")

    return metrics


def train_model(
        model: Model,
        train_generator,
        val_generator,
        epochs,
        patience=10,
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history_cnn = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[es]
        )

    print(f"✅ Model trained on {len(train_generator)} images with max val Accuracy: {round(np.max(history_cnn.history['val_accuracy']), 2)}")

    return model, history_cnn
