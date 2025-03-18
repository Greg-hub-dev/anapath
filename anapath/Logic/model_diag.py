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
    layers.Dense(32, activation='relu',  kernel_initializer='he_uniform'),
    layers.Dropout(0.2),

    # PREDICITVE LAYER
    layers.Dense(2, activation='softmax')
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
        model: Model,
        batch_size=1):
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None
    datagen = ImageDataGenerator(
        rescale = 1/255)
    test_generator = datagen.flow_from_directory(
        test_path,  # Dossier parent contenant un sous-dossier par classe
        target_size=(512, 1024),        # Redimensionnement des images
        batch_size=1,                 # Nombre d'images par lot
        class_mode='categorical')      # Type d'encodage des étiquettes (one-hot pour multi-classes)

    metrics = model.evaluate(
        test_generator,
        batch_size=batch_size,
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
        batch_size=2,
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
        batch_size=batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        )

    print(f"✅ Model trained on {len(train_generator)} rows with min val Accuracy: {round(np.min(history_cnn.history['val_accuracy']), 2)}")

    return model, history_cnn
