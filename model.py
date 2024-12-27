# model.py

import os
import tensorflow as tf
from tensorflow.keras import layers, models
import streamlit as st

from data import load_mnist_data

@st.cache_resource
def load_or_train_model(
    epochs=10, 
    batch_size=64, 
    model_path="saved_model.h5", 
    force_retrain=False
):
    """
    Load a pre-trained model if it exists and force_retrain is False.
    Otherwise, train a new model and save it to model_path.

    Args:
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        model_path (str): Where to save / load the model.
        force_retrain (bool): If True, always train a new model
                              and overwrite any existing model file.

    Returns:
        (model, history):
            - model (tf.keras.Model): The trained or loaded Keras model.
            - history (tf.keras.callbacks.History or None): The training
              history if trained, otherwise None if loaded.
    """
    # If we're NOT forcing retrain AND a model file already exists, load it
    if not force_retrain and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded existing model from {model_path}")
        # Return no history since we did not train here
        return model, None

    # Otherwise, train a new model and overwrite the old file
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Build the CNN (Functional API)
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='relu')(x)  # 50 neurons
    outputs = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test set accuracy: {test_acc:.4f}")

    # Save the newly trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return model, history


def build_activation_model(model):
    """
    Build a sub-model that outputs the activations of each layer
    in the given model.
    """
    layer_outputs = [layer.output for layer in model.layers]
    return models.Model(inputs=model.input, outputs=layer_outputs)
