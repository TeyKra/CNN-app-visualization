import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st

from model import build_activation_model

def get_activations(img_28x28, activation_model):
    """
    Return the list of activations for each layer of the activation_model.
    
    Args:
        img_28x28 (np.ndarray): Image of shape (28, 28, 1), normalized in [0..1].
        activation_model (tf.keras.Model): Model that returns the outputs of each layer.
    
    Returns:
        List of activation outputs for each layer.
    """
    batch = np.expand_dims(img_28x28, axis=0)
    return activation_model.predict(batch)

def predict_digit_and_show(img_28x28, model, activation_model):
    """
    1) Predict the digit from a (28x28) normalized image.
    2) Display the input image.
    3) Display the activations of each layer (both Convolutional and Dense).
    
    Args:
        img_28x28 (np.ndarray): Image of shape (28, 28, 1), normalized.
        model (tf.keras.Model): Trained digit classification model.
        activation_model (tf.keras.Model): Sub-model that outputs the activations.
    """
    # Perform the digit prediction
    pred = model.predict(tf.expand_dims(img_28x28, axis=0))
    predicted_digit = np.argmax(pred)

    st.markdown(f"**Predicted Digit:** {predicted_digit}")

    # Display the input image
    st.subheader("Normalized Input (28x28)")
    fig_input, ax = plt.subplots()
    ax.imshow(img_28x28.squeeze(), cmap='gray')
    ax.axis('off')
    st.pyplot(fig_input)
    plt.close(fig_input)

    # Retrieve and display activations
    st.subheader("Layer Activations")
    activations = get_activations(img_28x28, activation_model)

    for i, act in enumerate(activations):
        layer_name = activation_model.layers[i].name
        st.write(f"**Layer {i} – {layer_name}** : shape={act.shape}")

        # Convolution or pooling layers: shape = (1, H, W, filters)
        if len(act.shape) == 4:
            num_filters = act.shape[-1]
            num_filters_to_show = min(num_filters, 6)  # For display, limit to 6
            fig, axes = plt.subplots(1, num_filters_to_show, figsize=(15, 5))

            # If there's only one filter, axes might not be iterable
            if num_filters_to_show == 1:
                axes = [axes]

            for f in range(num_filters_to_show):
                axes[f].imshow(act[0, :, :, f], cmap='viridis')
                axes[f].axis('off')
            st.pyplot(fig)
            plt.close(fig)

        # Dense layers: shape = (1, num_neurons)
        elif len(act.shape) == 2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(act.shape[1]), act[0], color='steelblue')
            ax.set_title("Dense Layer Activations")
            ax.set_xlabel("Neuron Index")
            ax.set_ylabel("Activation")
            st.pyplot(fig)
            plt.close(fig)

        # Flatten or other layers that don't fit the above shapes
        else:
            st.write(act)
