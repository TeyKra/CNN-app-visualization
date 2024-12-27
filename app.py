# app.py

import os
import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

from model import load_or_train_model, build_activation_model
from visualization import plot_training_curves
from inference import predict_digit_and_show

MODEL_PATH = "saved_model.h5"

def main():
    st.title("MNIST Digit Recognition with a CNN (Functional API)")
    st.write(
        "Adjust training parameters in the sidebar, then choose to load an existing model or retrain a new one."
    )

    # --- Sidebar controls ---
    st.sidebar.header("Training Configuration")
    epochs = st.sidebar.slider(
        "Number of Epochs", 1, 100, 10, 1
    )
    batch_size = st.sidebar.number_input(
        "Batch Size", min_value=1, max_value=1024, value=64
    )

    # Checkbox to force a new training run (overwriting existing model)
    force_retrain = st.sidebar.checkbox("Retrain Model (Overwrite saved_model.h5)")

    # Button to explicitly load or train the model
    load_train_button = st.sidebar.button("Load/Train Model")

    # We'll store the model/history in Streamlit's session state
    if "model" not in st.session_state:
        st.session_state["model"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = None

    # If user clicks "Load/Train Model", we'll call load_or_train_model
    if load_train_button:
        with st.spinner("Loading or Training Model..."):
            model, history = load_or_train_model(
                epochs=epochs,
                batch_size=batch_size,
                model_path=MODEL_PATH,
                force_retrain=force_retrain
            )
        st.session_state["model"] = model
        st.session_state["history"] = history
        st.success("Model is ready!")
    
    # If we haven't loaded or trained in this session yet,
    # but a model file exists, we can load it automatically if the user wants.
    if st.session_state["model"] is None and os.path.exists(MODEL_PATH):
        st.info("An existing model was found, but hasn't been loaded yet.")
        st.write("Click **Load/Train Model** if you wish to use it or retrain a new model.")
    elif st.session_state["model"] is None:
        st.warning("No model is loaded. Please click **Load/Train Model**.")
    else:
        # We have a model in session state
        model = st.session_state["model"]
        history = st.session_state["history"]

        # Build sub-model for activations
        activation_model = build_activation_model(model)

        if history is not None:
            # Means we actually trained this session
            with st.expander("Show Training Curves"):
                plot_training_curves(history)
        else:
            st.write("Using a previously saved model (no new training in this session).")

        # --- Digit Drawing for Prediction ---
        st.subheader("Draw a digit and click Predict")
        canvas_result = st_canvas(
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("Predict"):
            if canvas_result.image_data is not None:
                # Extract the drawn image
                img = canvas_result.image_data[:, :, 0]

                # Resize to 28x28
                img_28x28 = tf.image.resize(img[..., tf.newaxis], (28, 28)).numpy()

                # Normalize [0..255] -> [0..1]
                img_28x28 = img_28x28 / 255.0

                # Predict
                predict_digit_and_show(img_28x28, model, activation_model)

    st.write("---")
    st.markdown(
        "**Usage Notes**:\n"
        "- If `saved_model.h5` exists, you can load it by clicking **Load/Train Model** with 'Retrain Model' **unchecked**.\n"
        "- If you want to overwrite the existing model, check **Retrain Model** and click **Load/Train Model**.\n"
        "- `history` is only available if you train a new model, so training curves won't appear if you just load a saved model.\n"
    )

if __name__ == "__main__":
    main()
