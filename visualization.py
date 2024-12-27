import matplotlib.pyplot as plt
import streamlit as st

def plot_training_curves(history):
    """
    Plot and display training and validation curves for loss and accuracy
    within Streamlit.
    
    Args:
        history: The history object returned by model.fit().
                 Must contain 'loss', 'accuracy', 'val_loss', and 'val_accuracy'.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # 1) Plot Loss
    ax[0].plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        ax[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')
    ax[0].legend()
    ax[0].grid(True)

    # 2) Plot Accuracy
    ax[1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    if 'val_accuracy' in history.history:
        ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)
    plt.close(fig)
