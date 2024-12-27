# MNIST Digit Recognition with a CNN

This project demonstrates how to build, train, and deploy a Convolutional Neural Network (CNN) to recognize hand-written digits (0-9) using the MNIST dataset. A user can draw a digit on a web-based canvas (via Streamlit) and see the model’s prediction along with intermediate layer activations.

- **data.py**  
  Handles loading and preprocessing of the MNIST dataset.

- **model.py**  
  Contains the logic for building and training (or loading from cache) the CNN.  
  Also provides a function to build a sub-model for visualizing layer activations.

- **visualization.py**  
  Includes functions to plot training curves (loss and accuracy).

- **inference.py**  
  Manages the prediction process, including image resizing, normalization, and displaying layer activations.

- **app.py**  
  Main Streamlit application that ties everything together (model loading/training, user interface, and results display).

- **build.bat**  
  Script to set up the environment (e.g., create a virtual environment, install dependencies, etc.).

- **launch.bat**  
  Script to run the application after the setup is complete.

- **requirements.txt**  
  Lists the Python dependencies for the project.

---

## Getting Started

### 1. Prerequisites

- **Python 3.7+**  
- (Recommended) Use a **virtual environment** (e.g., `venv` or Conda) to avoid dependency conflicts.
- GPU is optional but can speed up training.

### 2. Installation and Setup

1. **Clone the repository** (or download the ZIP and extract):
   ```bash
   git clone https://github.com/YourUsername/mnist_app.git
   cd mnist_app
   ```

2. **Run `build.bat`**  
   This script handles environment setup and dependency installation.  
   - On Windows, you can simply double-click `build.bat`, or run from a command prompt:
     ```bat
     build.bat
     ```
   - Check the script contents if you need to customize the virtual environment path or other settings.
---

## How to Run

1. **Launch the Application**  
   After the setup is complete, run the `launch.bat` script:
   ```bat
   launch.bat
   ```
   This will start the Streamlit server and open your web browser at the correct URL (e.g., http://localhost:8501).

2. **Draw a Digit & Predict**  
   In the Streamlit UI:
   - Draw a digit in the canvas (0-9).
   - Click **Predict** to see the model’s guess.
   - View the intermediate layer activations and the normalized 28×28 image.

---

***This code is not perfect and has room for improvement and adaptation. Feel free to adapt it to your needs and explore the fascinating world of GEN AI.***

