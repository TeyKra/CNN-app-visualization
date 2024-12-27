import tensorflow as tf

def load_mnist_data():
    """
    Load the MNIST dataset and preprocess it:
      1) Normalize pixel values from [0..255] to [0..1]
      2) Reshape data to (batch_size, 28, 28, 1)
    
    Returns:
        (x_train, y_train), (x_test, y_test):
            - x_train, x_test: Preprocessed images
            - y_train, y_test: Corresponding labels
    """
    # 1) Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2) Normalize: [0..255] -> [0..1]
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    # 3) Reshape to (batch, 28, 28, 1)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test  = x_test.reshape((-1, 28, 28, 1))

    return (x_train, y_train), (x_test, y_test)
