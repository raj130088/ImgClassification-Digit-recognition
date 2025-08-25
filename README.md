# MNIST Digit Recognition with CNN

A classic deep learning project implementing a Convolutional Neural Network (CNN) to classify handwritten digits from the famous MNIST dataset. This project was developed as a learning exercise to practice image classification fundamentals using TensorFlow and Keras.

## Project Overview

This repository contains code for building, training, and evaluating a CNN model that achieves high accuracy in recognizing handwritten digits (0-9). The project demonstrates a complete machine learning workflow, from data loading and preprocessing to model deployment and performance visualization.

## Features

*   **Data Handling:** Uses `tf.data.Dataset` for efficient data loading, preprocessing, and batching.
*   **CNN Architecture:** Implements a sequential CNN model with multiple convolutional and pooling layers for feature extraction, followed by dense layers for classification.
*   **Preprocessing:** Includes image normalization (scaling pixel values to [0, 1]) and one-hot encoding of labels.
*   **Training & Evaluation:** Trains the model using the Adam optimizer and categorical crossentropy, with validation on a held-out set.
*   **Visualization:** Provides code to plot training history (loss/accuracy curves) and visualize model predictions on sample images.

## Tech Stack

*   **Python**
*   **TensorFlow / Keras**
*   **NumPy**
*   **scikit-learn** (for train/validation split)
*   **Matplotlib** (for visualization)

## Dataset

The model is trained on the **MNIST dataset**, a benchmark dataset in machine learning containing 70,000 grayscale images of handwritten digits (28x28 pixels each).
*   60,000 images for training
*   5,000 images for validation (split from the training set)
*   10,000 images for testing

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/mnist-digit-recognition.git
    cd mnist-digit-recognition
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment. You can install the required packages using:
    ```bash
    pip install tensorflow numpy scikit-learn matplotlib
    ```

3.  **Run the Jupyter Notebook:**
    The main code is in the `Image_classification_digit_recgn.ipynb` notebook.
    ```bash
    jupyter notebook Image_classification_digit_recgn.ipynb
    ```

    The notebook will automatically download the MNIST dataset when run for the first time.

## Model Architecture

The model is a sequential CNN with the following structure:
```
Input (28, 28, 1)
↓
Conv2D (32 filters, 3x3, ReLU)
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters, 3x3, ReLU)
↓
MaxPooling2D (2x2)
↓
Conv2D (128 filters, 3x3, ReLU)
↓
MaxPooling2D (2x2)
↓
Flatten
↓
Dense (128 units, ReLU)
↓
Dense (10 units, Softmax) # Output layer
```

## Results

After 5 epochs of training, the model typically achieves:
*   **Training Accuracy:** ~99%
*   **Validation Accuracy:** ~98%

The learning curves and sample predictions (see notebook) confirm the model effectively learns to classify digits without significant overfitting on this dataset.

## Future Improvements

This being a practice project, there are several avenues for future exploration:
*   Experiment with different model architectures (deeper networks, adding Dropout/BatchNorm layers).
*   Implement data augmentation (rotations, shifts, zooms) to improve generalization.
*   Tune hyperparameters (learning rate, batch size, optimizer).
*   Export the trained model for use in a simple web application.

## Contributing

This is a personal practice project, but suggestions and feedback are always welcome! Feel free to open an issue or submit a pull request.


**Note:** This was one of my first projects while learning TensorFlow and deep learning. The code reflects a foundational understanding of CNNs and the Keras API.
