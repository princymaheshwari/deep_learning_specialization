# A logistic regression program that does binary classification end to end.
# Framework: NumPy

import numpy as np

# Step 1: Creating the sigmoid function to convert raw confidence scores array into probabilities array for the training examples

def sigmoid(z: np.ndarray) -> np.ndarray:
    probabilities_array = 1/(1+ np.exp(z))
    return probabilities_array

# Step 2: Creating the probabilities prediction function which calculates the raw z values and then converts them into probabilities

def predict_probability(X: np.ndarray, w: np.ndarray, b:float) -> np.ndarray:
    z = X @ w +b
    probabilities_array = sigmoid(z)
    return probabilities_array

# Step 3: Creating a function that converts the probabilities array into an array with class labels (0 or 1) using a threshold.

def predict(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    probabilities = predict_probability(X, w, b)
    predicted_class_labels = (probabilities >= threshold).astype(int)
    return predicted_class_labels
