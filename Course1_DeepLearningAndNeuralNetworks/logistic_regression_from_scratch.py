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

# Step 4: Creating a function that computes the cost function of logistic regression to 
# find the average error in the predicted probabilities of y and use it to learn the parameters w and b.

def cost_function(y_true : np.ndarray, y_predicted_probability: np.ndarray) -> float:

    epsilon = 1e-12 # Used to avoid log(0)
    y_predicted_probability = np.clip(y_predicted_probability, epsilon, 1-epsilon) # Forces all predicted probabilities of y to be between epsilon and 1-epsilon

    loss_function = (y_true * np.log(y_predicted_probability)) + ((1-y_true) * np.log(1 - y_predicted_probability))
    avg_error = float(np.mean(loss_function))

    return avg_error

