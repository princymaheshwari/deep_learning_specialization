# A logistic regression program that does binary classification end to end.
# Framework: NumPy

import numpy as np

# Step 1: Creating the sigmoid function to convert raw confidence scores array into probabilities array for the training examples

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500) # To avoid sigmoid reaching infinity which causes runtime error
    probabilities_array = 1/(1+ np.exp(-z))
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

# Step 5: Creating the gradient descent function which loops over the training set to learn and update the parameters w and b
# Learns w and b so that sigmoid(X @ w + b) matches y.

print()
def gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 2000):
    
    # Initializing the parameters w and b to 0
    m, n = X.shape
    w = np.zeros(n, dtype = float)
    b = 0.0

    # Setting up the learning loop
    for epoch in range(1, epochs+1):

        # Forward pass: Calculating the predicted probability
        y_hat = predict_probability(X, w, b)

        # Calculating partial derivatives of the cost function w.r.t to the parameters
        dw = (1/m) * (X.T @ (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)

        # Gradient descent update
        w = w - lr*dw
        b = b - lr*db

        if epoch % 400 == 0:
            current_loss = cost_function(y, y_hat)
            print(f"Epoch: {epoch:4d} | error = {current_loss:.4f}")

    return w, b

# Step 5: Trying the model on Demo dataset (binary classification)

# Domain example: "Will the user click an ad?"
# Features:
# x1 = time_on_site_minutes (scaled)
# x2 = number_of_pages_viewed (scaled)
# x3 = returning_user (0 or 1)

# Label:
# y = 1 if clicked, 0 if not clicked

X = np.array([
    [0.10, 0.20, 0],
    [0.15, 0.10, 0],
    [0.20, 0.30, 0],
    [0.40, 0.50, 1],
    [0.60, 0.70, 1],
    [0.80, 0.60, 1],
    [0.90, 0.90, 1],
    [0.30, 0.25, 0],
], dtype=float)

y = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)

# Train
w, b = gradient_descent(X, y, lr=0.5, epochs=2000)

print("\nLearned parameters:")
print("w =", w)
print("b =", b)

# Evaluate on training data
probabilities = predict_probability(X, w, b)
predictions = predict(X, w, b, threshold=0.5)

print("\nPredictions:")
for i in range(len(X)):
    print(f"x={X[i]} -> Probability(click)= {probabilities[i]:.3f} -> prediction={predictions[i]} actual={y[i]}")

print()

