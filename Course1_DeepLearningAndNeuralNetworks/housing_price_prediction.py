# A simple neural network that predicts the price of houses. 
# Framework: PyTorch
# Inputs(x): [size, bedrooms, zip_score, neighbourhood_wealth]
# Output(y): Price
# Model architecture: Linear(4, hidden) +ReLU + Linear(hidden, 1)
# Training: supervised learning with MSE (mean sqaured error)

import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Creating a demo dataset of 6 houses

# Features:
# x1 = size of the house in 1000 sq ft units, e.g. 1.2 = 1200 sq ft
# x2 = no of bedrooms in the house
# x3 = zip_score (0 to 1: walkability/schools access proxy)
# x4 = wealth_score (0 to 1: neighborhood affluence proxy)

x = torch.tensor([
    [0.85, 2.0, 0.20, 0.25],
    [1.10, 2.0, 0.35, 0.30],
    [1.30, 3.0, 0.40, 0.45],
    [1.60, 3.0, 0.70, 0.65],
    [1.90, 4.0, 0.80, 0.75],
    [2.20, 4.0, 0.90, 0.90],
], dtype=torch.float32)

# Prices of these 6 houses in $100,000 units. Example: 3.2 means $320,000.
y = torch.tensor([
    [1.9],
    [2.2],
    [2.8],
    [3.6],
    [4.3],
    [5.2],
], dtype = torch.float32)

# Step 2: Creating the model

model = nn.Sequential(
    nn.Linear(4, 8), # 4 input features -> 8 hidden units (neurons)
    nn.ReLU(), # ReLU activation: Applies max(0, z) elementwise; adds non-linearity.
    nn.Linear(8, 1) # 8 hidden outputs -> 1 predicted price
)

# Step 3: Creating the loss function

# Using mean sqaured error it compares predicted price vs true price
# Penalizes big errors more than small errors

loss_function = nn.MSELoss()

# Step 4: Creating the optimizer
# Using the optimizer function we decide how we want our model to update weights and biases

optimizer = optim.SGD(model.parameters(), lr = 0.05)

# Step 5: Creating the training loop

# The model improves the weights and biases used for calculation 
# based on the error calculated by the difference in the predicted price and actual price.

epochs = 2000
print()
for epoch in range(1, epochs+1):
    predictions = model(x) # Calculate the predicted values
    loss = loss_function(predictions, y) # Compute the error in the predicted prices calculated

    optimizer.zero_grad() #Clear the old gradients used for calculations
    loss.backward() # Compute the new gradients based on the error
    optimizer.step() # Update weights and biases based on the new gradients computed

    if epoch% 400 == 0:
      print(f"Epcoh {epoch:4d} | Loss: {loss.item():.6f}")

# Step 6: Testing predictions vs actual values of houses after training the model

with torch.no_grad(): # Disables gradient tracking for faster and cleaner inferences
   final_predictions = model(x)

print("\nPredictions vs Actual (in $100k units):")
for i in range(len(x)):
    print(f"House {i+1}: predicted value={final_predictions[i].item():.2f} | actual value={y[i].item():.2f}")


# Step 7: Using the model to predict the price of a new house

# Example new house:
# 1500 sq ft -> 1.50, 3 bedrooms, zip_score 0.55, wealth_score 0.50
new_house = torch.tensor([[1.50, 3.0, 0.55, 0.50]], dtype=torch.float32)

with torch.no_grad():
    new_price = model(new_house).item()

print(f"\nNew house predicted price: {new_price:.2f} ($100k units) ~ ${new_price*100000:.0f}\n")

