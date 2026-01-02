#Simple model that predicts the chance a user clicks on an ad using PyTorch.

import torch 
import torch.nn as nn
import torch.optim as optim

# Creating the model architecture
model = nn.Sequential(
    nn.Linear(5, 10),  # Input: 5 user features → 10 neurons → 10 outputs
    nn.ReLU(),         # Activation adds non-linearity
    nn.Linear(10, 1),  # Output: 1 value (probability)
    nn.Sigmoid()       # Squashes to 0-1 range
)

# Printing the model architecture
print(model)

#Creating fake input array to test on the model
x = torch.randn(1,5) # Generates a tensor filled with 5 random numbers from normal distribution

# Giving the 5 randomly generated numbers as input to the model
y = model(x)

# Printing the results
print(f"Input: {x}")
print(f"Predicted Click Probability {y}")

