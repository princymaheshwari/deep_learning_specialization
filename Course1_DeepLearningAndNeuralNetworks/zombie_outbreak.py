"""
Zombie Outbreak Simulation using Sigmoid Probability

This script models a simplified zombie outbreak in a population using a 
probabilistic approach inspired by machine learning concepts.

Each individual is assigned a randomly generated exposure score, which is 
converted into an infection probability using the sigmoid function. This 
demonstrates how raw numerical values can be transformed into probabilities.

Instead of using a fixed threshold, infections are determined using random 
sampling (Bernoulli trials), allowing the simulation to reflect real-world 
uncertainty.

The program then aggregates the results to compute:
- Total number of infected individuals
- Overall infection rate
- City-wide alert level based on severity

Concepts demonstrated:
- Sigmoid function as a probability mapping
- Vectorized computation using NumPy
- Probabilistic decision-making vs deterministic thresholds
- Basic simulation and aggregation logic
"""

import numpy as np

# Step 1: Creating the sigmoid function
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Step 2: Generating random exposure levels for the poopulation
def generate_exposure_levels(no_of_people):
    """
    Negative = safe
    Positive = danger
    """
    return np.random.randn(no_of_people) * 5

# Step 3: Coverting exposure levels to infection probability
def infection_probability(exposure_scores):
    return sigmoid(exposure_scores)

# Step 4: Simulating Infections using Random Sampling (Bernoulli Distribution)
def simulate_infections(probabilities):
    random_thresholds = np.random.rand(len(probabilities))
    return (probabilities > random_thresholds).astype(int)

# Step 5: Computing City Statistics
def compute_statistics(infections):
    total_people = len(infections)
    no_of_infected_people = sum(infections)
    infection_rate = no_of_infected_people / total_people

    return no_of_infected_people, infection_rate

# Step 6: Alert system for the city
def get_alert_level(infection_rate):
    if infection_rate < 0.2:
        return "🟢 Low Risk"
    elif infection_rate < 0.5:
        return "🟡 Moderate Risk"
    elif infection_rate < 0.8:
        return "🟠 High Risk"
    else:
        return "🔴 Critical"
    
# Step 7: Full Simulation of the city
def run_city_simulation(no_of_people= 500):
    raw_exposure_scores = generate_exposure_levels(no_of_people)
    infection_probabilities = infection_probability(raw_exposure_scores)
    infections = simulate_infections(infection_probabilities)

    no_of_infected_people, infection_rate = compute_statistics(infections)
    alert_level = get_alert_level(infection_rate)

    print(f"Total Population: {no_of_people}")
    print(f"No Of Infected People: {no_of_infected_people}")
    print(f"Infection Rate: {infection_rate}")
    print(f"Alert level: {alert_level}")

if __name__ == "__main__":
    run_city_simulation()