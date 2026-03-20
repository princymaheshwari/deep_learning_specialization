import numpy as np
import json

# Step 1: Creating the Sigmoid Function
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Step 2: Generating random exposure levels for the poopulation
def generate_exposure_levels(no_of_people):
    """
    Generate exposure scores for individuals in an area by combining randomness 
    with an area-specific bias.

    A density factor is sampled to represent how crowded the area is. This density 
    is transformed into a bias using a logarithmic function, which shifts the 
    distribution of exposure scores. Higher density increases the bias (more risk), 
    while lower density decreases it (safer conditions).

    The final exposure scores are drawn from a normal distribution and shifted by 
    this bias, producing a more realistic spread of risk across different areas.
    """

    density_factor = np.random.uniform(0.5, 2.0)
    area_bias = np.log(density_factor) * 2
    return np.random.randn(no_of_people) * 2 + area_bias

# Step 3: Coverting exposure levels to infection probability
def infection_probability(exposure_scores):
    return sigmoid(exposure_scores)

# Step 4: Simulating Infections using Random Sampling (Bernoulli Distribution)
def simulate_infections(probabilities):
    random_thresholds = np.random.rand(len(probabilities))
    return (probabilities > random_thresholds).astype(int)

# Step 5: Computing Area Statistics
def compute_statistics(infections):
    total_people = len(infections)
    no_of_infected_people = sum(infections)
    infection_rate = no_of_infected_people / total_people

    return no_of_infected_people, infection_rate

# Step 6: Loading GeoJSON and extracting area names
def load_areas(geojson_path):
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)

    areas = []

    for feature in geojson["features"]:
        area_name = feature["properties"]["name"]
        areas.append(area_name)

    return areas

# Step 7: Simulating the infection per area
def simulate_area(population):

    exposure_scores = generate_exposure_levels(population)
    probabilities = infection_probability(exposure_scores)
    infections = simulate_infections(probabilities)
    no_of_infected_people, infection_rate = compute_statistics(infections)

    return no_of_infected_people, infection_rate

# Step 8: Alert system for each area
def get_alert_level(infection_rate):
    if infection_rate < 0.45:
        return "🟢 Low Risk"
    elif infection_rate < 0.55:
        return "🟡 Moderate Risk"
    elif infection_rate < 0.65:
        return "🟠 High Risk"
    else:
        return "🔴 Critical"
    
# Step 9: Simulating all the areas
def run_simulation(geojson_path):
    areas = load_areas(geojson_path)

    results = []

    for area in areas:

        population = np.random.randint(100, 1000)
        no_of_infected_people, infection_rate = simulate_area(population)
        alert_level = get_alert_level (infection_rate)

        results.append({
            "name": area,
            "population" : population,
            "Infected People": no_of_infected_people,
            "Infection Rate": infection_rate,
            "Alert Level": alert_level
        })

    return results