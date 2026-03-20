import json
import pandas as pd
import plotly.express as px
import plotly.io as pio

from simulation import run_simulation

# Load GeoJSON
with open("atlanta.geojson") as f:
    geojson_data = json.load(f)

# Run Simulation
results = run_simulation("atlanta.geojson")

# Creating the DataFrame
df = pd.DataFrame(results)

# Creating the map
fig = px.choropleth(
    df,
    geojson=geojson_data,
    locations="name",
    featureidkey="properties.name",
    color="infection_rate",
    hover_data=["name","population", "infected_people", "alert_level"],
    color_continuous_scale="RdYlGn_r"
)

fig.update_geos(fitbounds="locations", visible=False)
pio.renderers.default = "browser"
fig.show()
