import pandas as pd
import numpy as np
import folium
from folium.plugins import PolyLineTextPath
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpBinary
import math
import io

# Function to read data from Excel files
def load_data(file_path):
    return pd.read_excel(file_path)

# Haversine formula to calculate distances
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Load data
hubs_df = load_data("Data/Hubs_Data_Actual_Reduced.xlsx")
sources_df = load_data("Data/Sources_Data_Actual.xlsx")

# Main inputs
minimum_total_production = 120000

# Cost inputs
haulage_cost_per_tonne_mile = 0.02
generic_capex = 100000
cost_co2 = 50
cost_ammonia = 100
cost_phosphorus = 80
cost_nitrogen = 70

# Process inputs
conversion_factor = 0.7
heat_required_per_tonne = 1.0
co2_per_tonne = 0.2
ammonia_per_tonne = 0.1
phosphorus_per_tonne = 0.05
nitrogen_per_tonne = 0.15

# Calculate distances between all sources and hubs
distances = {}
for _, source in sources_df.iterrows():
    for _, hub in hubs_df.iterrows():
        key = (source['Site Reference'], hub['Site Reference'])
        distances[key] = haversine(source['Y Coordinates'], source['X Coordinates'], hub['Y Coordinates'], hub['X Coordinates'])

# Optimization problem setup
prob = LpProblem("Minimize_Costs", LpMinimize)

# Decision variables
transport_vars = LpVariable.dicts("Transport", [(i, j) for i in sources_df['Site Reference'] for j in hubs_df['Site Reference']], lowBound=0, cat='Continuous')
hub_active = LpVariable.dicts("HubActive", hubs_df['Site Reference'], cat='Binary')

# Define costs
transportation_costs = lpSum([transport_vars[i, j] * distances[(i, j)] * haulage_cost_per_tonne_mile for i, j in transport_vars])
production_costs = lpSum([
    (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) *
    (hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne +
     co2_per_tonne * cost_co2 +
     ammonia_per_tonne * cost_ammonia +
     phosphorus_per_tonne * cost_phosphorus +
     nitrogen_per_tonne * cost_nitrogen)
    for j in hubs_df['Site Reference']
])
capex_costs = lpSum([hub_active[j] * generic_capex for j in hubs_df['Site Reference']])

# Objective function
prob += transportation_costs + production_costs + capex_costs, "Total Costs"

# Constraints
for i in sources_df['Site Reference']:
    prob += lpSum([transport_vars[i, j] for j in hubs_df['Site Reference']]) <= sources_df.set_index('Site Reference').at[i, 'Available Quantity (tonnes/year)'], f"Supply_constraint_{i}"
for j in hubs_df['Site Reference']:
    prob += (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) <= hubs_df.set_index('Site Reference').at[j, 'Max Capacity (tonnes/year)'], f"Demand_constraint_{j}"
for j in hubs_df['Site Reference']:
    prob += lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) <= hub_active[j] * sum(sources_df['Available Quantity (tonnes/year)']), f"Activation_constraint_{j}"
prob += lpSum([lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor for j in hubs_df['Site Reference']]) >= minimum_total_production, "Min_Total_Production"

# Solve the problem
print("Starting optimization...")
prob.solve()

# Output results
print("\nOptimization Results:")
print(f"Status: {LpStatus[prob.status]}")

# Calculate total costs
total_transportation_cost = sum(transport_vars[i, j].varValue * distances[(i, j)] * haulage_cost_per_tonne_mile for i, j in transport_vars)
total_production_cost = sum(
    (sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor) *
    (hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne +
     co2_per_tonne * cost_co2 +
     ammonia_per_tonne * cost_ammonia +
     phosphorus_per_tonne * cost_phosphorus +
     nitrogen_per_tonne * cost_nitrogen)
    for j in hubs_df['Site Reference']
)
total_capex = sum(hub_active[j].varValue * generic_capex for j in hubs_df['Site Reference'])
total_cost = total_transportation_cost + total_production_cost + total_capex
total_production_quantity = sum(sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor for j in hubs_df['Site Reference'])
cost_per_tonne_produced = total_cost / total_production_quantity

print(f"Total transportation cost: £{total_transportation_cost:.2f}")
print(f"Total production cost: £{total_production_cost:.2f}")
print(f"Total CAPEX: £{total_capex:.2f}")
print(f"Total cost: £{total_cost:.2f}")
print(f"Total production quantity: {total_production_quantity:.2f} tonnes")
print(f"Cost per tonne produced: £{cost_per_tonne_produced:.2f}")

# Debug output to verify the results of the optimization
print("\nProduction quantities at each hub:")
for j in hubs_df['Site Reference']:
    production_quantity = sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor
    print(f"Hub {j}: {production_quantity:.2f} tonnes")

print("\nAmount of feedstock transported between each source and hub:")
transport_data = []
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        amount_transported = transport_vars[i, j].varValue
        transport_data.append({"Source": i, "Hub": j, "Amount Transported (tonnes)": amount_transported})
        print(f"From {i} to {j}: {amount_transported:.2f} tonnes")

# Create a DataFrame for the transport data
transport_df = pd.DataFrame(transport_data)

# Save the transport data to an Excel file
transport_df.to_excel("transport_data.xlsx", index=False)
print("\nTransport data saved to 'transport_data.xlsx'.")

# Visualization with Folium
print("Generating network map...")
map_osm = folium.Map(location=[55, -3], zoom_start=6)
max_transport = max(transport_vars[i, j].varValue for i, j in transport_vars)
for idx, row in hubs_df.iterrows():
    production_quantity = sum(transport_vars[i, row['Site Reference']].varValue for i in sources_df['Site Reference']) * conversion_factor
    folium.Marker([row['X Coordinates'], row['Y Coordinates']],
                  popup=(f"Hub: {row['Site Reference']}<br>"
                         f"Quantity Used: {production_quantity:.2f} tonnes<br>"
                         f"Max Capacity: {row['Max Capacity (tonnes/year)']} tonnes"),
                  icon=folium.Icon(color='blue', icon='industry', prefix='fa')).add_to(map_osm)
for idx, row in sources_df.iterrows():
    used_quantity = sum(transport_vars[row['Site Reference'], j].varValue for j in hubs_df['Site Reference'] if (row['Site Reference'], j) in transport_vars)
    folium.Marker([row['X Coordinates'], row['Y Coordinates']],
                  popup=(f"Source: {row['Site Reference']}<br>"
                         f"Available Quantity: {row['Available Quantity (tonnes/year)']} tonnes<br>"
                         f"Quantity Used: {used_quantity:.2f} tonnes"),
                  icon=folium.Icon(color='green', icon='leaf', prefix='fa')).add_to(map_osm)
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        source = sources_df[sources_df['Site Reference'] == i].iloc[0]
        hub = hubs_df[hubs_df['Site Reference'] == j].iloc[0]
        line_weight = 10
        line = folium.PolyLine(locations=[(source['X Coordinates'], source['Y Coordinates']), (hub['X Coordinates'], hub['Y Coordinates'])],
                               weight=line_weight, color='red',
                               popup=(f"Transport from {i} to {j}: {transport_vars[i, j].varValue:.2f} tonnes")).add_to(map_osm)
map_osm.save("network_map.html")
print("Network map saved to 'network_map.html'.")
