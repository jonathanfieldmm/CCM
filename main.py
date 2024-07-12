import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpBinary
import numpy as np
import folium
from folium.plugins import PolyLineTextPath
import math
# Sample data ffor hubs
hubs_data = {
    'Site Reference': ['Hub01', 'Hub02', 'Hub03', 'Hub04', 'Hub05', 'Hub06', 'Hub07', 'Hub08', 'Hub09', 'Hub10'],
    'X Coordinates': [51.5074, 53.4808, 55.9533, 52.4862, 53.4084, 52.6309, 54.9783, 53.8008, 50.3755, 55.8642],
    'Y Coordinates': [-0.1278, -2.2426, -3.1883, -1.8904, -2.9916, 1.2974, -1.6174, -1.5491, -4.1427, -4.2518],
    'Heat Available (kWh/year)': [500000, 750000, 600000, 450000, 800000, 700000, 650000, 550000, 720000, 600000],
    'Cost of Heat (£/kWh)': [0.05, 0.04, 0.06, 0.05, 0.04, 0.06, 0.07, 0.05, 0.04, 0.06],
    'Max Capacity (tonnes/year)': [10000, 15000, 12000, 9000, 16000, 14000, 13000, 11000, 15000, 12000]
}
hubs_df = pd.DataFrame(hubs_data)

# Sample data for feedstock sources
sources_data = {
    'Site Reference': ['FS01', 'FS02', 'FS03', 'FS04', 'FS05', 'FS06', 'FS07', 'FS08', 'FS09', 'FS10'],
    'X Coordinates': [52.2053, 51.5095, 53.7893, 51.4543, 50.8225, 51.4816, 53.7632, 52.4862, 50.7192, 55.3781],
    'Y Coordinates': [0.1218, -0.0959, -2.2425, -2.5879, -0.1373, -0.4725, -2.7034, -1.8904, -1.8808, -3.4360],
    'Available Quantity (tonnes/year)': [20000, 25000, 18000, 30000, 15000, 22000, 28000, 27000, 23000, 21000],
    'Water Content (%)': [30, 25, 28, 22, 35, 32, 27, 29, 31, 26],
    'Purchase Price (£/tonne)': [45, 50, 42, 48, 40, 47, 44, 49, 41, 46]
}
sources_df = pd.DataFrame(sources_data)

# Key inputs
conversion_factor = 0.7 #Feedstock to fertilizer, adjust as needed
heat_required_per_tonne = 10  # in kWh per tonne of fertilizer, adjust as needed
haulage_cost_per_tonne_mile = 0.02  # Example haulage cost
minimum_total_production = 120000  # Example minimum total production requirement
generic_capex = 100000  # Example generic CAPEX value for each hub


# Define the Haversine formula to calculate distances
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


# Calculate distances between all sources and hubs
distances = {}
for _, source in sources_df.iterrows():
    for _, hub in hubs_df.iterrows():
        key = (source['Site Reference'], hub['Site Reference'])
        distances[key] = haversine(source['Y Coordinates'], source['X Coordinates'],
                                   hub['Y Coordinates'], hub['X Coordinates'])

# Optimization problem setup
prob = LpProblem("Minimize_Costs", LpMinimize)

# Decision variables
transport_vars = LpVariable.dicts("Transport",
                                  [(i, j) for i in sources_df['Site Reference'] for j in hubs_df['Site Reference']],
                                  lowBound=0, cat='Continuous')
# Binary variables to indicate if a hub is active
hub_active = LpVariable.dicts("HubActive",
                              hubs_df['Site Reference'],
                              cat='Binary')

# Define costs

# Transportation costs
transportation_costs = lpSum([transport_vars[i, j] * distances[(i, j)] * haulage_cost_per_tonne_mile
                              for i, j in transport_vars])

# Production costs
production_costs = lpSum([
    (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) *
    hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] *
    heat_required_per_tonne
    for j in hubs_df['Site Reference']
])

# CAPEX costs (only for active hubs)
capex_costs = lpSum([hub_active[j] * generic_capex for j in hubs_df['Site Reference']])

# Objective function
prob += transportation_costs + production_costs + capex_costs, "Total Costs"

# Constraints
for i in sources_df['Site Reference']:
    prob += lpSum([transport_vars[i, j] for j in hubs_df['Site Reference']]) <= \
            sources_df.set_index('Site Reference').at[i, 'Available Quantity (tonnes/year)'], f"Supply_constraint_{i}"

for j in hubs_df['Site Reference']:
    prob += (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) <= \
            hubs_df.set_index('Site Reference').at[j, 'Max Capacity (tonnes/year)'], f"Demand_constraint_{j}"

# Ensure that if any feedstock is transported to a hub, the hub is active and capex is iccured for it
for j in hubs_df['Site Reference']:
    prob += lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) <= hub_active[j] * sum(
        sources_df['Available Quantity (tonnes/year)']), f"Activation_constraint_{j}"



# Minimum total production constraint
prob += lpSum([lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor
               for j in hubs_df['Site Reference']]) >= minimum_total_production, "Min_Total_Production"

# Solve the problem
prob.solve()
print("Status:", LpStatus[prob.status])

# Calculate total costs
total_transportation_cost = sum(transport_vars[i, j].varValue * distances[(i, j)] * haulage_cost_per_tonne_mile
                                for i, j in transport_vars)

total_production_cost = sum(
    (sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor) *
    hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] for j in hubs_df['Site Reference'])

total_capex = sum(hub_active[j].varValue * generic_capex for j in hubs_df['Site Reference'])

total_cost = total_transportation_cost + total_production_cost + total_capex

# Calculate total production quantity
total_production_quantity = sum(
    sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor
    for j in hubs_df['Site Reference'])

# Calculate cost per tonne produced
cost_per_tonne_produced = total_cost / total_production_quantity

# Output the results
print(f"\nTotal transportation cost: £{total_transportation_cost:.2f}")
print(f"Total production cost: £{total_production_cost:.2f}")
print(f"Total CAPEX: £{total_capex:.2f}")
print(f"Total cost: £{total_cost:.2f}")
print(f"Total production quantity: {total_production_quantity:.2f} tonnes")
print(f"Cost per tonne produced: £{cost_per_tonne_produced:.2f}")

# Debug output to verify the results of the optimization
print("\nProduction quantities at each hub:")
for j in hubs_df['Site Reference']:
    production_quantity = sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor
    print(f"Hub {j}: {production_quantity} tonnes")

print("\nAmount of feedstock transported between each source and hub:")
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        print(f"Transport from Source {i} to Hub {j}: {transport_vars[i, j].varValue} tonnes")

# Visualization with Folium
map_osm = folium.Map(location=[55, -3], zoom_start=6)
# Calculate maximum transported quantity for scaling line thickness
max_transport = max(transport_vars[i, j].varValue for i, j in transport_vars)
# Add markers for hubs with quantity used and maximum quantity
for idx, row in hubs_df.iterrows():
    production_quantity = sum(
        transport_vars[i, row['Site Reference']].varValue for i in sources_df['Site Reference']) * conversion_factor
    folium.Marker(
        [row['X Coordinates'], row['Y Coordinates']],
        popup=(f"Hub: {row['Site Reference']}<br>"
               f"Quantity Used: {production_quantity:.2f} tonnes<br>"
               f"Max Capacity: {row['Max Capacity (tonnes/year)']} tonnes"),
        icon=folium.Icon(color='blue', icon='industry', prefix='fa')
    ).add_to(map_osm)

# Add markers for sources with available and used quantity
for idx, row in sources_df.iterrows():
    used_quantity = sum(transport_vars[row['Site Reference'], j].varValue for j in hubs_df['Site Reference'] if
                        (row['Site Reference'], j) in transport_vars)
    folium.Marker(
        [row['X Coordinates'], row['Y Coordinates']],
        popup=(f"Source: {row['Site Reference']}<br>"
               f"Available Quantity: {row['Available Quantity (tonnes/year)']} tonnes<br>"
               f"Quantity Used: {used_quantity:.2f} tonnes"),
        icon=folium.Icon(color='green', icon='leaf', prefix='fa')
    ).add_to(map_osm)

# Add lines for transportation routes (only where transport is greater than 0) with a single arrow in the center
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        source = sources_df[sources_df['Site Reference'] == i].iloc[0]
        hub = hubs_df[hubs_df['Site Reference'] == j].iloc[0]
        line_weight = (transport_vars[i, j].varValue / max_transport) * 10  # Scale line thickness

        line = folium.PolyLine(
            locations=[(source['X Coordinates'], source['Y Coordinates']),
                       (hub['X Coordinates'], hub['Y Coordinates'])],
            weight=line_weight, color='red',
            popup=(f"Transport from {i} to {j}: {transport_vars[i, j].varValue:.2f} tonnes")
        ).add_to(map_osm)



# Save or display the map
map_osm.save('full_network_map.html')