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

# Load primary data
hubs_df = load_data("Data/Industrial_Hubs_Main.xlsx")
sources_df = load_data("Data/Digestate_Sources_Main.xlsx")

# Tag rows to identify their source as 'main'
hubs_df['Source'] = 'main'
sources_df['Source'] = 'main'

# Separate options to include additional hubs and sources
include_additional_hubs = 'yes'   # additional hubs represent using data centres as a source of heat
include_additional_sources = 'yes' # additional sources represent using compost as a feedstock

# Load and merge additional hubs if user selects 'yes'
if include_additional_hubs == 'yes':
    additional_hubs_df = load_data("Data/Data_Centres_Main.xlsx")
    additional_hubs_df['Source'] = 'additional'  # Tag additional hubs
    hubs_df = pd.concat([hubs_df, additional_hubs_df], ignore_index=True)

# Load and merge additional sources if user selects 'yes'
if include_additional_sources == 'yes':
    additional_sources_df = load_data("Data/Compost_Sources_Main.xlsx")
    additional_sources_df['Source'] = 'additional'  # Tag additional sources
    sources_df = pd.concat([sources_df, additional_sources_df], ignore_index=True)


# Main inputs
minimum_total_production = 300000
minimum_hub_production = 15000  # Minimum production per hub

# Cost inputs
haulage_cost_per_mile = 1.86   # £/km
average_load=20  # Tonnes
haulage_cost_per_tonne_mile = haulage_cost_per_mile/average_load    # £/(tonne*km)
fixed_capex = 100000  # Fixed CAPEX per hub £
variable_capex_per_tonne = 300  # Variable CAPEX per tonne of production £/tonne
cost_dolomite = 40  # £
cost_urea = 60    # £

# Process inputs
conversion_factor = 0.7
heat_required_per_tonne = 1.0
dolomite_per_tonne = 0.1
urea_per_tonne = 0.2

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
# Extract the purchase price of feedstock from the sources data
purchase_prices = sources_df.set_index('Site Reference')['Purchase Price (£/tonne)']

# Define costs
production_costs = lpSum([
    (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) *
    (hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne +
     dolomite_per_tonne * cost_dolomite +
     urea_per_tonne * cost_urea) +
    lpSum([transport_vars[i, j] * purchase_prices[i] for i in sources_df['Site Reference']])  # Adding the purchase price of feedstock
    for j in hubs_df['Site Reference']
])

# Calculate CAPEX costs including both fixed and variable components
capex_costs = lpSum([
    hub_active[j] * fixed_capex +  # Fixed cost when the hub is active
    (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) * variable_capex_per_tonne  # Variable cost per tonne produced
    for j in hubs_df['Site Reference']
])

# Objective function
prob += transportation_costs + production_costs + capex_costs, "Total Costs"

# Constraints


# Constraints
for i in sources_df['Site Reference']:
    prob += lpSum([transport_vars[i, j] for j in hubs_df['Site Reference']]) <= sources_df.set_index('Site Reference').at[i, 'Available Quantity (tonnes/year)'], f"Supply_constraint_{i}"

for j in hubs_df['Site Reference']:
    # Total production at each hub
    hub_production = lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor

    # Lower production constraint - ensure minimum production when hub is active
    # If any feedstock is transported to the hub, hub_active[j] will be 1
    # Hub must produce at least the minimum required quantity
    prob += hub_production >= hub_active[j] * minimum_hub_production, f"Capacity_lower_constraint_{j}"

    # Ensure that if the hub receives any feedstock, it must be active (and thus incur capex), as well as Upper production constraint (capacity limit)
    prob += hub_production <= hubs_df.set_index('Site Reference').at[j, 'Max Capacity (tonnes/year)'] * hub_active[j], f"Hub_activation_constraint_{j}"

# Constraint to meet the total minimum production across all hubs
prob += lpSum([lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor for j in hubs_df['Site Reference']]) >= minimum_total_production, "Min_Total_Production"


# Solve the problem
print("Starting optimization...")
prob.solve()

# Output results
print("\nOptimization Results:")
print(f"Status: {LpStatus[prob.status]}")

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

# Plotting hubs
for idx, row in hubs_df.iterrows():
    production_quantity = sum(transport_vars[i, row['Site Reference']].varValue for i in sources_df['Site Reference']) * conversion_factor
    if row['Source'] == 'main':
        # Main hubs
        icon_color = 'blue'
        icon_type = 'industry'
    else:
        # Additional hubs
        icon_color = 'orange'
        icon_type = 'briefcase'
    folium.Marker(
        [row['X Coordinates'], row['Y Coordinates']],
        popup=(f"Hub: {row['Site Reference']}<br>"
               f"Quantity Used: {production_quantity:.2f} tonnes<br>"
               f"Max Capacity: {row['Max Capacity (tonnes/year)']} tonnes"),
        icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
    ).add_to(map_osm)

# Plotting sources
for idx, row in sources_df.iterrows():
    used_quantity = sum(transport_vars[row['Site Reference'], j].varValue for j in hubs_df['Site Reference'] if (row['Site Reference'], j) in transport_vars)
    if row['Source'] == 'main':
        # Main sources
        icon_color = 'green'
        icon_type = 'leaf'
    else:
        # Additional sources
        icon_color = 'purple'
        icon_type = 'tree'
    folium.Marker(
        [row['X Coordinates'], row['Y Coordinates']],
        popup=(f"Source: {row['Site Reference']}<br>"
               f"Available Quantity: {row['Available Quantity (tonnes/year)']} tonnes<br>"
               f"Quantity Used: {used_quantity:.2f} tonnes"),
        icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
    ).add_to(map_osm)

# Plotting transport lines
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        source = sources_df[sources_df['Site Reference'] == i].iloc[0]
        hub = hubs_df[hubs_df['Site Reference'] == j].iloc[0]
        line_weight = 10
        line = folium.PolyLine(
            locations=[(source['X Coordinates'], source['Y Coordinates']), (hub['X Coordinates'], hub['Y Coordinates'])],
            weight=line_weight, color='red',
            popup=(f"Transport from {i} to {j}: {transport_vars[i, j].varValue:.2f} tonnes")
        ).add_to(map_osm)

map_osm.save("network_map.html")
print("Network map saved to 'network_map.html'.")
# Extension: Generate Detailed Cost Report and Print Summary

# Prepare detailed cost breakdown for each hub
hub_costs = []

for j in hubs_df['Site Reference']:
    # Calculate production quantities and costs for each hub
    production_quantity = sum(transport_vars[i, j].varValue for i in sources_df['Site Reference']) * conversion_factor
    cost_of_heat = production_quantity * hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne
    purchase_cost = sum(transport_vars[i, j].varValue * purchase_prices[i] for i in sources_df['Site Reference'])
    fixed_capex_cost = hub_active[j].varValue * fixed_capex
    variable_capex_cost = production_quantity * variable_capex_per_tonne
    total_cost = cost_of_heat + purchase_cost + fixed_capex_cost + variable_capex_cost
    avg_cost_per_tonne = total_cost / production_quantity if production_quantity > 0 else 0

    hub_costs.append({
        'Hub': j,
        'Total_Production': production_quantity,
        'Cost_of_Heat': cost_of_heat,
        'Purchase_Cost': purchase_cost,
        'Capex_Fixed': fixed_capex_cost,
        'Capex_Variable': variable_capex_cost,
        'Total_Cost': total_cost,
        'Average_Cost_Per_Tonne': avg_cost_per_tonne
    })

# Create DataFrame for hub costs
hub_costs_df = pd.DataFrame(hub_costs)

# Calculate overall totals
overall_totals = {
    'Hub': 'Overall',
    'Total_Production': hub_costs_df['Total_Production'].sum(),
    'Cost_of_Heat': hub_costs_df['Cost_of_Heat'].sum(),
    'Purchase_Cost': hub_costs_df['Purchase_Cost'].sum(),
    'Capex_Fixed': hub_costs_df['Capex_Fixed'].sum(),
    'Capex_Variable': hub_costs_df['Capex_Variable'].sum(),
    'Total_Cost': hub_costs_df['Total_Cost'].sum(),
    'Average_Cost_Per_Tonne': hub_costs_df['Total_Cost'].sum() / hub_costs_df['Total_Production'].sum() if hub_costs_df['Total_Production'].sum() > 0 else 0
}

# Add overall totals to the DataFrame
hub_costs_df = hub_costs_df._append(overall_totals, ignore_index=True)

# Print summary of the results
print("\nDetailed Cost Breakdown:")
print(hub_costs_df.to_string(index=False))

# Save the report to an Excel file
report_path = "hub_cost_report.xlsx"
hub_costs_df.to_excel(report_path, index=False)
print(f"\nDetailed hub cost report saved to '{report_path}'.")
