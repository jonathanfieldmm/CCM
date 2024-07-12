import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import PolyLineTextPath
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpBinary
from streamlit_folium import folium_static
import math
import io

# Sample data for hubs and sources (replace with your actual data)
hubs_data = {
    'Site Reference': ['Hub01', 'Hub02', 'Hub03', 'Hub04', 'Hub05', 'Hub06', 'Hub07', 'Hub08', 'Hub09', 'Hub10'],
    'X Coordinates': [51.5074, 53.4808, 55.9533, 52.513, 53.4084, 52.6309, 54.9783, 53.8008, 50.3755, 55.8642],
    'Y Coordinates': [-0.1278, -2.2426, -3.1883, -1.8904, -2.9916, 1.2974, -1.6174, -1.5491, -4.1427, -4.2518],
    'Heat Available (kWh/year)': [500000, 750000, 600000, 450000, 800000, 700000, 650000, 550000, 720000, 600000],
    'Cost of Heat (£/kWh)': [0.05, 0.04, 0.06, 0.05, 0.04, 0.06, 0.07, 0.05, 0.04, 0.06],
    'Max Capacity (tonnes/year)': [10000, 15000, 12000, 9000, 16000, 14000, 13000, 11000, 15000, 12000]
}

sources_data = {
    'Site Reference': ['FS01', 'FS02', 'FS03', 'FS04', 'FS05', 'FS06', 'FS07', 'FS08', 'FS09', 'FS10'],
    'X Coordinates': [52.2053, 51.5095, 53.7893, 51.4543, 50.8225, 51.4816, 53.7632, 52.4862, 50.7192, 55.3781],
    'Y Coordinates': [0.1218, -0.0959, -2.2425, -2.5879, -0.1373, -0.4725, -2.7034, -1.8904, -1.8808, -3.4360],
    'Available Quantity (tonnes/year)': [20000, 25000, 18000, 30000, 15000, 22000, 28000, 27000, 23000, 21000],
    'Water Content (%)': [30, 25, 28, 22, 35, 32, 27, 29, 31, 26],
    'Purchase Price (£/tonne)': [45, 50, 42, 48, 40, 47, 44, 49, 41, 46]
}

# Streamlit UI components
st.title("Optimization Model for Production and Transportation Cost Minimization")

# Split sidebar into sections
st.sidebar.header("Main Inputs")
minimum_total_production = st.sidebar.number_input("Minimum Total Production (tonnes)", value=120000)

st.sidebar.header("Cost Inputs")
haulage_cost_per_tonne_mile = st.sidebar.number_input("Haulage Cost per Tonne Mile (£)", value=0.02)
generic_capex = st.sidebar.number_input("Generic CAPEX (£)", value=100000)
cost_co2 = st.sidebar.number_input("Cost of CO2 (£/tonne)", value=50)
cost_ammonia = st.sidebar.number_input("Cost of Ammonia (£/tonne)", value=100)
cost_phosphorus = st.sidebar.number_input("Cost of Phosphorus (£/tonne)", value=80)
cost_nitrogen = st.sidebar.number_input("Cost of Nitrogen (£/tonne)", value=70)

st.sidebar.header("Process Inputs")
conversion_factor = st.sidebar.number_input("Conversion Factor", value=0.7)
heat_required_per_tonne = st.sidebar.number_input("Heat Required per Tonne (kWh/tonne)", value=1.0)
co2_per_tonne = st.sidebar.number_input("CO2 Requirement per Tonne (tonnes)", value=0.2)
ammonia_per_tonne = st.sidebar.number_input("Ammonia Requirement per Tonne (tonnes)", value=0.1)
phosphorus_per_tonne = st.sidebar.number_input("Phosphorus Requirement per Tonne (tonnes)", value=0.05)
nitrogen_per_tonne = st.sidebar.number_input("Nitrogen Requirement per Tonne (tonnes)", value=0.15)

# Define the Haversine formula to calculate distances
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Calculate distances between all sources and hubs
distances = {}
for _, source in pd.DataFrame(sources_data).iterrows():
    for _, hub in pd.DataFrame(hubs_data).iterrows():
        key = (source['Site Reference'], hub['Site Reference'])
        distances[key] = haversine(source['Y Coordinates'], source['X Coordinates'], hub['Y Coordinates'], hub['X Coordinates'])

# Optimization problem setup
prob = LpProblem("Minimize_Costs", LpMinimize)

# Decision variables
transport_vars = LpVariable.dicts("Transport", [(i, j) for i in pd.DataFrame(sources_data)['Site Reference'] for j in pd.DataFrame(hubs_data)['Site Reference']], lowBound=0, cat='Continuous')
hub_active = LpVariable.dicts("HubActive", pd.DataFrame(hubs_data)['Site Reference'], cat='Binary')

# Define costs
transportation_costs = lpSum([transport_vars[i, j] * distances[(i, j)] * haulage_cost_per_tonne_mile for i, j in transport_vars])
production_costs = lpSum([
    (lpSum([transport_vars[i, j] for i in pd.DataFrame(sources_data)['Site Reference']]) * conversion_factor) *
    (pd.DataFrame(hubs_data).set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne +
     co2_per_tonne * cost_co2 +
     ammonia_per_tonne * cost_ammonia +
     phosphorus_per_tonne * cost_phosphorus +
     nitrogen_per_tonne * cost_nitrogen)
    for j in pd.DataFrame(hubs_data)['Site Reference']
])
capex_costs = lpSum([hub_active[j] * generic_capex for j in pd.DataFrame(hubs_data)['Site Reference']])

# Objective function
prob += transportation_costs + production_costs + capex_costs, "Total Costs"

# Constraints
for i in pd.DataFrame(sources_data)['Site Reference']:
    prob += lpSum([transport_vars[i, j] for j in pd.DataFrame(hubs_data)['Site Reference']]) <= pd.DataFrame(sources_data).set_index('Site Reference').at[i, 'Available Quantity (tonnes/year)'], f"Supply_constraint_{i}"
for j in pd.DataFrame(hubs_data)['Site Reference']:
    prob += (lpSum([transport_vars[i, j] for i in pd.DataFrame(sources_data)['Site Reference']]) * conversion_factor) <= pd.DataFrame(hubs_data).set_index('Site Reference').at[j, 'Max Capacity (tonnes/year)'], f"Demand_constraint_{j}"
for j in pd.DataFrame(hubs_data)['Site Reference']:
    prob += lpSum([transport_vars[i, j] for i in pd.DataFrame(sources_data)['Site Reference']]) <= hub_active[j] * sum(pd.DataFrame(sources_data)['Available Quantity (tonnes/year)']), f"Activation_constraint_{j}"
prob += lpSum([lpSum([transport_vars[i, j] for i in pd.DataFrame(sources_data)['Site Reference']]) * conversion_factor for j in pd.DataFrame(hubs_data)['Site Reference']]) >= minimum_total_production, "Min_Total_Production"

# Solve the problem
prob.solve()

# Calculate total costs
total_transportation_cost = sum(transport_vars[i, j].varValue * distances[(i, j)] * haulage_cost_per_tonne_mile for i, j in transport_vars)
total_production_cost = sum(
    (sum(transport_vars[i, j].varValue for i in pd.DataFrame(sources_data)['Site Reference']) * conversion_factor) *
    (pd.DataFrame(hubs_data).set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne +
     co2_per_tonne * cost_co2 +
     ammonia_per_tonne * cost_ammonia +
     phosphorus_per_tonne * cost_phosphorus +
     nitrogen_per_tonne * cost_nitrogen)
    for j in pd.DataFrame(hubs_data)['Site Reference']
)
total_capex = sum(hub_active[j].varValue * generic_capex for j in pd.DataFrame(hubs_data)['Site Reference'])
total_cost = total_transportation_cost + total_production_cost + total_capex
total_production_quantity = sum(sum(transport_vars[i, j].varValue for i in pd.DataFrame(sources_data)['Site Reference']) * conversion_factor for j in pd.DataFrame(hubs_data)['Site Reference'])
cost_per_tonne_produced = total_cost / total_production_quantity

# Display the results
st.write("## Optimization Results")
st.write(f"**Total transportation cost:** £{total_transportation_cost:.2f}")
st.write(f"**Total production cost:** £{total_production_cost:.2f}")
st.write(f"**Total CAPEX:** £{total_capex:.2f}")
st.write(f"**Total cost:** £{total_cost:.2f}")
st.write(f"**Total production quantity:** {total_production_quantity:.2f} tonnes")
st.write(f"**Cost per tonne produced:** £{cost_per_tonne_produced:.2f}")

# Debug output to verify the results of the optimization
st.write("### Production quantities at each hub:")
for j in pd.DataFrame(hubs_data)['Site Reference']:
    production_quantity = sum(transport_vars[i, j].varValue for i in pd.DataFrame(sources_data)['Site Reference']) * conversion_factor
    st.write(f"Hub {j}: {production_quantity:.2f} tonnes")

st.write("### Amount of feedstock transported between each source and hub:")
transport_data = []
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        amount_transported = transport_vars[i, j].varValue
        st.write(f"Transport from Source {i} to Hub {j}: {amount_transported:.2f} tonnes")
        transport_data.append({"Source": i, "Hub": j, "Amount Transported (tonnes)": amount_transported})

# Create a DataFrame for the transport data
transport_df = pd.DataFrame(transport_data)

# Create a downloadable Excel file
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    transport_df.to_excel(writer, index=False, sheet_name='Transport Data')

# Create a download button
st.download_button(
    label="Download Transport Data",
    data=output.getvalue(),
    file_name="transport_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Visualization with Folium
map_osm = folium.Map(location=[55, -3], zoom_start=6)
max_transport = max(transport_vars[i, j].varValue for i, j in transport_vars)
for idx, row in pd.DataFrame(hubs_data).iterrows():
    production_quantity = sum(transport_vars[i, row['Site Reference']].varValue for i in pd.DataFrame(sources_data)['Site Reference']) * conversion_factor
    folium.Marker([row['X Coordinates'], row['Y Coordinates']],
                  popup=(f"Hub: {row['Site Reference']}<br>"
                         f"Quantity Used: {production_quantity:.2f} tonnes<br>"
                         f"Max Capacity: {row['Max Capacity (tonnes/year)']} tonnes"),
                  icon=folium.Icon(color='blue', icon='industry', prefix='fa')).add_to(map_osm)
for idx, row in pd.DataFrame(sources_data).iterrows():
    used_quantity = sum(transport_vars[row['Site Reference'], j].varValue for j in pd.DataFrame(hubs_data)['Site Reference'] if (row['Site Reference'], j) in transport_vars)
    folium.Marker([row['X Coordinates'], row['Y Coordinates']],
                  popup=(f"Source: {row['Site Reference']}<br>"
                         f"Available Quantity: {row['Available Quantity (tonnes/year)']} tonnes<br>"
                         f"Quantity Used: {used_quantity:.2f} tonnes"),
                  icon=folium.Icon(color='green', icon='leaf', prefix='fa')).add_to(map_osm)
for (i, j) in transport_vars:
    if transport_vars[i, j].varValue > 0:
        source = pd.DataFrame(sources_data)[pd.DataFrame(sources_data)['Site Reference'] == i].iloc[0]
        hub = pd.DataFrame(hubs_data)[pd.DataFrame(hubs_data)['Site Reference'] == j].iloc[0]
        line_weight = 10
        line = folium.PolyLine(locations=[(source['X Coordinates'], source['Y Coordinates']), (hub['X Coordinates'], hub['Y Coordinates'])],
                               weight=line_weight, color='red',
                               popup=(f"Transport from {i} to {j}: {transport_vars[i, j].varValue:.2f} tonnes")).add_to(map_osm)
folium_static(map_osm)
