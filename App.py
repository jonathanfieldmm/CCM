import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import PolyLineTextPath
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, LpBinary
from streamlit_folium import folium_static
import math
import io
import zipfile

# Function to read uploaded file
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    return None

# Function to calculate distances using the Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of the Earth in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Streamlit UI components
st.title("CCM Optimisation Model for Fertiliser Production and Transportation Cost Minimization")

st.sidebar.header("Upload Data Files")
uploaded_hubs_file = st.sidebar.file_uploader("Upload Hubs Data", type=["xlsx"])
uploaded_sources_file = st.sidebar.file_uploader("Upload Sources Data", type=["xlsx"])

# Options to include additional hubs and sources
include_additional_hubs = st.sidebar.checkbox("Include Additional Hubs (Data Centres)", value=False)
include_additional_sources = st.sidebar.checkbox("Include Additional Sources (Compost)", value=False)

# Load the main data
if uploaded_hubs_file:
    hubs_df = load_data(uploaded_hubs_file)
    hubs_df['Source'] = 'main'
else:
    st.warning("Please upload a valid Hubs Data file.")

if uploaded_sources_file:
    sources_df = load_data(uploaded_sources_file)
    sources_df['Source'] = 'main'
else:
    st.warning("Please upload a valid Sources Data file.")

# Load additional data if selected
if include_additional_hubs:
    uploaded_additional_hubs_file = st.sidebar.file_uploader("Upload Additional Hubs Data", type=["xlsx"])
    if uploaded_additional_hubs_file:
        additional_hubs_df = load_data(uploaded_additional_hubs_file)
        additional_hubs_df['Source'] = 'additional'
        hubs_df = pd.concat([hubs_df, additional_hubs_df], ignore_index=True)

if include_additional_sources:
    uploaded_additional_sources_file = st.sidebar.file_uploader("Upload Additional Sources Data", type=["xlsx"])
    if uploaded_additional_sources_file:
        additional_sources_df = load_data(uploaded_additional_sources_file)
        additional_sources_df['Source'] = 'additional'
        sources_df = pd.concat([sources_df, additional_sources_df], ignore_index=True)

# Ensure both data files are uploaded
if uploaded_hubs_file and uploaded_sources_file:
    st.sidebar.header("Main Inputs")
    minimum_total_production = st.sidebar.number_input("Minimum Total Production (tonnes)", value=300000)
    minimum_hub_production = st.sidebar.number_input("Minimum Production per Hub (tonnes)", value=15000)

    st.sidebar.header("Cost Inputs")
    haulage_cost_per_mile = st.sidebar.number_input("Haulage Cost per Mile (£/km)", value=1.86)
    average_load = st.sidebar.number_input("Average Load (tonnes)", value=20)
    haulage_cost_per_tonne_mile = haulage_cost_per_mile / average_load
    fixed_capex = st.sidebar.number_input("Fixed CAPEX per Hub (£)", value=100000)
    variable_capex_per_tonne = st.sidebar.number_input("Variable CAPEX per Tonne (£/tonne)", value=300)
    cost_dolomite = st.sidebar.number_input("Cost of Dolomite (£)", value=40)
    cost_urea = st.sidebar.number_input("Cost of Urea (£)", value=60)

    st.sidebar.header("Process Inputs")
    conversion_factor = st.sidebar.number_input("Conversion Factor", value=0.7)
    heat_required_per_tonne = st.sidebar.number_input("Heat Required per Tonne (kWh/tonne)", value=1.0)
    dolomite_per_tonne = st.sidebar.number_input("Dolomite Requirement per Tonne (tonnes)", value=0.1)
    urea_per_tonne = st.sidebar.number_input("Urea Requirement per Tonne (tonnes)", value=0.2)

    # Button to run simulation
    if st.sidebar.button("Run Simulation"):
        with st.spinner('Running simulation...'):
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

            # Extract the purchase price of feedstock from the sources data
            purchase_prices = sources_df.set_index('Site Reference')['Purchase Price (£/tonne)']

            # Define costs
            transportation_costs = lpSum([transport_vars[i, j] * distances[(i, j)] * haulage_cost_per_tonne_mile for i, j in transport_vars])
            production_costs = lpSum([
                (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) *
                (hubs_df.set_index('Site Reference').at[j, 'Cost of Heat (£/kWh)'] * heat_required_per_tonne +
                 dolomite_per_tonne * cost_dolomite +
                 urea_per_tonne * cost_urea) +
                lpSum([transport_vars[i, j] * purchase_prices[i] for i in sources_df['Site Reference']])
                for j in hubs_df['Site Reference']
            ])
            capex_costs = lpSum([
                hub_active[j] * fixed_capex +
                (lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor) * variable_capex_per_tonne
                for j in hubs_df['Site Reference']
            ])

            # Objective function
            prob += transportation_costs + production_costs + capex_costs, "Total Costs"

            # Constraints
            for i in sources_df['Site Reference']:
                prob += lpSum([transport_vars[i, j] for j in hubs_df['Site Reference']]) <= sources_df.set_index('Site Reference').at[i, 'Available Quantity (tonnes/year)'], f"Supply_constraint_{i}"

            for j in hubs_df['Site Reference']:
                hub_production = lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor
                prob += hub_production >= hub_active[j] * minimum_hub_production, f"Capacity_lower_constraint_{j}"
                prob += hub_production <= hubs_df.set_index('Site Reference').at[j, 'Max Capacity (tonnes/year)'] * hub_active[j], f"Hub_activation_constraint_{j}"

            prob += lpSum([lpSum([transport_vars[i, j] for i in sources_df['Site Reference']]) * conversion_factor for j in hubs_df['Site Reference']]) >= minimum_total_production, "Min_Total_Production"

            # Solve the problem
            prob.solve()

            # Prepare detailed cost breakdown for each hub
            hub_costs = []
            for j in hubs_df['Site Reference']:
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

            # Transport data
            transport_data = []
            for (i, j) in transport_vars:
                if transport_vars[i, j].varValue > 0:
                    amount_transported = transport_vars[i, j].varValue
                    transport_data.append({"Source": i, "Hub": j, "Amount Transported (tonnes)": amount_transported})

            transport_df = pd.DataFrame(transport_data)

            # Create a downloadable Excel file with all results
            result_output = io.BytesIO()
            with pd.ExcelWriter(result_output, engine='xlsxwriter') as writer:
                transport_df.to_excel(writer, index=False, sheet_name='Transport Data')
                hub_costs_df.to_excel(writer, index=False, sheet_name='Cost Breakdown')

            # Visualization with Folium
            map_osm = folium.Map(location=[55, -3], zoom_start=6)
            for idx, row in hubs_df.iterrows():
                production_quantity = sum(transport_vars[i, row['Site Reference']].varValue for i in sources_df['Site Reference']) * conversion_factor
                icon_color = 'blue' if row['Source'] == 'main' else 'orange'
                folium.Marker(
                    [row['X Coordinates'], row['Y Coordinates']],
                    popup=(f"Hub: {row['Site Reference']}<br>"
                           f"Quantity Used: {production_quantity:.2f} tonnes<br>"
                           f"Max Capacity: {row['Max Capacity (tonnes/year)']} tonnes"),
                    icon=folium.Icon(color=icon_color, icon='industry', prefix='fa')
                ).add_to(map_osm)

            for idx, row in sources_df.iterrows():
                used_quantity = sum(transport_vars[row['Site Reference'], j].varValue for j in hubs_df['Site Reference'] if (row['Site Reference'], j) in transport_vars)
                icon_color = 'green' if row['Source'] == 'main' else 'purple'
                folium.Marker(
                    [row['X Coordinates'], row['Y Coordinates']],
                    popup=(f"Source: {row['Site Reference']}<br>"
                           f"Available Quantity: {row['Available Quantity (tonnes/year)']} tonnes<br>"
                           f"Quantity Used: {used_quantity:.2f} tonnes"),
                    icon=folium.Icon(color=icon_color, icon='leaf', prefix='fa')
                ).add_to(map_osm)

            for (i, j) in transport_vars:
                if transport_vars[i, j].varValue > 0:
                    source = sources_df[sources_df['Site Reference'] == i].iloc[0]
                    hub = hubs_df[hubs_df['Site Reference'] == j].iloc[0]
                    line = folium.PolyLine(
                        locations=[(source['X Coordinates'], source['Y Coordinates']), (hub['X Coordinates'], hub['Y Coordinates'])],
                        weight=10, color='red',
                        popup=(f"Transport from {i} to {j}: {transport_vars[i, j].varValue:.2f} tonnes")
                    ).add_to(map_osm)

            # Save map to HTML
            map_html = io.BytesIO()
            map_osm.save(map_html, close_file=False)

            # Create a zip file containing both the Excel and map files
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("ccm_optimization_results.xlsx", result_output.getvalue())
                zip_file.writestr("ccm_optimization_map.html", map_html.getvalue())

            # Provide the zip file as a downloadable result
            st.download_button(
                label="Download Results",
                data=zip_buffer.getvalue(),
                file_name="ccm_optimization_results.zip",
                mime="application/zip"
            )

            folium_static(map_osm)

else:
    st.stop()
