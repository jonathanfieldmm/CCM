import pandas as pd
from sklearn.cluster import DBSCAN
import folium
import pyproj

# Load the data
data_path = 'Hubs/co2_data.xlsx'
data = pd.read_excel(data_path)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Extract coordinates and emissions
coordinates = data[['Easting', 'Northing']]
emissions = data['Emissions']

# Convert Easting and Northing to Latitude and Longitude
transformer = pyproj.Transformer.from_crs("epsg:27700", "epsg:4326")
data['Latitude'], data['Longitude'] = transformer.transform(data['Easting'].values, data['Northing'].values)

# Define the maximum distance for points to be considered in the same cluster (in meters)
max_distance = 10000  # Adjust as needed

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=max_distance, min_samples=1, metric='euclidean')
data['cluster'] = dbscan.fit_predict(coordinates)

# Calculate total emissions per cluster
clustered_data = data[data['cluster'] != -1]  # Exclude noise points
cluster_totals = clustered_data.groupby('cluster')['Emissions'].sum()

# Filter clusters by the minimum total emissions threshold
min_emissions_threshold = 400000
significant_clusters = cluster_totals[cluster_totals >= min_emissions_threshold].index
filtered_data = clustered_data[clustered_data['cluster'].isin(significant_clusters)]

# Calculate average latitude and longitude for each significant cluster
cluster_centers = filtered_data.groupby('cluster').agg({
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Emissions': 'sum'
}).reset_index()

# Save the cluster data with average coordinates to an Excel file
output_path = 'Hubs/cluster_centers.xlsx'
cluster_centers.to_excel(output_path, index=False)

# Create a map centered around the UK
m = folium.Map(location=[54.0, -2.0], zoom_start=6)

# Define a color palette for the clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

# Add the clustered points to the map
for idx, row in filtered_data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=colors[row['cluster'] % len(colors)],
        fill=True,
        fill_color=colors[row['cluster'] % len(colors)],
        fill_opacity=0.7,
        popup=f"Site: {row['Site']}\nCluster: {row['cluster']}\nEmissions: {row['Emissions']} tons"
    ).add_to(m)

# Save the map to an HTML file
m.save('Hubs/filtered_clustered_map.html')

# Output the filtered clusters with average coordinates
cluster_centers

