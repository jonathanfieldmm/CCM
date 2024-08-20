import pandas as pd
from geopy.geocoders import Nominatim
import time

# Load the Excel file and adjust header and skip rows
file_path = 'Digestate/Anaerobic-Digestion-Deployment-Sorted.xlsx'
df = pd.read_excel(file_path, header=3)  # Assuming row 2 contains the header

# Initialize the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

# Function to get latitude and longitude
def get_lat_long(postcode):
    try:
        location = geolocator.geocode(postcode)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Create lists to store latitude and longitude
latitudes = []
longitudes = []

# Iterate over the postcodes and get the coordinates
for postcode in df['Postcode']:  # Now directly using 'Postcode' from the header
    lat, long = get_lat_long(postcode)
    latitudes.append(lat)
    longitudes.append(long)
    time.sleep(1)  # to avoid hitting the rate limit of the geocoding service

# Add latitude and longitude to the dataframe
df['Latitude'] = latitudes
df['Longitude'] = longitudes

# Save the new dataframe to a new Excel file
output_file_path = 'Digestate/Anaerobic-Digestion-Deployment-With-Lat-Long.xlsx'
df.to_excel(output_file_path, index=False)

print(f"File saved to {output_file_path}")

