import xml.etree.ElementTree as ET
import pandas as pd
from geopy.geocoders import Nominatim
import time

def parse_kml(kml_file_path):
    tree = ET.parse(kml_file_path)
    root = tree.getroot()

    # Namespace dictionary to handle namespaces in the KML file
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    sites_data = []
    for placemark in root.findall('.//kml:Placemark', ns):
        name = placemark.find('.//kml:name', ns).text if placemark.find('.//kml:name', ns) is not None else "Unknown"
        postcode = placemark.find(".//kml:ExtendedData/kml:Data[@name='Postcode']/kml:value", ns)
        if postcode is not None:
            postcode = postcode.text
            sites_data.append({'Name': name, 'Postcode': postcode})

    return pd.DataFrame(sites_data)

def get_lat_long(df):
    geolocator = Nominatim(user_agent="geoapiExercises")
    latitudes = []
    longitudes = []

    for postcode in df['Postcode']:
        try:
            location = geolocator.geocode(postcode)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except Exception as e:
            print(f"Error: {e}")
            latitudes.append(None)
            longitudes.append(None)
        time.sleep(1)  # to avoid hitting the rate limit
    df['Latitude'] = latitudes
    df['Longitude'] = longitudes
    return df

# Path to the KML file
kml_file_path = 'Map of UK compost sites.kml'

# Parse the KML file to get data
df_sites = parse_kml(kml_file_path)

# Get coordinates from postcodes
df_with_coords = get_lat_long(df_sites)

# Save the dataframe with coordinates
df_with_coords.to_csv('compost_sites_with_coordinates.csv', index=False)
print(df_with_coords)


