import os
import requests
from pystac_client import Client
import json

# CONFIGURATION
# Best practice: Load these from environment variables or a .env file
USERNAME = os.getenv("CDSE_EMAIL", "m.isa.almusawi@gmail.com")
PASSWORD = os.getenv("CDSE_PASSWORD", "Va!zaado154292")

# API ENDPOINTS
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"

def get_access_token(username, password):
    """Generates a temporary Bearer token from Copernicus Identity."""
    payload = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    response = requests.post(AUTH_URL, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]

# def download_asset(href, token, filename):
#     """Streams the file download using the access token."""
#     headers = {"Authorization": f"Bearer {token}"}
    
#     print(f"Downloading {filename}...")
#     with requests.get(href, headers=headers, stream=True) as r:
#         r.raise_for_status()
#         with open(filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     print(f"Saved: {filename}")

# def main():
#     # 1. Authenticate
#     print("Authenticating...")
#     token = get_access_token(USERNAME, PASSWORD)
    
#     # 2. Search the Catalog (STAC)
#     print("Searching Catalog...")
#     client = Client.open(STAC_URL)
    
#     # Jakarta Coordinates (BBox): [min_lon, min_lat, max_lon, max_lat]
#     bbox = [106.70, -6.30, 106.90, -6.10] 
    
#     search = client.search(
#         collections=["SENTINEL-2"],
#         bbox=bbox,
#         datetime="2023-01-01/2024-12-30",
#         query={"eo:cloud_cover": {"lt": 100}},
#         max_items=5,
#         sortby=[{"field": "eo:cloud_cover", "direction": "asc"}] # <--- CHANGED THIS
#     )

#     items = list(search.items())
#     if not items:
#         print("No images found matching criteria.")
#         return

#     item = items[0]
#     print(f"Found Image: {item.id} (Date: {item.datetime})")

#     # 3. Select the Asset
#     # Sentinel-2 has many 'assets' (bands). 
#     # 'visual' is the pre-generated True Color (RGB) image.
#     # 'B04' would be Red, 'B08' NIR, etc.
#     if "visual" in item.assets:
#         asset_url = item.assets["visual"].href
#         output_file = f"{item.id}_visual.tif"
#         download_asset(asset_url, token, output_file)
#     else:
#         print("Visual asset not found in this item.")

def search_images(token):
    print("2. Searching Catalog (Direct HTTP POST)...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 2. DEFINE SEARCH PARAMETERS
    # Jakarta Coordinates
    bbox = [106.70, -6.30, 106.90, -6.10] 

    # SEARCH
    # Collection: 'landsat-c2-l2' (Collection 2, Level 2 - Analysis Ready)
    print("Searching for Landsat 8/9 images...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        # TARGET THE DRY SEASON (August) to avoid clouds!
        datetime="2023-06-01/2023-10-30", 
        query={"eo:cloud_cover": {"lt": 50}}, # Allow up to 50% clouds
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}], # Get the clearest one
        max_items=1
    )

    response = requests.post(SEARCH_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"!!! Search Failed: {response.status_code}")
        print(response.text)
        exit(1)

    data = response.json()
    features = data.get("features", [])
    print(f"-> Found {len(features)} images.")
    
    if not features:
        return None
        
    return features[0]

def download_image(token, item):
    print(f"3. Processing Image: {item['id']}")
    
    # Try to find the Visual asset (TCI)
    assets = item["assets"]
    if "visual" in assets:
        download_url = assets["visual"]["href"]
        filename = f"{item['id']}_visual.tif"
    elif "B04" in assets: # Fallback to Red band if visual is missing
        print("Visual not found, downloading Red Band (B04)...")
        download_url = assets["B04"]["href"]
        filename = f"{item['id']}_B04.tif"
    else:
        print("No suitable download asset found.")
        print("Available assets:", list(assets.keys()))
        return

    print(f"-> Downloading to {filename}...")
    headers = {"Authorization": f"Bearer {token}"}
    
    with requests.get(download_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("-> Download Complete!")

def main():
    if not USERNAME or not PASSWORD:
        print("Error: Environment variables CDSE_EMAIL or CDSE_PASSWORD are not set.")
        return

    token = get_access_token(USERNAME, PASSWORD)
    image_item = search_images(token)
    
    if image_item:
        download_image(token, image_item)
    else:
        print("No images found in that date range.")

if __name__ == "__main__":
    main()