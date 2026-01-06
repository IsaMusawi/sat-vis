import pystac_client
import planetary_computer
import requests
import os

# CONFIGURATION
MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def search_images(bbox, date_range, max_clouds):
    """
    Searches the Planetary Computer for Landsat 8/9 images 
    meeting specific criteria.
    """
    print(f"1. Searching Landsat (Cloud Cover < {max_clouds}%)...")
    
    catalog = pystac_client.Client.open(
        MPC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_clouds}},
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
        max_items=1
    )
    
    items = list(search.items())
    if not items:
        return None
    
    return items[0]

def download_band(url, filename):
    """
    Helper function to download a single file from a URL.
    """
    print(f"   Downloading: {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("   -> Success.")
    except Exception as e:
        print(f"   -> Failed: {e}")

def main():
    # 1. DEFINE PARAMETERS
    # Jakarta Area
    bbox = [106.70, -6.30, 106.90, -6.10]
    
    # Dry Season (August - October) to ensure clear skies
    date_range = "2023-08-01/2023-10-31" 
    max_clouds = 30 # Percentage

    # 2. SEARCH
    item = search_images(bbox, date_range, max_clouds)
    
    if not item:
        print("No images found matching criteria.")
        return

    print(f"2. Selected Image: {item.id}")
    print(f"   Date: {item.datetime}")
    print(f"   Cloud Cover: {item.properties['eo:cloud_cover']}%")

    # 3. DOWNLOAD BANDS (R, G, B)
    # Mapping: readable_name -> asset_key
    bands_to_download = {
        "Red": "red",
        "Green": "green",
        "Blue": "blue"
    }

    print("3. Starting Download...")
    for name, key in bands_to_download.items():
        if key in item.assets:
            url = item.assets[key].href
            filename = f"{item.id}_{name}.tif"
            download_band(url, filename)
        else:
            print(f"   Warning: Band '{name}' not found in assets.")

if __name__ == "__main__":
    main()