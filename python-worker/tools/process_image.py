import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image

def normalize(band):
    """
    Normalizes a raw 16-bit band to 8-bit (0-255) for display.
    """
    # Clip the top 2% of pixels to remove outliers (makes it brighter)
    lower, upper = np.percentile(band, (2, 98))
    band = np.clip(band, lower, upper)
    
    # Scale to 0-255
    band = ((band - lower) / (upper - lower) * 255).astype(np.uint8)
    return band

def combine_bands(red_path, green_path, blue_path, output_path):
    print("Opening bands...")
    
    # Open the files
    with rasterio.open(red_path) as src_r:
        red = src_r.read(1)
        meta = src_r.meta # Keep metadata (coordinates, projection)

    with rasterio.open(green_path) as src_g:
        green = src_g.read(1)

    with rasterio.open(blue_path) as src_b:
        blue = src_b.read(1)

    print("Normalizing colors...")
    # Normalize each band separately
    r_norm = normalize(red)
    g_norm = normalize(green)
    b_norm = normalize(blue)

    # Stack them into a generic RGB array (Height, Width, 3)
    rgb_composite = np.dstack((r_norm, g_norm, b_norm))

    print(f"Saving to {output_path}...")
    
    # Plot using Matplotlib (Easy way to save as PNG)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_composite)
    plt.axis('off') # Hide axes for a clean image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print("Done!")

if __name__ == "__main__":
    # Use '..' to step up from 'python-worker' to 'sat-vis'
    # Then go down into 'sample_images'
    base_dir = "../sample_images" 
    
    # Make sure to match your ACTUAL filenames from the download!
    # (The error says you are looking for Red.tif, ensure it exists)
    red_file = f"{base_dir}/LC08_L2SP_122065_20230906_02_T1_Red.tif" 
    green_file = f"{base_dir}/LC08_L2SP_122065_20230906_02_T1_Green.tif"
    blue_file = f"{base_dir}/LC08_L2SP_122065_20230906_02_T1_Blue.tif"
    
    output_file = "final_jakarta_rgb.png"
    
    combine_bands(red_file, green_file, blue_file, output_file)