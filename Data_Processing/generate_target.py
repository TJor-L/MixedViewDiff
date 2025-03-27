import os
import json
import random
import shutil
from tqdm import tqdm

DEGREES_PER_PIXEL_LAT = 2.680e-6
DEGREES_PER_PIXEL_LON = 2.0357e-6
IMAGE_SIZE = 512

def get_lat_lon_from_folder(folder_name):
    lat, lon = map(float, folder_name.split('_'))
    return lat, lon

def is_within_target_box(target_lat, target_lon, lat, lon):
    lat_min = target_lat - (IMAGE_SIZE / 2) * DEGREES_PER_PIXEL_LAT
    lat_max = target_lat + (IMAGE_SIZE / 2) * DEGREES_PER_PIXEL_LAT
    lon_min = target_lon - (IMAGE_SIZE / 2) * DEGREES_PER_PIXEL_LON
    lon_max = target_lon + (IMAGE_SIZE / 2) * DEGREES_PER_PIXEL_LON
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def main(base_folder, output_folder):
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    
    # Step 1: Filter folders by coverage ratio
    valid_folders = []
    folder_info = {}
    for folder in folders:
        info_path = os.path.join(base_folder, folder, "info.json")
        with open(info_path, 'r') as f:
            info = json.load(f)
        if info["coverage_ratio"] >= 0.9:
            valid_folders.append(folder)
            folder_info[folder] = info

    # Step 2: Randomly select 20% of folders as target
    random.seed(0)  # Set random seed
    target_folders = random.sample(valid_folders, int(0.2 * len(valid_folders)))
    target_folder_path = os.path.join(output_folder, "target")
    near_folder_path = os.path.join(output_folder, "near")
    os.makedirs(target_folder_path, exist_ok=True)
    os.makedirs(near_folder_path, exist_ok=True)

    # Step 3: Find nearby folders
    for target in tqdm(target_folders, desc="Processing target folders"):
        target_lat, target_lon = get_lat_lon_from_folder(target)
        nearby_folders = []

        # Create subfolder for the target in `near` folder
        target_subfolder = os.path.join(near_folder_path, target)
        os.makedirs(target_subfolder, exist_ok=True)
        near_streetviews_folder = os.path.join(target_subfolder, "near_streetviews")
        near_heights_folder = os.path.join(target_subfolder, "near_heights")
        os.makedirs(near_streetviews_folder, exist_ok=True)
        os.makedirs(near_heights_folder, exist_ok=True)

        target_folders_set = set(target_folders)

        # Collect all nearby folders
        for folder in valid_folders:
            if folder not in target_folders_set:  # Exclude all target folders
                lat, lon = get_lat_lon_from_folder(folder)
                if is_within_target_box(target_lat, target_lon, lat, lon):
                    nearby_folders.append(folder)

        # Skip processing if fewer than 3 nearby folders are found
        if len(nearby_folders) < 3:
            shutil.rmtree(target_subfolder)  # Remove incomplete subfolder
            continue
        # print(len(nearby_folders))
        # Randomly keep only 3 nearby folders
        nearby_folders = random.sample(nearby_folders, 3)

        # Copy target streetview image to target folder
        target_streetview_path = os.path.join(base_folder, target, "streetview.jpg")
        target_streetview_new_name = f"{target_lat}_{target_lon}.jpg"
        shutil.copy(target_streetview_path, os.path.join(target_folder_path, target_streetview_new_name))

        # Copy target structure (height) and overhead image to its subfolder
        target_structure_path = os.path.join(base_folder, target, "height_map.npy")
        target_overhead_path = os.path.join(base_folder, target, "overhead.jpg")
        shutil.copy(target_structure_path, os.path.join(target_subfolder, "target_height.npy"))
        shutil.copy(target_overhead_path, os.path.join(target_subfolder, "overhead.jpg"))

        # Prepare JSON file for the target
        near_json = {
            "target_latitude": target_lat,
            "target_longitude": target_lon,
            "nearby_positions": []
        }

        for near_folder in nearby_folders:
            lat, lon = get_lat_lon_from_folder(near_folder)

            # Copy nearby streetview image to near_streetviews folder
            near_streetview_path = os.path.join(base_folder, near_folder, "streetview.jpg")
            near_streetview_new_name = f"{lat}_{lon}.jpg"
            shutil.copy(near_streetview_path, os.path.join(near_streetviews_folder, near_streetview_new_name))

            # Copy nearby structure (height) to near_heights folder
            near_structure_path = os.path.join(base_folder, near_folder, "height_map.npy")
            near_structure_new_name = f"{lat}_{lon}.npy"
            shutil.copy(near_structure_path, os.path.join(near_heights_folder, near_structure_new_name))
            
            # Add nearby folder information to JSON
            near_json["nearby_positions"].append({"latitude": lat, "longitude": lon})

        # Save the JSON file in the target subfolder
        target_json_path = os.path.join(target_subfolder, "near_info.json")
        with open(target_json_path, 'w') as f:
            json.dump(near_json, f, indent=4)

if __name__ == "__main__":
    base_folder = "/project/cigserver5/export1/david.w/MixViewDiff/brooklyn/concat_data"  # Replace with your dataset path
    output_folder = "/project/cigserver5/export1/david.w/MixViewDiff/brooklyn/concat_data_target_flip"  # Replace with your output path
    os.makedirs(output_folder, exist_ok=True)
    main(base_folder, output_folder)
