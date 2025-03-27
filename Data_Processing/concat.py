import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm


DEGREES_PER_PIXEL_LAT = 2.0357e-6  
DEGREES_PER_PIXEL_LON = 2.680e-6  


BOX_SIZE = 512


HALF_BOX_LAT = (BOX_SIZE / 2) * DEGREES_PER_PIXEL_LAT
HALF_BOX_LON = (BOX_SIZE / 2) * DEGREES_PER_PIXEL_LON

streetview_folder = 'streetview'
overhead_folder = 'overhead'
height_folder = 'labels/height'
output_folder = 'concat_data_subset'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


json_file = os.path.join(streetview_folder, 'intersecting_bboxes.json')
with open(json_file, 'r') as f:
    data = json.load(f)


images_file = os.path.join(streetview_folder, 'images.txt')
with open(images_file, 'r') as f:
    images_lines = f.readlines()


coverage_ratios = []
longitudes = []
latitudes = []


for idx, info in tqdm(data.items(), desc="Processing Data", total=len(data)):
    idx_int = int(idx) - 1  
    center_lat = info['center_lat']
    center_lon = info['center_lon']
    intersecting_bboxes = info['intersecting_bboxes']

    lat_min = center_lat - HALF_BOX_LAT
    lat_max = center_lat + HALF_BOX_LAT
    lon_min = center_lon - HALF_BOX_LON
    lon_max = center_lon + HALF_BOX_LON


    final_image = Image.new('RGB', (BOX_SIZE, BOX_SIZE), (0, 0, 0))


    height_map = np.full((BOX_SIZE, BOX_SIZE), 0)


    coverage_mask = Image.new('1', (BOX_SIZE, BOX_SIZE), 0)  # 1-bit pixels, black and white

    height_paths = []
    overhead_paths = []


    for bbox in intersecting_bboxes:
        image_rel_path = bbox['image_path']
        image_path = os.path.join(overhead_folder, 'images', image_rel_path)
        height_path = os.path.join(height_folder, image_rel_path.replace('.jpg', '.npz'))

        height_paths.append(height_path)
        overhead_paths.append(image_path)


        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue


        if not os.path.exists(height_path):
            print(f"Height file not found: {height_path}")
            continue


        sat_image = Image.open(image_path)


        try:
            height_data = np.load(height_path)['data']
            height_data = height_data.reshape(256, 256)
        except ValueError:
            # Skip this iteration if reshaping fails
            continue

        sat_lat_min = bbox['lat_min']
        sat_lat_max = bbox['lat_max']
        sat_lon_min = bbox['lon_min']
        sat_lon_max = bbox['lon_max']


        overlap_lat_min = max(lat_min, sat_lat_min)
        overlap_lat_max = min(lat_max, sat_lat_max)
        overlap_lon_min = max(lon_min, sat_lon_min)
        overlap_lon_max = min(lon_max, sat_lon_max)


        if overlap_lat_max <= overlap_lat_min or overlap_lon_max <= overlap_lon_min:
            continue  # No overlap, skip


        x_start_sat = int(round((overlap_lon_min - sat_lon_min) / DEGREES_PER_PIXEL_LON))
        x_end_sat = int(round((overlap_lon_max - sat_lon_min) / DEGREES_PER_PIXEL_LON))
        y_start_sat = int(round((sat_lat_max - overlap_lat_max) / DEGREES_PER_PIXEL_LAT))
        y_end_sat = int(round((sat_lat_max - overlap_lat_min) / DEGREES_PER_PIXEL_LAT))


        x_start_sat = max(0, x_start_sat)
        y_start_sat = max(0, y_start_sat)
        x_end_sat = min(sat_image.width, x_end_sat)
        y_end_sat = min(sat_image.height, y_end_sat)


        x_start_dst = int(round((overlap_lon_min - lon_min) / DEGREES_PER_PIXEL_LON))
        x_end_dst = int(round((overlap_lon_max - lon_min) / DEGREES_PER_PIXEL_LON))
        y_start_dst = int(round((lat_max - overlap_lat_max) / DEGREES_PER_PIXEL_LAT))
        y_end_dst = int(round((lat_max - overlap_lat_min) / DEGREES_PER_PIXEL_LAT))


        x_start_dst = max(0, x_start_dst)
        y_start_dst = max(0, y_start_dst)
        x_end_dst = min(final_image.width, x_end_dst)
        y_end_dst = min(final_image.height, y_end_dst)


        sat_crop_width = x_end_sat - x_start_sat
        sat_crop_height = y_end_sat - y_start_sat
        dst_crop_width = x_end_dst - x_start_dst
        dst_crop_height = y_end_dst - y_start_dst


        crop_width = min(sat_crop_width, dst_crop_width)
        crop_height = min(sat_crop_height, dst_crop_height)

        if crop_width <= 0 or crop_height <= 0:
            continue  


        sat_crop = sat_image.crop((x_start_sat, y_start_sat, x_start_sat + crop_width, y_start_sat + crop_height))


        height_crop = height_data[y_start_sat:y_start_sat + crop_height, x_start_sat:x_start_sat + crop_width]


        final_image.paste(sat_crop, (x_start_dst, y_start_dst))


        height_map[y_start_dst:y_start_dst + crop_height, x_start_dst:x_start_dst + crop_width] = height_crop


        mask_region = Image.new('1', (crop_width, crop_height), 1)
        coverage_mask.paste(mask_region, (x_start_dst, y_start_dst))


    covered_pixels = np.array(coverage_mask).sum()
    total_pixels = BOX_SIZE * BOX_SIZE
    coverage_ratio = covered_pixels / total_pixels
    coverage_ratios.append(coverage_ratio)
    longitudes.append(center_lon)
    latitudes.append(center_lat)


    folder_name = f"{center_lon}_{center_lat}"
    folder_path = os.path.join(output_folder, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    output_image_path = os.path.join(folder_path, 'overhead.jpg')
    final_image.save(output_image_path)


    output_height_path = os.path.join(folder_path, 'height_map.npy')
    np.save(output_height_path, height_map)


    streetview_image_rel_path = images_lines[idx_int].strip()
    streetview_image_path = os.path.join(streetview_folder, 'images', streetview_image_rel_path)
    output_streetview_image_path = os.path.join(folder_path, 'streetview.jpg')


    info = {
        'latitude': center_lat,
        'longitude': center_lon,
        'image_size': [BOX_SIZE, BOX_SIZE],
        'coverage_ratio': coverage_ratio,
        'streetview_image_path' : streetview_image_path,
        'overhead_path' : overhead_paths,
        'height_path' : height_paths
    }
    info_json_path = os.path.join(folder_path, 'info.json')
    with open(info_json_path, 'w') as f:
        json.dump(info, f, indent=4)

    if os.path.exists(streetview_image_path):
        streetview_image = Image.open(streetview_image_path)
        streetview_image.save(output_streetview_image_path)
    else:
        print(f"Streetview image not found: {streetview_image_path}")

    # print(f"Saved images, height map, and info to: {folder_path}")


plt.figure(figsize=(8, 6))
sns.kdeplot(coverage_ratios, shade=True)
plt.xlabel('Coverage Ratio')
plt.ylabel('Density')
plt.title('Kernel Density Estimation of Coverage Ratios')
kde_path = os.path.join(output_folder, 'coverage_ratio_kde.png')
plt.savefig(kde_path)
plt.close()
print(f"KDE plot saved to: {kde_path}")


plt.figure(figsize=(8, 6))
plt.scatter(longitudes, latitudes, c=coverage_ratios, cmap='viridis', s=50)
plt.colorbar(label='Coverage Ratio')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Coverage Ratio Distribution')
heatmap_path = os.path.join(output_folder, 'coverage_ratio_distribution.png')
plt.savefig(heatmap_path)
plt.close()
print(f"Coverage ratio distribution plot saved to: {heatmap_path}")
