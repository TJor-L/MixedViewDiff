import os
import json
from tqdm import tqdm


DEGREES_PER_PIXEL_LAT = 2.0357e-6  
DEGREES_PER_PIXEL_LON = 2.680e-6   


BOX_SIZE = 512


HALF_BOX_LAT = (BOX_SIZE / 2) * DEGREES_PER_PIXEL_LAT
HALF_BOX_LON = (BOX_SIZE / 2) * DEGREES_PER_PIXEL_LON


MAX_LOCATIONS = -1 


streetview_folder = 'streetview'
locations_file = os.path.join(streetview_folder, 'locations.txt')
with open(locations_file, 'r') as f:
    lines = f.readlines()
    if MAX_LOCATIONS > 0:
        lines = lines[:MAX_LOCATIONS]
    locations = [line.strip().split(',') for line in lines]
    locations = [(float(lat), float(lon)) for lat, lon in locations]


overhead_folder = 'overhead'
bboxes_file = os.path.join(overhead_folder, 'bboxes.txt')
with open(bboxes_file, 'r') as f:
    bboxes = []
    for line in f:
        parts = line.strip().split(',')
        image_path = parts[0]
        lat1, lon1, lat2, lon2 = map(float, parts[1:])
        bbox = {
            'image_path': image_path,
            'lat_min': min(lat1, lat2),
            'lat_max': max(lat1, lat2),
            'lon_min': min(lon1, lon2),
            'lon_max': max(lon1, lon2)
        }
        bboxes.append(bbox)


def boxes_intersect(box1, box2):
    return not (box1['lat_max'] < box2['lat_min'] or
                box1['lat_min'] > box2['lat_max'] or
                box1['lon_max'] < box2['lon_min'] or
                box1['lon_min'] > box2['lon_max'])


results = {}
for idx, (center_lat, center_lon) in enumerate(tqdm(locations, desc="Processing Locations")):

    if idx % 10000 == 0:
        print("finished", idx)


    location_box = {
        'lat_min': center_lat - HALF_BOX_LAT,
        'lat_max': center_lat + HALF_BOX_LAT,
        'lon_min': center_lon - HALF_BOX_LON,
        'lon_max': center_lon + HALF_BOX_LON
    }


    intersecting_bboxes = []
    for bbox in bboxes:
        if boxes_intersect(location_box, bbox):
            intersecting_bboxes.append({
                'image_path': bbox['image_path'],
                'lat_min': bbox['lat_min'],
                'lat_max': bbox['lat_max'],
                'lon_min': bbox['lon_min'],
                'lon_max': bbox['lon_max']
            })


    results[idx + 1] = {
        'center_lat': center_lat,
        'center_lon': center_lon,
        'intersecting_bboxes': intersecting_bboxes
    }


output_file = os.path.join(streetview_folder, 'intersecting_bboxes.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"结果已保存到 {output_file}")
