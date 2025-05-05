import csv
import numpy as np

def load_gps_csv(csv_path):
    gps_coords = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['image'].strip().upper()
            coords = np.array([float(row['easting']), float(row['northing']), float(row['altitude'])])
            gps_coords[name] = coords
    return gps_coords

def load_colmap_cameras(images_txt_path):
    colmap_coords = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('#') or len(line.strip()) == 0:
            continue

        tokens = line.strip().split()
        if len(tokens) != 10:
            continue

        image_name = tokens[-1].strip().upper()
        qw, qx, qy, qz = map(float, tokens[1:5])
        tx, ty, tz = map(float, tokens[5:8])

        colmap_coords[image_name] = np.array([tx, ty, tz])
    return colmap_coords

def compute_scale(csv_path, images_txt_path, img1, img2):
    gps_data = load_gps_csv(csv_path)
    model_data = load_colmap_cameras(images_txt_path)
    gps_data = {k.upper(): v for k, v in gps_data.items()}
    model_data = {k.upper(): v for k, v in model_data.items()}

    print("\nImages in GPS data:")
    for name in gps_data.keys():
        print(f"- {name}")

    print("\nImages in COLMAP model:")
    for name in model_data.keys():
        print(f"- {name}")

    if img1 not in gps_data or img2 not in gps_data:
        print(f"Images not found in GPS data: {img1}, {img2}")
        return

    if img1 not in model_data or img2 not in model_data:
        print(f"Images not found in COLMAP data: {img1}, {img2}")
        return

    real_dist = np.linalg.norm(gps_data[img1] - gps_data[img2])
    model_dist = np.linalg.norm(model_data[img1] - model_data[img2])
    scale = real_dist / model_dist

    print(f"\n=== Scale Calculation ===")
    print(f"Real-world distance: {real_dist:.3f} meters")
    print(f"Model distance: {model_dist:.3f} units")
    print(f"Scale factor (multiply model by this): {scale:.6f}")

if __name__ == "__main__":
    gps_csv_path = "gps_data.csv"
    colmap_images_txt = "data/flight_1/workspace/sparse/0/images.txt"
    image1 = "DJI_0414.JPG"  # change to your image name
    image2 = "DJI_0455.JPG"  # change to your image name
    image1 = image1.upper()
    image2 = image2.upper()
    compute_scale(gps_csv_path, colmap_images_txt, image1, image2)