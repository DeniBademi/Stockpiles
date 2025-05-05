import os
import subprocess
import open3d as o3d
import numpy as np

from utils import (
    extract_gps_metadata,
    calculate_distance,
    run_colmap,
    scale_ply_with_gps,
    remove_outliers,
    convert_ply_to_las
)

def calculate_scale_factor(image_dir):
    """Calculate scale factor based on GPS coordinates from images."""
    # Get all images and their GPS coordinates
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gps_coords = []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        gps = extract_gps_metadata(image_path)
        if gps:
            gps_coords.append((filename, gps))

    if len(gps_coords) < 2:
        print("Warning: Not enough GPS data found in images")
        return 1.0

    # Calculate distances between consecutive images
    distances = []
    for i in range(len(gps_coords) - 1):
        dist = calculate_distance(
            gps_coords[i][1][0], gps_coords[i][1][1],
            gps_coords[i+1][1][0], gps_coords[i+1][1][1]
        )
        distances.append(dist)

    # Use the median distance as it's more robust to outliers
    median_distance = np.median(distances)
    print(f"Median distance between consecutive images: {median_distance:.2f} meters")

    # Get the COLMAP reconstruction scale
    ply_path = os.path.join(os.path.dirname(image_dir), "workspace", "sparse", "0", "points3D.ply")
    if os.path.exists(ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        # Calculate the median distance between consecutive points in the reconstruction
        point_distances = []
        for i in range(len(points) - 1):
            dist = np.linalg.norm(points[i+1] - points[i])
            point_distances.append(dist)

        median_point_distance = np.median(point_distances)
        print(f"Median distance between consecutive points in reconstruction: {median_point_distance:.2f} units")

        # Calculate scale factor
        if median_point_distance > 0:
            scale_factor = median_distance / median_point_distance
            print(f"Calculated scale factor: {scale_factor:.2f}")
            return scale_factor

    return 1.0

def run_colmap_sparse_reconstruction(image_dir, workspace_dir):
    database_path = os.path.join(workspace_dir, "database.db")
    sparse_model_path = os.path.join(workspace_dir, "sparse")

    os.makedirs(sparse_model_path, exist_ok=True)

    # Run COLMAP pipeline
    success = run_colmap(image_dir, workspace_dir)
    if not success:
        print("COLMAP reconstruction failed")
        return None

    # Step 4: Convert to PLY
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", os.path.join(sparse_model_path, "0"),
        "--output_path", os.path.join(sparse_model_path, "0", "points3D.ply"),
        "--output_type", "PLY"
    ], check=True)

    # Step 5: Scale the PLY file based on GPS data
    scale_factor = calculate_scale_factor(image_dir)
    if scale_factor != 1.0:
        ply_path = os.path.join(sparse_model_path, "0", "points3D.ply")
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.scale(scale_factor, center=pcd.get_center())
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Scaled point cloud by factor: {scale_factor}")

    # Step 6: Convert to LAS format
    ply_path = os.path.join(sparse_model_path, "0", "points3D.ply")
    las_path = os.path.join(sparse_model_path, "0", "points3D.las")
    convert_ply_to_las(ply_path, las_path)

    # Step 7: Remove outliers
    # filtered_las_path = remove_outliers(las_path, nb_neighbors=20, std_ratio=2.0)

    print(f"Sparse model saved to: {sparse_model_path}")

if __name__ == "__main__":
    # Replace these with your actual paths
    image_dir = "data/flight_1/images"
    workspace_dir = "data/flight_1/workspace"

    # Run the reconstruction
    run_colmap_sparse_reconstruction(image_dir, workspace_dir)
