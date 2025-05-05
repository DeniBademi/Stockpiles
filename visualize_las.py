import sys

# assert sys.base_prefix != sys.prefix, "You are running this script in the base environment, please run it in a virtual environment"

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from utils.point_cloud import align_with_principal_axes
from utils.visualization import visualize_las_file

def extract_las_metadata(file_path, align_pca=False):
    """
    Extract metadata from a LAS file, optionally aligning with principal axes.

    Args:
        file_path: str
            Path to the LAS file
        align_pca: bool
            Whether to align the point cloud with principal axes before showing statistics
    """
    # Read LAS file
    las = laspy.read(file_path)

    # Extract header information
    print("\n=== LAS File Metadata ===")
    print(f"Point count: {las.header.point_count}")
    print(f"Version: {las.header.version}")
    print(f"Point format: {las.header.point_format}")

    # Units and coordinate system information
    print("\n=== Coordinate System Information ===")
    print(f"Scale factors (x,y,z): {las.header.scales}")
    print(f"Offsets (x,y,z): {las.header.offsets}")
    # The units in LAS files are typically meters, but can be feet in some cases
    # The scale factor helps determine the precision of the coordinates
    print(f"Coordinate units: {'Meters' if las.header.scales[0] < 0.01 else 'Feet'} (estimated based on scale factor)")

    # Create point cloud for PCA if needed
    if align_pca:
        pcd = o3d.geometry.PointCloud()
        points = np.vstack((las.x, las.y, las.z)).T
        pcd.points = o3d.utility.Vector3dVector(points)

        # Add colors if available
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Align with principal axes
        aligned_pcd = align_with_principal_axes(pcd)
        aligned_points = np.asarray(aligned_pcd.points)

        print("\n=== Original Spatial Extent ===")
        print(f"X range: {las.header.x_min:.2f} to {las.header.x_max:.2f}, {las.header.x_max - las.header.x_min:.2f} meters")
        print(f"Y range: {las.header.y_min:.2f} to {las.header.y_max:.2f}, {las.header.y_max - las.header.y_min:.2f} meters")
        print(f"Z range: {las.header.z_min:.2f} to {las.header.z_max:.2f}, {las.header.z_max - las.header.z_min:.2f} meters")

        print("\n=== Aligned Spatial Extent (Principal Axes) ===")
        print(f"X range: {np.min(aligned_points[:, 0]):.2f} to {np.max(aligned_points[:, 0]):.2f}, {np.max(aligned_points[:, 0]) - np.min(aligned_points[:, 0]):.2f} meters")
        print(f"Y range: {np.min(aligned_points[:, 1]):.2f} to {np.max(aligned_points[:, 1]):.2f}, {np.max(aligned_points[:, 1]) - np.min(aligned_points[:, 1]):.2f} meters")
        print(f"Z range: {np.min(aligned_points[:, 2]):.2f} to {np.max(aligned_points[:, 2]):.2f}, {np.max(aligned_points[:, 2]) - np.min(aligned_points[:, 2]):.2f} meters")
    else:
        print("\n=== Spatial Extent ===")
        print(f"X range: {las.header.x_min:.2f} to {las.header.x_max:.2f}, {las.header.x_max - las.header.x_min:.2f} meters")
        print(f"Y range: {las.header.y_min:.2f} to {las.header.y_max:.2f}, {las.header.y_max - las.header.y_min:.2f} meters")
        print(f"Z range: {las.header.z_min:.2f} to {las.header.z_max:.2f}, {las.header.z_max - las.header.z_min:.2f} meters")

    # Available point attributes
    print("\n=== Available Point Attributes ===")
    print("Point dimensions:", list(las.point_format.dimension_names))

    # Point data statistics
    print("\n=== Point Data Statistics ===")
    if hasattr(las, 'classification'):
        unique_classes = np.unique(las.classification)
        print("Classifications found:", unique_classes)
        for cls in unique_classes:
            count = np.sum(las.classification == cls)
            print(f"Class {cls}: {count} points")

    if hasattr(las, 'intensity'):
        print(f"\nIntensity range: {np.min(las.intensity)} to {np.max(las.intensity)}")

    return las

def convert_ply_to_las(ply_file, las_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    las = laspy.create(point_format=3, file_version="1.2")

    las.x = -points[:, 0]
    las.y = -points[:, 1]
    las.z = -points[:, 2]

    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 65535).astype(np.uint16)  # LAS uses 16-bit colors
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]

    las.write(las_file)


def select_two_points_and_measure(file_path):
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    print("Please select exactly 2 points using Shift + left click. Press Q when done.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    if len(picked_ids) != 2:
        print(f"❗ Expected 2 points, but got {len(picked_ids)}. Try again.")
        return

    pt1 = points[picked_ids[0]]
    pt2 = points[picked_ids[1]]
    dist = np.linalg.norm(pt1 - pt2)

    print(f"\n📍 Point 1: {pt1}")
    print(f"📍 Point 2: {pt2}")
    print(f"📏 Distance: {dist:.3f} meters")

if __name__ == "__main__":
    las_file = "data/1 stockpile 19-13-2025_group1_densified_point_cloud.las"
    # las_file = "data/Stockpile 2 19-03-2025_group1_densified_point_cloud.las"

    ply_file = "data/flight_1/workspace/sparse/0/points3D.ply"
    las_file = "data/flight_1/workspace/sparse/0/points3D_filtered.las"
    select_two_points_and_measure(las_file)
    # convert_ply_to_las(ply_file, las_file)

    # las_file = "scaled_output.las"
    # visualize_las_file(las_file, use_rgb=True)

    # las = extract_las_metadata(las_file, align_pca=True)
    # visualize_las_file(las_file, use_rgb=True, align_pca=True)
    # # Remove ground plane and visualize both ground and non-ground points
    # non_ground, ground = remove_ground_plane(las_file, distance_threshold=4.5)

    # # Perform clustering on the non-ground points
    # labels, n_clusters, pcd = cluster_points(non_ground, eps=2, min_points=500)

    # # Compute and print cluster volumes
    # cluster_volumes = compute_cluster_volumes(pcd, labels)

    # # Visualize the clusters
    # visualize_clusters(pcd, labels)
    # visualize_las_file(las_file, use_rgb=True)


# Original las file:
# X range: 375140.58 to 375276.53, 135.95 meters
# Y range: 6309471.52 to 6309642.49, 170.97 meters
# Z range: 75.72 to 90.27, 14.55 meters

# Ours:
# === Spatial Extent ===
# X range: -29.39 to 28.95, 58.34 meters
# Y range: -48.52 to 40.43, 88.94 meters
# Z range: -23.58 to 9.27, 32.86 meters