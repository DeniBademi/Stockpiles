import sys

# assert sys.base_prefix != sys.prefix, "You are running this script in the base environment, please run it in a virtual environment"

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

from utils.point_cloud import remove_ground_plane, align_with_principal_axes, center_point_cloud, remove_outliers
from utils.clusters import cluster_points, compute_cluster_volumes, downsample_pcd, visualize_clusters
from utils.visualization import load_pcd_from_file, visualize_las_file, visualize_pcd


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

if __name__ == "__main__":


    # las_file = "data/1 stockpile 19-13-2025_group1_densified_point_cloud.las"
    las_file = "data/flight_1/workspace/sparse/0/points3D.las"
    pcd = load_pcd_from_file(las_file)
    # visualize_pcd(pcd, plot_rgb=True)

    #remove outliers
    pcd = remove_outliers(pcd)

    # downsample
    pcd = downsample_pcd(pcd, voxel_size=0.1, verbose=False)

    # normalize coordinates
    pcd = center_point_cloud(pcd, verbose=False)

    # align with principal axes
    pcd = align_with_principal_axes(pcd, verbose=True)

    # remove ground plane
    non_ground, ground = remove_ground_plane(pcd, distance_threshold=4.5)
    # visualize_pcd(non_ground, plot_rgb=True)

    # # Perform clustering on the non-ground points
    labels, n_clusters, pcd = cluster_points(non_ground, eps=2, min_points=50)
    visualize_clusters(pcd, labels)

    # # Compute and print cluster volumes
    cluster_volumes = compute_cluster_volumes(pcd, labels)
