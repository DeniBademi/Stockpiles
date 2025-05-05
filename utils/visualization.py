import numpy as np
import open3d as o3d
import laspy
from .alignment import align_with_principal_axes
from .point_cloud import remove_ground_plane, center_point_cloud
import os

def visualize_las_file(file_path, use_rgb=True, only_non_ground=False, center=True, align_pca=True):

    pcd = load_pcd_from_file(file_path)

    # if only_non_ground:
    #     non_ground, ground = remove_ground_plane(file_path, distance_threshold=0.3)
    #     pcd = non_ground
    #     colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(non_ground.points)
    #     pcd.colors = o3d.utility.Vector3dVector(non_ground.colors)

    visualize_pcd(pcd, plot_rgb=use_rgb, center=center, align_pca=align_pca)

def visualize_pcd(pcd, plot_rgb=True, center=False, align_pca=False):
    # Center the point cloud if requested
    if center:
        pcd = center_point_cloud(pcd)

    # Align with principal axes if requested
    if align_pca:
        pcd = align_with_principal_axes(pcd)

    # Create coordinate frame
    points = np.asarray(pcd.points)
    max_extent = max(
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    )
    frame_size = max_extent * 0.1  # 10% of the maximum extent

    # Create coordinate frame at the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size,
        origin=[0, 0, 0]
    )

    # Visualize
    o3d.visualization.draw_geometries([pcd, coordinate_frame])


def load_pcd_from_file(file_path):

    # Read LAS file
    las = laspy.read(file_path)

    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add RGB colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # Normalize RGB values to [0,1] range
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd