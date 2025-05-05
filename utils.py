import laspy
import numpy as np
import open3d as o3d

def extract_las_metadata(file_path):
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

    # Spatial extent
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

def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    Detect the ground plane using RANSAC
    Returns the ground plane model and inlier indices
    Args:
    pcd: open3d.geometry.PointCloud
        The input point cloud
    distance_threshold: float, optional (default=0.1)
        Maximum distance a point can be from the plane to be considered an inlier
    ransac_n: int, optional (default=3)
        Number of points to sample for each RANSAC iteration
    num_iterations: int, optional (default=1000)
        Number of RANSAC iterations to perform

    Returns:
    --------
    tuple
        plane_model: The coefficients (a,b,c,d) of the plane equation ax + by + cz + d = 0
        inliers: Indices of points that belong to the ground plane

    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    return plane_model, inliers

def remove_ground_plane(file_path, distance_threshold=0.1):
    """
    Remove the ground plane and points below it from the point cloud
    """
    # Read LAS file
    las = laspy.read(file_path)


    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    print("AAA")
    pcd.points = o3d.utility.Vector3dVector(points)
    print("BBB")
    # Add RGB colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # Normalize RGB values to [0,1] range
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("\n=== RGB Information ===")
        print(f"RGB values range: {np.min(colors)} to {np.max(colors)}")

    # Detect ground plane
    plane_model, inliers = detect_ground_plane(pcd, distance_threshold)

    # Get all points
    all_points = np.asarray(pcd.points)

    # Calculate distance of each point to the plane
    # Plane equation: ax + by + cz + d = 0
    a, b, c, d = plane_model
    distances = (a * all_points[:, 0] + b * all_points[:, 1] + c * all_points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

    # Keep points that are above or on the plane (distance >= 0)
    above_plane_indices = np.where(distances >= 0)[0]

    # Create new point cloud with only points above the plane
    above_plane = pcd.select_by_index(above_plane_indices)

    # Get ground points (for visualization)
    ground = pcd.select_by_index(inliers)

    # Print ground plane information
    print("\n=== Ground Plane Information ===")
    print(f"Plane equation: {plane_model[0]:.2f}x + {plane_model[1]:.2f}y + {plane_model[2]:.2f}z + {plane_model[3]:.2f} = 0")
    print(f"Number of ground points: {len(inliers):,}")
    print(f"Number of points above ground: {len(above_plane_indices):,}")
    print(f"Number of points removed: {len(points) - len(above_plane_indices):,}")

    return above_plane, ground

def visualize_las_file(file_path, use_rgb=True, only_non_ground=False):
    # Read LAS file
    las = laspy.read(file_path)

    # Extract points
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add RGB colors if available and requested
    if use_rgb and hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # Normalize RGB values to [0,1] range
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("\n=== RGB Information ===")
        print(f"RGB values range: {np.min(colors)} to {np.max(colors)}")


    if only_non_ground:
        non_ground, ground = remove_ground_plane(file_path, distance_threshold=0.3)
        pcd = non_ground
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(non_ground.points)
        pcd.colors = o3d.utility.Vector3dVector(non_ground.colors)
        # pcd.estimate_normals()
        # pcd.orient_normals_consistent_tangent_plane(10)
        # pcd.paint_uniform_color([0, 0, 1])
        # pcd.remove_radius_outlier(nb_points=20, radius=0.05)
        # pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)


    # Visualize
    o3d.visualization.draw_geometries([pcd])

