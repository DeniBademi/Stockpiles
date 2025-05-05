import sys

# assert sys.base_prefix != sys.prefix, "You are running this script in the base environment, please run it in a virtual environment"

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

# print versions
print(f"laspy version: {laspy.__version__}")
print(f"numpy version: {np.__version__}")
print(f"open3d version: {o3d.__version__}")

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

def visualize_pcd(pcd, plot_rgb=True):

    # Visualize
    o3d.visualization.draw_geometries([pcd])

def cluster_points(pcd, eps=0.5, min_points=10):
    """
    Perform DBSCAN clustering on the point cloud

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        eps: float, optional (default=0.5)
            The maximum distance between two points to be considered neighbors
        min_points: int, optional (default=10)
            The minimum number of points required to form a cluster

    Returns:
        tuple
            labels: Array of cluster labels for each point (-1 for noise)
            n_clusters: Number of clusters found
    """
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    print(f"Number of points before downsampling: {len(points):,}")
    # downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.1) # this down
    points = np.asarray(pcd.points)
    print(f"Number of points after downsampling: {len(points):,}")

    # Perform DBSCAN clustering
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_

    # Number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"\n=== Clustering Information ===")
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {np.sum(labels == -1)}")

    return labels, n_clusters, pcd

def visualize_clusters(pcd, labels):
    """
    Visualize the point cloud with different colors for each cluster

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        labels: numpy.ndarray
            Array of cluster labels for each point
    """
    # Create a color map for the clusters

    # remove points that are not in the clusters
    pcd = pcd.select_by_index(np.where(labels != -1)[0])
    labels = labels[np.where(labels != -1)[0]]
    max_label = labels.max()
    colors = np.zeros((len(labels), 3))

    # Generate random colors for each cluster
    for label in range(max_label + 1):
        if label == -1:  # Noise points
            colors[labels == label] = [0, 0, 0]  # Black for noise
        else:
            colors[labels == label] = np.random.rand(3)  # Random color for each cluster

    # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd])

def compute_cluster_volumes(pcd, labels):
    """
    Compute the volume of each cluster using convex hulls

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        labels: numpy.ndarray
            Array of cluster labels for each point

    Returns:
        dict
            Dictionary mapping cluster labels to their volumes
    """
    # Get points as numpy array
    points = np.asarray(pcd.points)

    # Dictionary to store volumes
    cluster_volumes = {}

    # Process each cluster
    for label in np.unique(labels):
        if label == -1:  # Skip noise points
            continue

        # Get points belonging to this cluster
        cluster_points = points[labels == label]

        # Skip clusters with too few points to form a convex hull
        if len(cluster_points) < 4:  # Need at least 4 points for 3D convex hull
            cluster_volumes[label] = 0
            continue

        try:
            # Compute convex hull
            hull = ConvexHull(cluster_points)

            # Store volume
            cluster_volumes[label] = hull.volume

        except Exception as e:
            print(f"Error computing volume for cluster {label}: {str(e)}")
            cluster_volumes[label] = 0

    # Print volume information
    print("\n=== Cluster Volumes ===")
    total_volume = sum(cluster_volumes.values())
    for label, volume in cluster_volumes.items():
        percentage = (volume / total_volume) * 100 if total_volume > 0 else 0
        print(f"Cluster {label}: {volume:.2f} cubic units ({percentage:.1f}% of total)")
    print(f"Total volume: {total_volume:.2f} cubic units")

    return cluster_volumes



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
    print("Exported to output.las")

if __name__ == "__main__":
    las_file = "data/1 stockpile 19-13-2025_group1_densified_point_cloud.las"

    ply_file = "data/flight_1/workspace/sparse/0/points3D.ply"
    # las_file = "data/flight_1/workspace/sparse/0/points3D.las"
    # convert_ply_to_las(ply_file, las_file)
    # las_file = "data/Stockpile 2 19-03-2025_group1_densified_point_cloud.las"

    # Extract metadata
    las = extract_las_metadata(las_file)

    # # Remove ground plane and visualize both ground and non-ground points
    # non_ground, ground = remove_ground_plane(las_file, distance_threshold=1.5)

    # # visualize_pcd(non_ground)
    # # # Perform clustering on the non-ground points
    # labels, n_clusters, pcd = cluster_points(non_ground, eps=1, min_points=50)

    # # Compute and print cluster volumes
    # cluster_volumes = compute_cluster_volumes(pcd, labels)

    # # Visualize the clusters
    # visualize_clusters(non_ground, labels)
    # visualize_las_file(las_file, use_rgb=True)
