import numpy as np
import open3d as o3d
import laspy
from scipy.spatial import ConvexHull

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers from a point cloud.

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        nb_neighbors: int
            Number of neighbors to analyze for each point
        std_ratio: float
            Standard deviation ratio threshold for outlier removal

    Returns:
        open3d.geometry.PointCloud
            The filtered point cloud
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

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

def center_point_cloud(pcd):
    """
    Center the point cloud by subtracting the mean from each axis.

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud

    Returns:
        open3d.geometry.PointCloud
            The centered point cloud
    """
    points = np.asarray(pcd.points)
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    centered_pcd = o3d.geometry.PointCloud()
    centered_pcd.points = o3d.utility.Vector3dVector(centered_points)

    if pcd.has_colors():
        centered_pcd.colors = pcd.colors

    print("\n=== Centering Information ===")
    print(f"Original mean: {mean}")
    print(f"New mean: {np.mean(centered_points, axis=0)}")

    return centered_pcd

def convert_ply_to_las(ply_path, las_path):
    """
    Convert PLY file to LAS format.

    Args:
        ply_path: str
            Path to input PLY file
        las_path: str
            Path to output LAS file
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

    las = laspy.LasData(header)
    las.x = -points[:, 0]
    las.y = -points[:, 1]
    las.z = -points[:, 2]

    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 65535).astype(np.uint16)
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]

    las.write(las_path)
    print(f"Converted PLY to LAS: {las_path}")

def compute_cluster_volumes(pcd, labels):

    """
    Compute the volume of each cluster using convex hulls.

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        labels: numpy.ndarray
            Array of cluster labels for each point

    Returns:
        dict
            Dictionary mapping cluster labels to their volumes
    """
    points = np.asarray(pcd.points)
    cluster_volumes = {}

    for label in np.unique(labels):
        if label == -1:  # Skip noise points
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) < 4:
            cluster_volumes[label] = 0
            continue

        try:
            hull = ConvexHull(cluster_points)
            cluster_volumes[label] = hull.volume
        except Exception as e:
            print(f"Error computing volume for cluster {label}: {str(e)}")
            cluster_volumes[label] = 0

    total_volume = sum(cluster_volumes.values())
    print("\n=== Cluster Volumes ===")
    for label, volume in cluster_volumes.items():
        percentage = (volume / total_volume) * 100 if total_volume > 0 else 0
        print(f"Cluster {label}: {volume:.2f} cubic units ({percentage:.1f}% of total)")
    print(f"Total volume: {total_volume:.2f} cubic units")

    return cluster_volumes

def center_point_cloud(pcd):
    """
    Center the point cloud by subtracting the mean from each axis.

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud

    Returns:
        open3d.geometry.PointCloud
            The centered point cloud
    """
    # Get points as numpy array
    points = np.asarray(pcd.points)

    # Calculate mean for each axis
    mean = np.mean(points, axis=0)

    # Subtract mean from all points
    centered_points = points - mean

    # Create new point cloud with centered points
    centered_pcd = o3d.geometry.PointCloud()
    centered_pcd.points = o3d.utility.Vector3dVector(centered_points)

    # Copy colors if they exist
    if pcd.has_colors():
        centered_pcd.colors = pcd.colors

    print("\n=== Centering Information ===")
    print(f"Original mean: {mean}")
    print(f"New mean: {np.mean(centered_points, axis=0)}")

    return centered_pcd

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

def align_with_principal_axes(pcd, verbose=True):
    """
    Align the point cloud with its principal axes using PCA.
    More robust version that handles sparse point clouds better.

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        verbose: bool
            Whether to print detailed information

    Returns:
        open3d.geometry.PointCloud
            The aligned point cloud
    """
    # Get points as numpy array
    points = np.asarray(pcd.points)

    # Center the points
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    # Compute covariance matrix
    cov_matrix = np.cov(centered_points.T)

    # Get eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    # Create rotation matrix
    rotation_matrix = eigenvectors.T

    # Apply rotation
    aligned_points = np.dot(centered_points, rotation_matrix)

    # Check if we need to flip any axes
    # This helps ensure consistent orientation for sparse point clouds
    for i in range(3):
        # If the spread is more negative than positive, flip the axis
        if np.sum(aligned_points[:, i] < 0) > np.sum(aligned_points[:, i] > 0):
            aligned_points[:, i] = -aligned_points[:, i]
            rotation_matrix[i, :] = -rotation_matrix[i, :]

    # Create new point cloud with aligned points
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)

    # Copy colors if they exist
    if pcd.has_colors():
        aligned_pcd.colors = pcd.colors

    if verbose:
        print("\n=== PCA Alignment Information ===")
        print("Original mean:", mean)
        print("New mean:", np.mean(aligned_points, axis=0))
        print("Eigenvalues (variances):", eigenvalues)
        print("Principal directions:")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            print(f"Direction {i+1}: {vec} (variance: {val:.2f})")

        # Print extent information
        print("\nExtent before alignment:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            min_val = np.min(points[:, i])
            max_val = np.max(points[:, i])
            print(f"{axis}: {min_val:.2f} to {max_val:.2f}, range: {max_val - min_val:.2f}")

        print("\nExtent after alignment:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            min_val = np.min(aligned_points[:, i])
            max_val = np.max(aligned_points[:, i])
            print(f"{axis}: {min_val:.2f} to {max_val:.2f}, range: {max_val - min_val:.2f}")

    return aligned_pcd