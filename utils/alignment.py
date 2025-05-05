import numpy as np
import open3d as o3d

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

def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    Detect the ground plane using RANSAC.

    Args:
        pcd: open3d.geometry.PointCloud
            The input point cloud
        distance_threshold: float
            Maximum distance a point can be from the plane to be considered an inlier
        ransac_n: int
            Number of points to sample for each RANSAC iteration
        num_iterations: int
            Number of RANSAC iterations to perform

    Returns:
        tuple
            plane_model: The coefficients (a,b,c,d) of the plane equation ax + by + cz + d = 0
            inliers: Indices of points that belong to the ground plane
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers