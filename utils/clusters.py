import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

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
