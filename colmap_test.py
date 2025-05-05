import os
import subprocess
import open3d as o3d

def run_colmap_sparse_reconstruction(image_dir, workspace_dir):
    database_path = os.path.join(workspace_dir, "database.db")
    sparse_model_path = os.path.join(workspace_dir, "sparse")

    os.makedirs(sparse_model_path, exist_ok=True)

    # # Step 1: Feature Extraction
    # subprocess.run([
    #     "colmap", "feature_extractor",
    #     "--database_path", database_path,
    #     "--image_path", image_dir
    # ], check=True)

    # # Step 2: Feature Matching
    # subprocess.run([
    #     "colmap", "exhaustive_matcher",
    #     "--database_path", database_path
    # ], check=True)

    # # Step 3: Sparse Reconstruction
    # subprocess.run([
    #     "colmap", "mapper",
    #     "--database_path", database_path,
    #     "--image_path", image_dir,
    #     "--output_path", sparse_model_path
    # ], check=True)

    # Step 4: Convert to PLY
    subprocess.run([
        "colmap", "model_converter",
        "--input_path", os.path.join(sparse_model_path, "0"),
        "--output_path", os.path.join(sparse_model_path, "0", "points3D.ply"),
        "--output_type", "LAS"
    ], check=True)

    print(f"Sparse model saved to: {sparse_model_path}")
    visualize_sparse_model(sparse_model_path)

def visualize_sparse_model(model_path):
    ply_path = os.path.join(model_path, "0", "points3D.ply")
    if not os.path.exists(ply_path):
        print(f"PLY file not found at {ply_path}")
        return
    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Replace these with your actual paths
    image_dir = "data/flight_1/images"
    workspace_dir = "data/flight_1/workspace"

    run_colmap_sparse_reconstruction(image_dir, workspace_dir)
