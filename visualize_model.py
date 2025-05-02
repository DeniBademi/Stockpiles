import os
import subprocess
import open3d as o3d

def visualize_sparse_model(model_path):
    ply_path = os.path.join(model_path, "0", "points3D.ply")

    # Convert BIN to PLY if PLY doesn't exist
    if not os.path.exists(ply_path):
        print("PLY file not found. Attempting to convert BIN to PLY...")
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", os.path.join(model_path, "0"),
            "--output_path", os.path.join(model_path, "0", "points3D.ply"),
            "--output_type", "PLY"
        ], check=True)



    # Try loading again
    if not os.path.exists(ply_path):
        print(f"PLY file still not found at {ply_path}")
        return

    pcd = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries([pcd])

# Set this to your actual sparse model path
sparse_model_path = "data/flight_1/workspace/sparse"
visualize_sparse_model(sparse_model_path)

# ply_path = os.path.join("data/flight_1/workspace/sparse", "0", "points3D.ply")
# print("Looking for:", os.path.abspath(ply_path))
# print("File exists?", os.path.exists(ply_path))