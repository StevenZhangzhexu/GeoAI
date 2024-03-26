# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
import sys
import laspy
import open3d as o3d
import numpy as np


o3d.visualization.webrtc_server.enable_webrtc()


# Define a colormap for labels
def label_to_color(label):
    # Define your colormap here
    colormap = [
        # [0, 0, 0],     # Label 0 (background) as black
        [255, 0, 0],   # Label 1 as red
        [0, 255, 0],   # Label 2 as green
        [0, 0, 255],   # Label 3 as blue
        [255, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [0, 128, 255],
        [255, 128, 0],
        [0, 255, 128],
        [0, 128, 0],
        [0, 0, 128],
        [128, 0, 0],
        [128, 128, 0],
        [128, 0, 128],
        [128, 128, 0],
        # Add more colors for additional labels as needed
    ]
    if label < len(colormap):
        return colormap[label]
    else:
        return [255, 255, 255]  # Default to white for unknown labels


def test_viz():
    cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube_red.compute_vertex_normals()
    cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    o3d.visualization.draw(cube_red)


def o3d_viz(cloud, predicted=False):
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T

    # Create an Open3D PointCloud object with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if predicted:
        predictions = cloud.pred
        colors = np.array([label_to_color(label) for label in predictions])  # Map labels to colors
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    # Visualize the merged point cloud with colors
    # o3d.visualization.draw_geometries([pcd])
    pcd.paint_uniform_color((1.0, 0.0, 0.0))
    o3d.visualization.draw(pcd)


# Function to visualize point cloud
def visualize_point_cloud(file_path, annotated=False):
    # Load point cloud data using laspy
    las_data = laspy.read(file_path)
    o3d_viz(las_data, predicted=annotated)


if __name__ == "__main__":
    visualize_point_cloud(sys.argv[1])
