import os
import sys
import laspy
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False


def move_forward(vis):
    ctr = vis.get_view_control()
    ctr.rotate(5.0, 0.0)
    ctr.camera_local_rotate(5.0, 0.0)
    ctr.change_field_of_view()
    ctr.scale(1.5)

    return False


def get_kitti_bbox_info(file_path):
    # Load the .laz file
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T

    # Calculate center point
    center = np.mean(points, axis=0)

    try:
        # Apply PCA and extract eigenvectors
        # Compute the covariance matrix
        cov = np.cov(points, rowvar=False)

        # Perform eigenvalue decomposition to get the rotation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        rot_mat = eigenvectors
    except:
        rot_mat = np.identity(3)  # Identity matrix representing no rotation
        # rot_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Transform points using the rotation matrix
    rotated_points = np.dot(points, rot_mat)

    # Calculate box size aligned with the elongation
    box_size = np.max(rotated_points, axis=0) - np.min(rotated_points, axis=0)

    print(center, rot_mat, box_size)
    bbox = o3d.geometry.OrientedBoundingBox(center, rot_mat, box_size)

    return bbox


def viz_pred_objdet(filename):
    # Directory containing .laz files
    directory_path = f'av_randlanet_scfnet/results/{filename}/separate_objects/'
    main_pcd_path = f'av_randlanet_scfnet/results/{filename}/predictions/{filename}'

    main = laspy.read(main_pcd_path)
    coords = np.vstack((main.x, main.y, main.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # Define a lookat point (you can adjust this based on your specific needs)
    center = np.mean(coords, axis=0)
    lookat = [center[0], center[1], center[2] + 5]  # Looking 5 units ahead of the center

    # Define the up vector (perpendicular to front)
    front = np.asarray(center) - np.asarray(lookat)
    front /= np.linalg.norm(front)
    up = np.cross([1, 0, 0], front)
    up /= np.linalg.norm(up)

    # Set the view parameters
    zoom = 0.7
    front = list(front)
    lookat = list(lookat)
    up = list(up)

    print(f"zoom={zoom}, front={front}, lookat={lookat}, up={up}")

    items = [pcd]
    for file in os.listdir(directory_path):
        if file.endswith(".laz"):
            laz_file_path = os.path.join(directory_path, file)
            bbox = get_kitti_bbox_info(laz_file_path)
            items.append(bbox)

    # visualization
    o3d.visualization.draw_geometries(items, zoom=zoom, front=front, up=up, lookat=lookat)
    # o3d.visualization.draw_geometries_with_animation_callback(items, callback_function=rotate_view)
    # o3d.visualization.draw_geometries_with_animation_callback(items, callback_function=move_forward)


if __name__ == '__main__':
    viz_pred_objdet(sys.argv[1])
