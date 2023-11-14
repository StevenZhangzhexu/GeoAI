import os
import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
from threading import Thread
from time import sleep


label_to_names = {
                    0: 'Bollard',
                    1: 'Building',
                    2: 'BusStop',
                    3: 'ControlBox',
                    4: 'Ground',
                    5: 'LampPost',
                    6: 'Pole',
                    7: 'Railing',
                    8: 'Road',
                    9: 'Shrub',
                    10: 'Sign',
                    11: 'SolarPanel',
                    12: 'Tree'
                }


def save_separate_laz_point_cloud(output_file_path, points):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    las_writer = laspy.LasData(header)
    las_writer.x = points[:, 0]
    las_writer.y = points[:, 1]
    las_writer.z = points[:, 2]

    las_writer.write(output_file_path)
    print(len(points), "point cloud saved to:", output_file_path)


def clustering_and_save_objects(coordinates, output_dir, segment_id):
    print("CLustering and separating objects from", label_to_names[segment_id], "...")
    # Apply DBSCAN clustering to the segment's coordinates
    clustering = DBSCAN(eps=0.5, min_samples=100).fit(coordinates)

    # Get unique cluster labels from clustering results
    unique_labels = np.unique(clustering.labels_)

    # Iterate over clusters and save them as separate files
    for cluster_label in unique_labels:
        if cluster_label == -1:
            continue  # Skip noise points

        # Select points belonging to the current cluster
        cluster_points = coordinates[clustering.labels_ == cluster_label]

        # Save the cluster as a separate .laz file
        output_file = os.path.join(output_dir,
                                   f"segment_{segment_id}_{label_to_names[segment_id]}_object_{cluster_label}.laz")
        save_separate_laz_point_cloud(output_file, cluster_points)

    sleep(5)


def separate_segmented_point_clouds(filename):
    pred_dir = 'av_randlanet_scfnet/results/%s/predictions/' % filename
    segment_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    output_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    os.makedirs(output_dir, exist_ok=True)

    inFile = laspy.read(os.path.join(pred_dir, filename[:-4] + '.laz'))
    print(len(inFile.points))

    segment_ids = set(np.unique(inFile.pred))
    print(segment_ids)

    for segment_id in segment_ids:
        segment_points = inFile.points[inFile.pred == segment_id]
        coordinates = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T
        print(label_to_names[segment_id], len(coordinates))

        # Save the segmented point cloud as a .laz file
        output_file = os.path.join(segment_dir, f"segment_{segment_id}_{label_to_names[segment_id]}.laz")
        save_separate_laz_point_cloud(output_file, coordinates)

        # threading
        thread = Thread(target=clustering_and_save_objects, args=(coordinates, output_dir, segment_id))
        thread.start()
        thread.join()


def separate_and_cluster_point_clouds(filename):
    pred_dir = 'av_randlanet_scfnet/results/%s/predictions/' % filename
    segment_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    output_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    inFile = laspy.read(os.path.join(pred_dir, filename[:-4] + '.laz'))
    print(len(inFile.points))

    segment_ids = set(np.unique(inFile.pred))
    print(segment_ids)

    # Save segmented point clouds and perform clustering in parallel
    with Pool(processes=4) as pool:
        for segment_id in segment_ids:
            segment_points = inFile.points[inFile.pred == segment_id]
            coordinates = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T

            # Save the segmented point cloud
            segment_file = os.path.join(segment_dir, f"segment_{segment_id}_{label_to_names[segment_id]}.laz")
            save_separate_laz_point_cloud(segment_file, coordinates)

            # Cluster each segment in parallel
            pool.apply_async(clustering_and_save_objects, args=(coordinates, output_dir, segment_id))
