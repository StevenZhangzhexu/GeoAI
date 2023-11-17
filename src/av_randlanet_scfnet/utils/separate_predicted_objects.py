import os
import sys
import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from time import sleep
import helper_las


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
    print("Clustering and separating objects from", label_to_names[segment_id], "...")
    # Apply DBSCAN clustering to the segment's coordinates
    clustering = DBSCAN(eps=0.5, min_samples=100).fit(coordinates)

    # Get unique cluster labels from clustering results
    unique_labels = np.unique(clustering.labels_)
    print(f"Number of objects in {label_to_names[segment_id]}:", len(unique_labels))

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

    print("Clustering and saving objects from", label_to_names[segment_id], "completed.")


def separate_segmented_point_clouds(filename):
    pred_dir = 'av_randlanet_scfnet/results/%s/predictions/' % filename
    segment_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    output_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    os.makedirs(output_dir, exist_ok=True)

    inFile = laspy.read(os.path.join(pred_dir, filename[:-4] + '.laz'))
    print(len(inFile.points))

    segment_ids = set(np.unique(inFile.pred))
    print(segment_ids)

    # Create a queue to manage the tasks
    task_queue = Queue()

    # Add each segment to the task queue
    for segment_id in segment_ids:
        segment_points = inFile.points[inFile.pred == segment_id]
        coordinates = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T
        task_queue.put((coordinates, output_dir, segment_id))

        # Save the segmented point cloud as a .laz file
        output_file = os.path.join(segment_dir, f"segment_{segment_id}_{label_to_names[segment_id]}.laz")
        save_separate_laz_point_cloud(output_file, coordinates)

    # Create a process pool to cluster the segments in parallel
    pool = Pool(processes=3)

    # Start the workers
    for i in range(pool._processes):
        # pool.apply_async(clustering_and_save_objects, task_queue.get(), args=(coordinates, output_dir, segment_id))
        pool.apply_async(clustering_and_save_objects, task_queue.get())

    # Wait for all tasks to finish
    pool.close()
    pool.join()

    # copy the results to shared folder
    helper_las.copy_predictions()

    # Print a message indicating that all tasks have finished
    print("All tasks have finished for ", filename)


def separate_and_segment_point_clouds(filename):
    pred_dir = 'av_randlanet_scfnet/results/%s/predictions/' % filename
    segment_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    # output_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    os.makedirs(segment_dir, exist_ok=True)
    # os.makedirs(output_dir, exist_ok=True)

    inFile = laspy.read(os.path.join(pred_dir, filename[:-4] + '.laz'))
    print(len(inFile.points))

    segment_ids = set(np.unique(inFile.pred))
    print(segment_ids)

    # Save segmented point clouds and perform clustering in parallel
    # with Pool(processes=4) as pool:
    for segment_id in segment_ids:
        segment_points = inFile.points[inFile.pred == segment_id]
        coordinates = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T

        # Save the segmented point cloud
        segment_file = os.path.join(segment_dir, f"segment_{segment_id}_{label_to_names[segment_id]}.laz")
        save_separate_laz_point_cloud(segment_file, coordinates)

        # Cluster each segment in parallel
        # pool.apply_async(clustering_and_save_objects, args=(coordinates, output_dir, segment_id))


def separate_instance_objects(input_file, output_dir, label_id):
    # Open the input file
    inFile = laspy.read(input_file)

    print(len(inFile.points))

    # Retrieve unique segment IDs
    inst_ids = set(np.unique(inFile.segment_id))
    print(inst_ids)

    # Iterate over each segment ID and create a separate file for each segment
    for obj_id in inst_ids:
        # Copy the points from the input file that belong to the current segment
        segment_points = inFile.points[inFile.segment_id == obj_id]

        # Extract the coordinates from the segment_points array
        coordinates = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T
        print(len(coordinates))

        # Save the point cloud as a .laz file
        output_file = os.path.join(output_dir, f"segment_{label_id}_{label_to_names[label_id]}_{obj_id}.laz")
        save_separate_laz_point_cloud(output_file, coordinates)


def separate_and_cluster_point_cloud_objects(filename):
    # pred_dir = 'av_randlanet_scfnet/results/%s/predictions/' % filename
    # segment_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    segment_dir = 'av_randlanet_scfnet/results/%s/separate_instances/' % filename
    output_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    # os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # inFile = laspy.read(os.path.join(pred_dir, filename[:-4] + '.laz'))
    # print(len(inFile.points))

    # segment_ids = set(np.unique(inFile.pred))
    # print(segment_ids)

    segments = os.listdir(segment_dir)

    # Save segmented point clouds and perform clustering in parallel
    with Pool(processes=2) as pool:
        for segment_file in segments:
            # segment_points = laspy.read(os.path.join(segment_dir, segment_file))
            # print(len(segment_points.points))

            # segment_points = inFile.points[inFile.pred == segment_id]
            # coordinates = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T

            # Save the segmented point cloud
            # segment_file = os.path.join(segment_dir, f"segment_{segment_id}_{label_to_names[segment_id]}.laz")
            # save_separate_laz_point_cloud(segment_file, coordinates)

            # Cluster each segment in parallel
            # pool.apply_async(clustering_and_save_objects, args=(coordinates, output_dir, int(segment_file.split("_")[1])))

            # separate instances
            filepath = os.path.join(segment_dir, segment_file)
            pool.apply_async(separate_instance_objects, args=(filepath, output_dir, int(segment_file.split("_")[1])))
