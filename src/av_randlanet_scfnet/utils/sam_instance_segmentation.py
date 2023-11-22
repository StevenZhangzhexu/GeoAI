import os
import sys
import numpy as np
import laspy
import helper_las


samlidar_pythonpath = "/home/pc1/miniconda3/envs/samlidar/bin/python"
sys.path.append(samlidar_pythonpath)
os.environ["PYTHONPATH"] = samlidar_pythonpath

try:
    sys.path.index(samlidar_pythonpath)    # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(samlidar_pythonpath)    # Or os.getcwd() for this directory


def save_separate_laz_point_cloud_objects(output_file_path, las_reader, object_id):
    segment_points = las_reader.points[las_reader.segment_id == object_id]
    points = np.vstack((segment_points['x'], segment_points['y'], segment_points['z'])).T

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])
    las_writer = laspy.LasData(header)

    las_writer.x = points[:, 0]
    las_writer.y = points[:, 1]
    las_writer.z = points[:, 2]

    las_writer.write(output_file_path)
    print(len(points), "point cloud saved to:", output_file_path)


def separate_and_cluster_point_cloud_objects(segment_file, output_dir):
    # Save segmented point clouds
    seg_name = segment_file[:-4].split("/")[-1]
    inFile = laspy.read(segment_file)
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
        output_filepath = os.path.join(output_dir, f"{seg_name}_object_{obj_id}.laz")
        save_separate_laz_point_cloud_objects(output_filepath, inFile, obj_id)
        helper_las.convert_and_save_wgs84(output_filepath, coordinates)


def run_sam_instance_segmentation(filename):
    from segment_lidar import samlidar
    # import samlidar
    print("Running SAM-LiDAR Instance Segmentation for", filename)
    seg_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
    save_dir = seg_dir.replace("separate_segments", "separate_instances")
    os.makedirs(save_dir, exist_ok=True)
    output_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(seg_dir)
    for each in files:
        seg_path = os.path.join(save_dir, each)
        try:
            points = model.read(os.path.join(seg_dir, each))
            labels, *_ = model.segment(points=points)
            model.write(points=points, segment_ids=labels, save_path=seg_path)
            print("Saved instance segmentation for", each)
            separate_and_cluster_point_cloud_objects(seg_path, output_dir)
        except Exception as err:
            print(err)

    helper_las.copy_predictions()


if __name__ == '__main__':
    run_sam_instance_segmentation(sys.argv[1])
