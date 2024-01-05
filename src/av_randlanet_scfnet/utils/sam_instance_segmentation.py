import os
import sys
import numpy as np
import laspy
import helper_las
import traceback


samlidar_pythonpath = "/home/pc1/miniconda3/envs/samlidar/bin/python"
sys.path.append(samlidar_pythonpath)
os.environ["PYTHONPATH"] = samlidar_pythonpath

try:
    sys.path.index(samlidar_pythonpath)    # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(samlidar_pythonpath)    # Or os.getcwd() for this directory


label_to_min_points = {
                    0: 1000,
                    1: 5000,
                    2: 5000,
                    3: 3000,
                    4: 4000,
                    5: 3000,
                    6: 2000,
                    7: 5000,
                    8: 5000,
                    9: 3000,
                    10: 3000,
                    11: 4000,
                    12: 10000
                }
labels_with_elongations = [5, 6, 10, 11, 12]
labels_with_orientations = [5, 6, 10]


def save_separate_laz_point_cloud_objects(output_file_path, las_reader, object_id):
    segment_points = las_reader.points[las_reader.segment_id == object_id]
    points = np.vstack(
        (segment_points['x'], segment_points['y'], segment_points['z'])).T

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
    print("Separating objects from", segment_file)
    inFile = laspy.read(segment_file)
    print(len(inFile.points))

    # lists column names in las file.
    point_format = inFile.point_format
    print(list(point_format.dimension_names))

    # Retrieve unique segment IDs
    inst_ids = set(np.unique(inFile.segment_id))
    print(inst_ids)

    object_coords = []
    seg_name = segment_file[:-4].split("/")[-1]
    label_id = int(seg_name.split("_")[1])
    label_min_points = label_to_min_points[label_id]

    # Iterate over each segment ID and create a separate file for each segment
    i = 0
    for obj_id in inst_ids:
        try:
            # Copy the points from the input file that belong to the current segment
            segment_points = inFile.points[inFile.segment_id == obj_id]

            # check if its possibly a full esteemed object
            if len(segment_points) > label_min_points:
                # Extract the coordinates from the segment_points array
                coordinates = np.vstack(
                    (segment_points['x'], segment_points['y'], segment_points['z'])).T
                print(len(coordinates))

                # Save the point cloud as a .laz file
                output_filepath = os.path.join(
                    output_dir, f"{seg_name}_object_{obj_id}.laz")
                save_separate_laz_point_cloud_objects(
                    output_filepath, inFile, obj_id)
                calc_coords = helper_las.convert_and_save_wgs84(
                    output_filepath, coordinates, label_id)
                object_coords.append({
                    "id": seg_name + "_" + str(i),
                    # "coords": bc_coord
                    "start": calc_coords[0],
                    "end": calc_coords[1],
                    "center": calc_coords[2],
                    "orientation": calc_coords[3]
                })
                i += 1
        except Exception as err:
            print(err)
            traceback.print_exc()

    return {"label": seg_name.split("_")[1], "objects": object_coords}


def run_sam_instance_segmentation(filename):
    # from segment_lidar import samlidar
    import samlidar
    # from segment_lidar import view

    print("Running SAM-LiDAR Instance Segmentation for", filename)
    seg_dir = 'av_randlanet_scfnet/results/%s/separate_segments/' % filename
    model = samlidar.SamLidar(ckpt_path="sam_vit_h_4b8939.pth")
    # viewpoint = view.TopView()
    sam_dir = 'av_randlanet_scfnet/results/%s/separate_instances/' % filename
    os.makedirs(sam_dir, exist_ok=True)
    obj_dir = 'av_randlanet_scfnet/results/%s/separate_objects/' % filename
    os.makedirs(obj_dir, exist_ok=True)

    # define .json dicts
    segment_objects = []

    seg_files = os.listdir(seg_dir)
    print(seg_files)

    for each in seg_files:
        seg_path = os.path.join(seg_dir, each)
        sam_path = os.path.join(sam_dir, each)
        if os.path.exists(seg_path):
            try:
                points = model.read(seg_path)
                labels, *_ = model.segment(points=points)
                # labels, *_ = model.segment(points=points, view=viewpoint, image_path="raster.tif",
                #                            labels_path="labeled.tif")
                model.write(points=points, segment_ids=labels,
                            save_path=sam_path)
                print("Saved SAM instance segmentation for", each)
                object_coords = separate_and_cluster_point_cloud_objects(
                    sam_path, obj_dir)
                segment_objects.append(object_coords)
            except Exception as err:
                print(err)
                traceback.print_exc()

    helper_las.save_segment_object_bc_coords(filename, segment_objects)
    print("Saved segmented objects list of positional coordinates for", filename)

    # helper_las.copy_predictions()


if __name__ == '__main__':
    run_sam_instance_segmentation(sys.argv[1])
