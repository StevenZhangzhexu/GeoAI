import os
import laspy
import numpy as np
from pyproj import Proj, itransform
import shutil
from distutils.dir_util import copy_tree
import helper_json


def read_las(pcd_filepath):
    # Add extension if not there
    if not pcd_filepath.endswith('.laz') or not pcd_filepath.endswith('.las'):
        if pcd_filepath.endswith('.ply'):
            pcd_filepath = str(pcd_filepath).replace('input_0.060', 'original_las').replace('.ply', '.laz')

    las_reader = laspy.read(pcd_filepath)
    print("File loaded from:", pcd_filepath)

    return las_reader


def get_laz_points(filepath):
    inFile = laspy.read(filepath)
    coordinates = np.vstack((inFile['x'], inFile['y'], inFile['z'])).T
    return coordinates


def write_laz(save_filepath, original_las, points, preds):
    # Add extension if not there
    if not save_filepath.endswith('.laz'):
        save_filepath = save_filepath[:-4] + '.laz'

    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.int32))
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a Las
    las_writer = laspy.LasData(header)

    # Define a list of attributes to transfer
    attributes_to_transfer = list(original_las.point_format.dimension_names)
    print(attributes_to_transfer)

    # Transfer attributes from split_laz to las_writer
    for attr_name in attributes_to_transfer:
        setattr(las_writer, attr_name, getattr(original_las, attr_name))

    las_writer.x = points[:, 0]
    las_writer.y = points[:, 1]
    las_writer.z = points[:, 2]
    las_writer.pred = preds

    las_writer.write(save_filepath)
    print("Prediction in .laz saved in path:", save_filepath)


def write_sub_laz(save_filepath, points):
    # Add extension if not there
    if not save_filepath.endswith('.laz'):
        save_filepath = save_filepath[:-4] + '.laz'

    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # 2. Create a Las
    las_writer = laspy.LasData(header)
    las_writer.x = points[:, 0]
    las_writer.y = points[:, 1]
    las_writer.z = points[:, 2]

    las_writer.write(save_filepath)
    print("Prediction in .laz saved in path:", save_filepath)


def convert_svy21_to_wgs84(svy21_points):
    # Define the SVY21 projection
    svy21_proj = Proj(init='epsg:3414')  # SVY21 Projection

    # Define the WGS 84 projection for geolocation
    wgs84_proj = Proj(init='epsg:4326')  # WGS 84 (latitude, longitude) projection

    # Convert SVY21 coordinates to WGS 84 geolocation
    print("Converting points to WGS84 format...")
    wgs84_points = itransform(svy21_proj, wgs84_proj, svy21_points)
    wgs84_points = np.array(list(wgs84_points))

    return wgs84_points


def save_coordinates(save_path, original_laz, svy21_points, preds, file_format="laz"):
    # convert coordinates
    wgs84_points = convert_svy21_to_wgs84(svy21_points)

    # save files
    if file_format == "laz":
        write_laz(save_path + "_WGS84.laz", original_laz, wgs84_points, preds)
    elif file_format == "txt":
        np.savetxt(save_path + "_SVY21.txt", svy21_points, delimiter=',', newline='\n')
        np.savetxt(save_path + "_WGS84.txt", wgs84_points, delimiter=',', newline='\n')


def get_main_base_center_coord(svy21_points):
    wgs84_points = convert_svy21_to_wgs84(svy21_points)
    return helper_json.get_center_base_coord(wgs84_points)


def convert_and_save_wgs84(save_path, svy21_points):
    # Add extension if not there
    if not save_path.endswith('_WGS84.laz'):
        save_path = save_path[:-4] + '_WGS84.laz'

    # convert coordinates
    wgs84_points = convert_svy21_to_wgs84(svy21_points)

    # save files
    write_sub_laz(save_path, wgs84_points)

    return helper_json.get_center_base_coord(wgs84_points)


def copy_predictions():
    print("Copying the predicted results to ftp...")
    from_directory = "av_randlanet_scfnet/results/"
    # from_file = os.path.join("av_randlanet_scfnet/results/%s/predictions/" % f.filename, f.filename[:-4] + ".laz")
    to_directory = "/home/pc1/shared"
    copy_tree(from_directory, to_directory)
    # shutil.copy(from_file, to_directory)


def save_segment_object_bc_coords(filename, all_objects):
    svy21_points = get_laz_points(os.path.join('av_randlanet_scfnet/results/%s/predictions/' % filename, filename))

    final_json = {
        'json': all_objects,
        'origin': get_main_base_center_coord(svy21_points)
    }

    # Save all objects as a single JSON file

    # Add extension if not there
    save_filename = filename[:-4] + '.json'
    if not filename.endswith('.laz'):
        save_filename = filename + '.json'

    # save_objects_json(output_file, all_objects)
    out_pth = os.path.join('av_randlanet_scfnet/results/%s/predictions/' % filename, save_filename)
    helper_json.save_objects_json(out_pth, final_json)
