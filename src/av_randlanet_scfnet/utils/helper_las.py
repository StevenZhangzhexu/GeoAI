import os
import laspy
import numpy as np
from pyproj import Proj, itransform
import shutil
from distutils.dir_util import copy_tree


def read_las(pcd_filepath):
    # Add extension if not there
    if not pcd_filepath.endswith('.laz') or not pcd_filepath.endswith('.las'):
        if pcd_filepath.endswith('.ply'):
            pcd_filepath = str(pcd_filepath).replace('input_0.060', 'original_las').replace('.ply', '.laz')

    las_reader = laspy.read(pcd_filepath)
    print("File loaded from:", pcd_filepath)

    return las_reader


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

    las_writer.pred = preds

    las_writer.write(save_filepath)
    print("Prediction in .laz saved in path:", save_filepath)


def save_coordinates(save_dir, filename, svy21_points):
    # Define the SVY21 projection
    svy21_proj = Proj(init='epsg:3414')  # SVY21 Projection

    # Define the WGS 84 projection for geolocation
    wgs84_proj = Proj(init='epsg:4326')  # WGS 84 (latitude, longitude) projection

    # Convert SVY21 coordinates to WGS 84 geolocation
    print("Converting points to WGS84 format...")
    wgs84_points = itransform(svy21_proj, wgs84_proj, svy21_points)
    wgs84_points = np.array(list(wgs84_points))

    # save files
    np.savetxt(os.path.join(save_dir, filename + "_SVY21.txt"), svy21_points, delimiter=',', newline='\n')
    np.savetxt(os.path.join(save_dir, filename + "_WGS84.txt"), wgs84_points, delimiter=',', newline='\n')


def copy_predictions():
    print("Copying the predicted results to ftp...")
    from_directory = "av_randlanet_scfnet/results/"
    # from_file = os.path.join("av_randlanet_scfnet/results/%s/predictions/" % f.filename, f.filename[:-4] + ".laz")
    to_directory = "/home/pc1/shared"
    copy_tree(from_directory, to_directory)
    # shutil.copy(from_file, to_directory)
