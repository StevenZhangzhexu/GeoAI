import laspy
import numpy as np
# from pyproj import Proj, itransform
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


def write_laz(save_filepath, original_las, points, labels, preds):
    # Add extension if not there
    if not save_filepath.endswith('.laz'):
        save_filepath = save_filepath[:-4] + '.laz'

    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))
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

    las_writer.label = labels
    las_writer.pred = preds

    las_writer.write(save_filepath)
    print("Prediction in .laz saved in path:", save_filepath)


def write_laz_inf(save_filepath, original_las, points, preds):
    # Add extension if not there
    if not save_filepath.endswith('.laz'):
        save_filepath = save_filepath + '.laz'

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
    # WGS 84 (latitude, longitude) projection
    wgs84_proj = Proj(init='epsg:4326')

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
        np.savetxt(save_path + "_SVY21.txt", svy21_points,
                   delimiter=',', newline='\n')
        np.savetxt(save_path + "_WGS84.txt", wgs84_points,
                   delimiter=',', newline='\n')




def copy_predictions():
    print("Copying the predicted results to ftp...")
    from_directory = "av_randlanet_scfnet/results/"
    # from_file = os.path.join("av_randlanet_scfnet/results/%s/predictions/" % f.filename, f.filename[:-4] + ".laz")
    to_directory = "/home/pc1/shared"
    copy_tree(from_directory, to_directory)
    # shutil.copy(from_file, to_directory)


