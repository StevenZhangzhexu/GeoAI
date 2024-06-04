import laspy
import numpy as np
from pyproj import Proj, itransform
from distutils.dir_util import copy_tree


def read_las(pcd_filepath):
    # Add extension if not there
    if not pcd_filepath.endswith('.laz') or not pcd_filepath.endswith('.las'):
        if pcd_filepath.endswith('.ply'):
            pcd_filepath = str(pcd_filepath).replace('input_0.060', 'original_las').replace('.ply', '.laz')

    las_data = laspy.read(pcd_filepath)
    print("File loaded from:", pcd_filepath)

    return las_data


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
    attributes_input = list(original_las.point_format.dimension_names)
    # print(attributes_input)

    # Transfer attributes from split_laz to las_writer
    for attr_name in attributes_input:
        setattr(las_writer, attr_name, getattr(original_las, attr_name))

    las_writer.label = labels
    las_writer.pred = preds

    las_writer.write(save_filepath)
    print("Prediction in .laz saved in path:", save_filepath)


# def write_laz_inf(save_filepath, original_las, points, preds):
#     # Add extension if not there
#     if not save_filepath.endswith('.laz'):
#         save_filepath = save_filepath + '.laz'

#     # Define a list of attributes to transfer
#     attributes_input = list(original_las.point_format.dimension_names)
#     # print(attributes_input)

#     # 1. Create a new header
#     header = laspy.LasHeader(point_format=3, version="1.2")
#     header.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.int32))
#     if 'label' in attributes_input:
#         header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))
#     header.offsets = np.min(points[:, :3], axis=0)
#     header.scales = np.array([0.1, 0.1, 0.1])

#     # 2. Create a Las
#     las_writer = laspy.LasData(header)

#     # Transfer attributes from split_laz to las_writer
#     for attr_name in attributes_input:
#         setattr(las_writer, attr_name, getattr(original_las, attr_name))

#     las_writer.x = points[:, 0]
#     las_writer.y = points[:, 1]
#     las_writer.z = points[:, 2]
#     if 'label' in attributes_input:
#         las_writer.label = points[:, 3]
#     las_writer.pred = preds

#     las_writer.write(save_filepath)
#     print("Prediction in .laz saved in path:", save_filepath)

def update_input(original_las, points, mask, file_path):
    attributes_input = list(original_las.point_format.dimension_names)
    header = laspy.LasHeader(point_format=3, version="1.2")
    if 'label' in attributes_input:
        header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))
    header.offsets = np.min(points[mask, :3], axis=0)
    print('offset shape', header.offsets.shape)
    header.scales = np.array([0.1, 0.1, 0.1])
    las_writer = laspy.LasData(header)
    # print(las_writer.x.shape)
    for attr_name in attributes_input:
        arr = getattr(original_las, attr_name)
        setattr(las_writer, attr_name, arr[mask])

    print('shape check')
    pts = points[mask, :3]
    print(pts.shape)
    print(las_writer.x.shape)

    # num_points = np.sum(mask)
    # las_writer.x = np.zeros(num_points, dtype=np.float64)
    # las_writer.y = np.zeros(num_points, dtype=np.float64)
    # las_writer.z = np.zeros(num_points, dtype=np.float64)
    # print('shape check after init')
    # print(las_writer.x.shape)

    las_writer.x = points[mask, 0]
    las_writer.y = points[mask, 1]
    las_writer.z = points[mask, 2]
    if 'label' in attributes_input:
        las_writer.label = points[mask, 3]
    las_writer.write(file_path)
    print("Unclassified points updated in path:", file_path)

def get_output(original_las, points, preds, mask, move):
    #######################
    update_dict = {0: {4:14, 8:15, 7:11}, 1:{}, 2:{0: 12, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, 9: 21, 10: 22}}
    #######################
    attributes_input = list(original_las.point_format.dimension_names)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.int32))
    if 'label' in attributes_input:
        header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))
    header.offsets = np.min(points[mask, :3], axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])
    las_writer = laspy.LasData(header)
    # Transfer attributes from split_laz to las_writer
    # for attr_name in attributes_input:
    #     setattr(las_writer, attr_name, getattr(original_las, attr_name))

    las_writer.x = points[mask, 0]
    las_writer.y = points[mask, 1]
    las_writer.z = points[mask, 2]
    if 'label' in attributes_input:
        las_writer.label = points[mask, 3]
    las_writer.pred =  np.array([update_dict[move].get(k,k) for k in preds[mask]])
    print(f'move:{move}, pred set:{set(las_writer.pred)}')
    print("Pred points saved")
    return las_writer

def write_output(full_lasdata, output_file_path):
    if not output_file_path.endswith('.laz'):
        output_file_path = output_file_path + '.laz'
    attributes_input = list(full_lasdata[0].point_format.dimension_names)
    if 'label' in attributes_input:
        labels = []
    points = []
    preds = []
    for las_data in full_lasdata:
        points.append(np.vstack((las_data.x, las_data.y, las_data.z)).T)
        if 'label' in attributes_input:
            labels.append(las_data.label)
        preds.append(las_data.pred)
    if not points:
        print(f"no points to write")
        return

    # Merge the point clouds and labels
    merged_points = np.concatenate(points)
    if 'label' in attributes_input:
        merged_labels = np.concatenate(labels)
    merged_preds = np.concatenate(preds)

    header = laspy.LasHeader(point_format=2, version="1.2")
    if 'label' in attributes_input:
        header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="pred", type=np.int32))
    header.offsets = np.min(merged_points, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])
    las_writer = laspy.LasData(header)
    las_writer.x = merged_points[:, 0]
    las_writer.y = merged_points[:, 1]
    las_writer.z = merged_points[:, 2]
    if 'label' in attributes_input:
        las_writer.label = merged_labels
    las_writer.pred = merged_preds
    las_writer.write(output_file_path)
    return las_writer

def update_laz_inf( file_path, points, preds, move):
    original_las = read_las(file_path)
    if move==0:
        mask = (preds != 7)
        print('len(points)', len(points))
        print('len(preds)', len(preds))
        print('len(mask)', len(mask))
        update_input(original_las, points, mask, file_path)
        mask = np.isin(preds, (4,7,8))
        lazdata = get_output(original_las, points, preds, mask, move)
    elif move==1:
        mask = (preds == 12)
        print('len(points)', len(points))
        print('len(preds)', len(preds))
        print('len(mask)', len(mask))
        update_input(original_las, points, mask, file_path)
        mask = (preds != 12)
        lazdata = get_output(original_las, points, preds, mask, move)
    else:
        mask = (preds == 11)
        print('len(points)', len(points))
        print('len(preds)', len(preds))
        print('len(mask)', len(mask))        
        update_input(original_las, points, mask, file_path)
        mask = (preds != 11)
        lazdata = get_output(original_las, points, preds, mask, move) 

    return lazdata



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
    from_directory = "UNext/results/"
    to_directory = "/home/pc1/shared"
    copy_tree(from_directory, to_directory)
