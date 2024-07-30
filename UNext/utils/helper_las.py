import laspy
import numpy as np
import os
# from pyproj import Proj, itransform
# from distutils.dir_util import copy_tree


def read_las(pcd_filepath):
    # Add extension if not there
    if not pcd_filepath.endswith('.laz') or not pcd_filepath.endswith('.las'):
        if pcd_filepath.endswith('.ply'):
            pcd_filepath = str(pcd_filepath).replace('input_0.060', 'original_las').replace('.ply', '.laz')

    las_data = laspy.read(pcd_filepath)
    print("File loaded from:", pcd_filepath)

    return las_data

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

    # print('shape check')
    # pts = points[mask, :3]
    # print(pts.shape)
    # print(las_writer.x.shape)
    # print(las_writer.x.shape)
    print("Unclassified points is writing ...") #####
    las_writer.x = points[mask, 0]
    las_writer.y = points[mask, 1]
    las_writer.z = points[mask, 2]
    las_writer.red =  points[mask, 3]
    las_writer.green =  points[mask, 4]
    las_writer.blue =  points[mask, 5]
    las_writer.intensity =  points[mask, 6]
    if 'label' in attributes_input:
        las_writer.label = points[mask, 7]
    las_writer.write(file_path)
    print("Unclassified points updated in path:", file_path)

def get_output(original_las, points, preds, mask, move):
    #######################
    update_dict = {0: {4:14, 8:15, 7:11}, 1:{}, 2:{0: 12, 1: 13, 2: 14, 3: 15, 4: 16, 5: 17, 6: 18, 7: 19, 8: 20, 9: 21, 10: 22}}
    #######################
    if mask is None:
        mask = np.ones(points.shape[0], dtype=bool)
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
    las_writer.green =  points[mask, 4]
    las_writer.blue =  points[mask, 5]
    las_writer.intensity =  points[mask, 6]
    if 'label' in attributes_input:
        las_writer.label = points[mask, 7]
    if move<3:
        las_writer.pred =  np.array([update_dict[move].get(k,k) for k in preds[mask]])
    else:
        las_writer.pred = preds[mask]
    print(f'move:{move}, pred set:{set(las_writer.pred)}')
    # print('pred len', len(las_writer.pred), type(las_writer.pred))
    # print('x len', len(las_writer.x), type(las_writer.x))
    print("Pred points saved")
    return las_writer

def write_output(full_lasdata, output_file_path):
    if not output_file_path.endswith('.laz'):
        output_file_path = output_file_path + '.laz'
    attributes_input = list(full_lasdata[0].point_format.dimension_names)
    points = []
    labels = [] if 'label' in attributes_input else None
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

def update_laz_inf(file_path, points, preds, move):
    original_las = read_las(file_path)
    if move==0:
        # mask = (preds != 7)
        # print('len(points)', len(points))
        # print('len(preds)', len(preds))
        # print('len(mask)', len(mask))
        # update_input(original_las, points, mask, file_path)
        mask = np.isin(preds, (4,7,8))
        lazdata = get_output(original_las, points, preds, mask, move)
    elif move==1:
        mask = (preds == 12)
        # print('len(points)', len(points))
        # print('len(preds)', len(preds))
        # print('len(mask)', len(mask))
        update_input(original_las, points, mask, file_path)
        mask = (preds != 12)
        lazdata = get_output(original_las, points, preds, mask, move)
    elif move==2:
        mask = (preds == 11)
        # print('len(points)', len(points))
        # print('len(preds)', len(preds))
        # print('len(mask)', len(mask))        
        update_input(original_las, points, mask, file_path)
        mask = (preds != 11)
        lazdata = get_output(original_las, points, preds, mask, move) 
    elif move==3:
        mask = (preds == 0)
        # print('len(points)', len(points))
        # print('len(preds)', len(preds))
        # print('len(mask)', len(mask))        
        update_input(original_las, points, mask, file_path)
        return
    elif move ==4:
        mask = None
        lazdata = get_output(original_las, points, preds, mask, move)
 
    return lazdata

def gen_fullRM_input(res_folders, upload_path):
    '''
    read from res_folders, upload_path
    save at upload_path
    '''
    points = []
    intensities = []
    reds = []
    greens = []
    blues = []
    path = os.path.join(upload_path, f'Road.laz')
    for subfolder in res_folders:
        laz_files = [f for f in os.listdir(subfolder) if f.lower().endswith('.laz')]
        pc = laz_files[0]
        las_reader = laspy.read(pc)
        mask = (las_reader.labels == 8) # Road
        points.append(np.vstack((las_reader.x[mask], las_reader.y[mask], las_reader.z[mask])).T)
        reds.append(las_reader.red[mask])
        greens.append(las_reader.green[mask])
        blues.append(las_reader.blue[mask])
        intensities.append(las_reader.intensity[mask])

    if not points:
        print('---'*21)
        print('|',' '*25, 'Warning!',' '*25,'|')
        print("| No road points found. Road marking extraction is TERMINATED... |")
        print('---'*21)
        return
    else:
        merged_points = np.concatenate(points)
        merged_reds = np.concatenate(reds)
        merged_greens = np.concatenate(greens)
        merged_blues = np.concatenate(blues)
        merged_intensities = np.concatenate(intensities)
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = np.min(merged_points, axis=0)
        header.scales = np.array([0.1, 0.1, 0.1])
        las_writer = laspy.LasData(header)
        las_writer.x = merged_points[:, 0]
        las_writer.y = merged_points[:, 1]
        las_writer.z = merged_points[:, 2]
        las_writer.red = merged_reds
        las_writer.green = merged_greens
        las_writer.blue = merged_blues
        las_writer.intensity = merged_intensities
        las_writer.write(path)
        
    return path

def gen_RMinput(saving_path, file_name):
    pc = os.path.join(saving_path, file_name + '.laz')
    path = os.path.join(saving_path, f'Road.laz')
    las_reader = laspy.read(pc)
    points = []
    intensities = []
    reds = []
    greens = []
    blues = []
    mask = (las_reader.pred == 15) # Road
    points.append(np.vstack((las_reader.x[mask], las_reader.y[mask], las_reader.z[mask])).T)
    reds.append(las_reader.red[mask])
    greens.append(las_reader.green[mask])
    blues.append(las_reader.blue[mask])
    intensities.append(las_reader.intensity[mask])

    if not points:
        print('---'*21)
        print('|',' '*25, 'Warning!',' '*25,'|')
        print("| No road points found. Road marking extraction is disabled... |")
        print('---'*21)
        return
    else:
        merged_points = np.concatenate(points)
        merged_reds = np.concatenate(reds)
        merged_greens = np.concatenate(greens)
        merged_blues = np.concatenate(blues)
        merged_intensities = np.concatenate(intensities)
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = np.min(merged_points, axis=0)
        header.scales = np.array([0.1, 0.1, 0.1])
        las_writer = laspy.LasData(header)
        las_writer.x = merged_points[:, 0]
        las_writer.y = merged_points[:, 1]
        las_writer.z = merged_points[:, 2]
        las_writer.red = merged_reds
        las_writer.green = merged_greens
        las_writer.blue = merged_blues
        las_writer.intensity = merged_intensities
        las_writer.write(path)
    
    return path
    
