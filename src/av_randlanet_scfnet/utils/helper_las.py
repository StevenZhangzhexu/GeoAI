import laspy
import numpy as np


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
