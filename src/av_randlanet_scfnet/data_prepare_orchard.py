from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, pickle
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils.helper_ply import write_ply
from utils.helper_las import read_las
from utils.helper_tool import DataProcessing as DP


grid_size = 0.06
original_dataset_path = '../data/Takashimaya/orchard_road'
processed_dataset_path = 'data/orchard_road'
train_files = ['Orchard_0913_labelled_A', 'Orchard_0913_labelled_B', 'Orchard_0913_labelled_C', 'Orchard_0913_labelled_D']
val_files = ['Orchard_0913_labelled_E']
# UTM_OFFSET = [627285, 4841948, 0]
# original_pc_folder = join(dataset_path, 'original_las')
original_pc_folder = original_dataset_path
sub_pc_folder = join(processed_dataset_path, 'input_{:.3f}'.format(grid_size))
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

for pc_path in [join(original_pc_folder, fname + '.laz') for fname in train_files + val_files]:
    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    pc = read_las(pc_path)
    # labels = pc['label'].astype(np.uint8)
    labels = pc.label.astype(np.uint8)
    # xyz = np.vstack((pc['x'] - UTM_OFFSET[0], pc['y'] - UTM_OFFSET[1], pc['z'] - UTM_OFFSET[2])).T.astype(np.float32)
    xyz = np.vstack((pc.x, pc.y, pc.z)).T.astype(np.float32)
    color = np.vstack((pc.red, pc.green, pc.blue)).T.astype(np.uint8)
    intensity = pc.intensity.astype(np.uint8).reshape(-1, 1)
    #  Subsample to save space
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, color, labels, grid_size)
    _, sub_intensity = DP.grid_sub_sampling(xyz, features=intensity, grid_size=grid_size)

    sub_colors = sub_colors / 255.0
    sub_intensity = sub_intensity[:, 0] / 255.0
    sub_ply_file = join(sub_pc_folder, file_name + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_intensity, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class'])

    search_tree = KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    if file_name not in train_files:
        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)
