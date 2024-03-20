from av_randlanet_scfnet.utils.helper_tool import DataProcessing as DP
from av_randlanet_scfnet.utils.helper_las import read_las
from av_randlanet_scfnet.utils.helper_ply import write_ply
from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os
import pickle
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)


# def prepare_data(pc_path, grid_size=0.06, dataset_path='av_randlanet_scfnet/data/orchard_road'):
def prepare_data(pc_path, grid_size=0.06, dataset_path='av_randlanet_scfnet\data\orchard_road'):
    # UTM_OFFSET = [627285, 4841948, 0]
    # original_pc_folder = join(dataset_path, 'original_las')
    sub_pc_folder = join(dataset_path, 'input_{:.3f}'.format(grid_size))
    os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    pc = read_las(pc_path)
    xyz = np.vstack((pc.x, pc.y, pc.z)).T.astype(np.float32)
    try:
        color = np.vstack((pc.red, pc.green, pc.blue)).T.astype(np.uint8)
    except:
        color = np.zeros_like(xyz).astype(np.uint8)
    try:
        intensity = pc.intensity.astype(np.uint8).reshape(-1, 1)
    except:
        intensity = np.zeros_like(pc.x).astype(np.uint8).reshape(-1, 1)

    #  Subsample to save space
    sub_xyz, sub_colors = DP.grid_sub_sampling(
        xyz, features=color, labels=None, grid_size=grid_size)
    _, sub_intensity = DP.grid_sub_sampling(
        xyz, features=intensity, labels=None, grid_size=grid_size)

    sub_colors = sub_colors / 255.0
    sub_intensity = sub_intensity[:, 0] / 255.0
    sub_ply_file = join(sub_pc_folder, file_name + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_intensity], [
              'x', 'y', 'z', 'red', 'green', 'blue', 'intensity'])

    search_tree = KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx], f)
