#!/usr/bin/env python
import os
import sys
import logging
import open3d.ml.tf as ml3d  # just switch to open3d.ml.tf for tf usage
import numpy as np
from os.path import join
import laspy
from bbox import bbox_pcd
import pickle


open3d_pythonpath = "/home/steven/miniconda3/envs/vis/bin/python"
sys.path.append(open3d_pythonpath)
os.environ["PYTHONPATH"] = open3d_pythonpath

try:
    sys.path.index(open3d_pythonpath)    # Or os.getcwd() for this directory
except ValueError:
    sys.path.append(open3d_pythonpath)    # Or os.getcwd() for this directory

log = logging.getLogger(__name__)

orchard_labels = {
    0: 'Bollard',
    1: 'Building',
    2: 'BusStop',
    3: 'ControlBox',
    4: 'Ground',
    5: 'LampPost',
    6: 'Pole',
    7: 'Railing',
    8: 'Road',
    9: 'Shrub',
    10: 'Sign',
    11: 'SolarPanel',
    12: 'Tree'
}


def prepare_custom_data(pc_names, path):
    print("Loading orchard evaluation data...")

    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(path, name)
        pcd = laspy.read(pc_path)
        if 'label' in list(pcd.header.point_format.dimension_names):
            points = np.vstack((pcd.x, pcd.y, pcd.z, pcd.label)).T 
            data = {
            'name': "OrchardRoad_RandLAnet",
            'points': points,
            'feat': pcd.intensity,
            'label': pcd.label,
            'pred': pcd.pred,
            }
        else:
            points = np.vstack((pcd.x, pcd.y, pcd.z)).T
            data = {
            'name': "OrchardRoad_RandLAnet",
            'points': points,
            'feat': pcd.intensity,
            'pred': pcd.pred,
            }
         

        
        pc_data.append(data)
    
    if  'label' in list(pcd.header.point_format.dimension_names):
        return pc_data, set(pcd.label), set(pcd.pred)


    return pc_data, set(), set(pcd.pred)


# --------------semantic segmentation----------------


def viz_pred_semseg(filename):

    results_dir = 'UNext/results'
    # List directories in UNext/results
    directories = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    # Sort directories by modification time (latest first)
    directories.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    # Take the first directory (latest one)
    latest_folder = directories[0]
    chosen_folder = os.path.join(results_dir, latest_folder)
    pc_names = [filename]
    pcs_with_pred , lbs, pds = prepare_custom_data(pc_names, chosen_folder)

    print("Visualizing Semseg predictions...")
    v = ml3d.vis.Visualizer()
    lut1 = ml3d.vis.LabelLUT()
    lut2 = ml3d.vis.LabelLUT()
    for val, label_name in orchard_labels.items():
        if val in lbs:
            lut1.add_label(label_name, val)
        if val in pds:
            lut2.add_label(label_name, val)
    v.set_lut("labels", lut1)
    v.set_lut("pred", lut2)
    v.visualize(pcs_with_pred)

    print("Visualizing Bounding Boxes...")
    path = os.path.join(chosen_folder,pc_names[0])
    bbox_dict = bbox_pcd(path)
    file_path = os.path.join(chosen_folder, 'bbox_dict.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(bbox_dict, file)

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    viz_pred_semseg(sys.argv[1])
