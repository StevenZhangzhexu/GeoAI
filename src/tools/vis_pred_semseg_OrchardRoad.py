#!/usr/bin/env python
import os
import sys
import logging
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import numpy as np
from os.path import join
import laspy


open3d_pythonpath = "/home/pc1/miniconda3/envs/open3d/bin/python"
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
        points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()

        data = {
            'name': "OrchardRoad_RandLAnet",
            'points': points,
            # 'feat': pcd.intensity,
            'pred': pcd.pred,
        }
        pc_data.append(data)

    return pc_data


# --------------semantic segmentation----------------


def viz_pred_semseg(filename):
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(orchard_labels.keys()):
        lut.add_label(orchard_labels[val], val)
    v.set_lut("pred", lut)

    # chosen_folder = 'results/Orchard_0913_labelled_E.laz/predictions/'
    chosen_folder = f'av_randlanet_scfnet/results/{filename}/predictions/'
    # pc_names = ["Orchard_0913_labelled_E.laz"]
    pc_names = [filename]
    pcs_with_pred = prepare_custom_data(pc_names, chosen_folder)

    print("Visualizing predictions...")
    v.visualize(pcs_with_pred)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    viz_pred_semseg(sys.argv[1])
