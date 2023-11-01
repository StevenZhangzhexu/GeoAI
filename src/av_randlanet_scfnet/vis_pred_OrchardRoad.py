#!/usr/bin/env python
import logging
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import numpy as np
import os
from os.path import join
import laspy

log = logging.getLogger(__name__)


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
            'feat': pcd.intensity,
            'labels': pcd.label,
            'pred': pcd.pred,
        }
        pc_data.append(data)

    return pc_data


# ------------------------------


def main():
    orchard_labels = {
                        0: 'Bollard',
                        1: 'Building',
                        2: 'Bus Stop',
                        3: 'Control Box',
                        4: 'Ground',
                        5: 'Lamp Post',
                        6: 'Pole',
                        7: 'Railing',
                        8: 'Road',
                        9: 'Shrub',
                        10: 'Sign',
                        11: 'Solar Panel',
                        12: 'Tree'
                    }
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(orchard_labels.keys()):
        lut.add_label(orchard_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    logs = np.sort([os.path.join('test', f) for f in os.listdir('test') if f.startswith('Log')])
    chosen_folder = logs[-1]
    pred_path = os.path.join(chosen_folder, 'predictions')
    pc_names = ["Orchard_0913_labelled_E.laz"]
    pcs_with_pred = prepare_custom_data(pc_names, pred_path)

    print("Visualizing predictions...")
    v.visualize(pcs_with_pred)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
