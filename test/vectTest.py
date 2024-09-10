from os.path import join
# from networks.RandLANet_UNext_mds import Network
from UNext.networks.RandLANet_UNext_inf import Network
from UNext.tester_infer import ModelTester
from UNext.vectorize import bbox_to_shp, update_shp, merge_shp
from UNext.vectorizeRM import RM_bbox_to_shp
from UNext.utils.helper_ply import read_ply
from UNext.utils.helper_tool import ConfigOrchardRoad as cfg0, Config_UN_G1 as cfg1, Config_UN_G2 as cfg2,  ConfigRM_UN_bin as cfg3, ConfigRM_UN as cfg4
from UNext.utils.helper_tool import DataProcessing as DP
from UNext.utils.helper_tool import Plot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time, pickle, os
from UNext.utils import data_prepare
from UNext.utils.helper_las import gen_fullRM_input, gen_RMinput

name_dict = {   0: 'Pole',
                1: 'LampPost',
                2: 'Bollard',
                3: 'TrafficLight',
                4: 'Hydrant',
                5: 'ZebraBeaconPole',
                6: 'Tree',
                7: 'Shrub',
                8: 'TrashBin',
                9: 'ControlBox',
                10: 'Barrier',
                11: 'Railing',
                12: 'Building',
                13: 'BusStop',
                14: 'Ground',
                15: 'Road',
                16: 'Sign',
                17: 'SolarPanel',
                18: 'Parapet',
                19: 'CoveredLinkway',
                20: 'Pathway',
                21: 'PedestrianOverheadBridge',
                22: 'RetainingWall' 
}


def shape_Lp(files, upload_path, download_path):
    filepaths =[]
    for f in files:
        filepaths.append(os.path.join('UNext/test_inputs/', f.filename))
    res_folders = merge_shp(filepaths, name_dict, download_path)
    filepath = gen_fullRM_input(res_folders, upload_path)
    print(f'Road.laz saved to {filepath}')


if __name__=='__main__':
    uploadpath = '/home/steven/Desktop/git/GeoAI_UNext/UNext/test_inputs/Bishan'
    path = ' /home/steven/Desktop/'
