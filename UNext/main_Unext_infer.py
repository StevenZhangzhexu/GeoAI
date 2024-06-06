from os.path import join
# from networks.RandLANet_UNext_mds import Network
from UNext.networks.RandLANet_UNext_inf import Network
from UNext.tester_infer import ModelTester
from UNext.vectorize import bbox_to_shp, update_shp, merge_shp
from UNext.utils.helper_ply import read_ply
from UNext.utils.helper_tool import ConfigOrchardRoad as cfg0, Config_UN_G1 as cfg1, Config_UN_G2 as cfg2
from UNext.utils.helper_tool import DataProcessing as DP
from UNext.utils.helper_tool import Plot
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time, pickle, os
from UNext.utils import data_prepare


class OrchardRoad:
    def __init__(self, filepath, uploadpath, move, cfg):
        self.name = 'OrchardRoad'
        self.move = move
        self.cfg = cfg
        self.path = uploadpath
        self.label_to_names_dict = {
            0:  { # original 13 classes
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
                },
            1:  { # Group1
                0: 'Pole',
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
                12: 'Unclassified'
                },
            2:  { # Group2
                0: 'Building',
                1: 'BusStop',
                2: 'Ground',
                3: 'Road',
                4: 'Sign',
                5: 'SolarPanel',
                6: 'Parapet',
                7: 'CoveredLinkway',
                8: 'Pathway',
                9: 'PedestrianOverheadBridge',
                10: 'RetainingWall',
                11: 'Unclassified'
                }
        }
        self.label_to_names = self.label_to_names_dict[move]
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k in self.label_to_names.keys()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([])

        self.test_files = [filepath]

        # Initiate containers
        self.test_proj = []
        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'pred': []}
        self.input_colors = {'pred': []}
        self.load_sub_sampled_clouds(self.cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = self.test_files

        for i, file_path in enumerate(files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            cloud_split = 'pred'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            cfg = self.cfg
            # read RGB / intensity accoring to configuration
            if cfg.use_rgb and cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'], data['intensity'])).T
            elif cfg.use_rgb and not cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            elif not cfg.use_rgb and cfg.use_intensity:
                sub_colors = data['intensity'].reshape(-1,1)
            else:
                sub_colors = np.ones((data.shape[0],1))


            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        
        t0 = time.time()
        cloud_name = file_path.split('/')[-1][:-4]

        proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
        with open(proj_file, 'rb') as f:
            proj_idx = pickle.load(f)
        self.test_proj += [proj_idx]
        print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split='pred'):
        cfg = self.cfg
        num_per_epoch = cfg.val_steps * cfg.val_batch_size

        # assign number of features according to input
        n_features = 1  # use xyz only by default
        if cfg.use_rgb and cfg.use_intensity:
            n_features = 4
        elif cfg.use_rgb and not cfg.use_intensity:
            n_features = 3

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]

                #debug
                # print("queried_pc_xyz:", queried_pc_xyz.shape)
                # print("queried_pc_colors:", queried_pc_colors.shape)


                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))


        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, n_features], [None], [None])
        return gen_func, gen_types, gen_shapes
    
    # data augmentation
    def tf_augment_input(self,inputs):
        cfg = self.cfg
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random.uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random.uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random.uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random.uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random.normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        stacked_features = tf.concat([transformed_xyz, features], axis=-1)
        return stacked_features

    def get_tf_mapping2(self):
        # Collect flat inputs
        cfg = self.cfg
        def tf_map(batch_xyz, batch_features, batch_pc_idx, batch_cloud_idx):
            #batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points


            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating prediction pipelines')
        cfg = self.cfg
        cfg.ignored_label_inds = [self.label_to_idx[ign_label]
                                  for ign_label in self.ignored_labels]
        gen_function_test, gen_types, gen_shapes = self.get_batch_gen('pred')

        self.test_data = tf.data.Dataset.from_generator(
            gen_function_test, gen_types, gen_shapes)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_test_data = self.batch_test_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.prefetch(
            cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(
            self.batch_test_data.output_types, self.batch_test_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.test_init_op = iter.make_initializer(self.batch_test_data)


def predict(filepath,uploadpath):
    print("Starting prediction...")

    start_time = time.time()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    file_name = filepath.split('/')[-1]
    if 'laz' in file_name:
        file_name = file_name[:-4]
    folder_name = filepath.split('/')[-2]
    cfgs = (cfg0, cfg1, cfg2)
    full_lasdata = []

    for move in range(3): 
        data_prepare.prepare_data(pc_path=filepath, dataset_path=uploadpath)
        tf.reset_default_graph()
        cfg = cfgs[move]
        dataset = OrchardRoad(filepath, uploadpath, move, cfg)
        dataset.init_input_pipeline()

        snap_path = f'UNext/checkpoints/snapshots{move}'
        snap_steps = [int(f[:-5].split('-')[-1])
                    for f in os.listdir(snap_path) if f[-5:] == '.meta']
        chosen_step = np.sort(snap_steps)[-1]
        chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))

        model = Network(dataset, cfg)

        tester = ModelTester(model, dataset, cfg, folder_name, file_name,
                            restore_snap=chosen_snap, move=move)
        lasdata = tester.infer(model, dataset, id=move)
        full_lasdata.append(lasdata)

        if move==0:
            generate_shp(file_name, tester.saving_path, move, lasdata) # shape files
        if move ==2:
            tester.write_out(full_lasdata) # final ouput
            generate_shp(file_name, tester.saving_path, move)

    print("Prediction finished!")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of {file_name}: ", execution_time)

    # return tester.saving_path

def generate_shp(filename, saving_path, move, lasdata=None):
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

    chosen_folder = saving_path
    if 'laz' in filename:
        filename = filename[:-4]
    if move==0:
        bbox_to_shp(filename = filename, name_dict = name_dict, output_folder = chosen_folder, las_data=lasdata)
    else:
        bbox_to_shp(filename = filename, name_dict = name_dict, restore=True, output_folder = chosen_folder)
        update_shp(output_folder = chosen_folder)
    

def shape_output(files, download_path):
    filepaths =[]
    for filename in files:
        filepaths.append(os.path.join('UNext/test_inputs/', filename))

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
    merge_shp(filepaths, name_dict, download_path)