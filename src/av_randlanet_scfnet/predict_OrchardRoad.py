from os.path import join
from av_randlanet_scfnet.RandLANet import Network
# from SCFNet import Network
from av_randlanet_scfnet.tester_OrchardRoad import ModelTester
from av_randlanet_scfnet.utils.helper_tool import ConfigOrchardRoad as cfg
from av_randlanet_scfnet.utils.helper_tool import DataProcessing as DP
from av_randlanet_scfnet.utils.helper_ply import read_ply

import tensorflow as tf
import numpy as np
import pickle, os

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class OrchardRoad:
    def __init__(self, filepath=''):
        self.name = 'OrchardRoad'
        self.path = 'av_randlanet_scfnet/data/orchard_road'
        self.label_to_names = {
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
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([])

        self.test_pc_folder = join(self.path, 'test_inputs')

        # Initial training-validation-testing files
        self.test_inputs = [filepath.split("/")[-1]]
        # self.test_files = [os.path.join(self.test_pc_folder, files) for files in self.test_inputs]
        self.test_files = [filepath]

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'predict': []}
        self.input_colors = {'predict': []}

        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        files = self.test_files

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            cloud_split = 'predict'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            # read RGB / intensity accoring to configuration
            if cfg.use_rgb and cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'], data['intensity'])).T
            elif cfg.use_rgb and not cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            elif not cfg.use_rgb and cfg.use_intensity:
                sub_colors = data['intensity'].reshape(-1, 1)
            else:
                sub_colors = np.ones((data.shape[0],1))

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]

            # Get test re_projection indices
            print('\nPreparing reprojection indices for {}'.format(cloud_name))
            proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
            with open(proj_file, 'rb') as f:
                proj_idx = pickle.load(f)
            self.test_proj += [proj_idx]

        print('finished')
        return

    # Generate the input data flow
    def get_batch_gen(self, split="predict"):
        num_per_epoch = cfg.val_steps * cfg.val_batch_size
        
        # assign number of features according to input
        n_features = 1  # use xyz only by default
        if cfg.use_rgb and cfg.use_intensity:
            n_features = 4
        elif cfg.use_rgb and not cfg.use_intensity:
            n_features = 3

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

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
                query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                from utils.helper_tool import DataProcessing as DP
                query_idx = DP.shuffle_idx(query_idx)

                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[query_idx]
                queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]

                queried_pt_weight = 1

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                if True:
                    yield (queried_pc_xyz,
                           queried_pc_colors.astype(np.float32),
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, n_features], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.map_fn(self.tf_augment_input, [batch_xyz, batch_features], dtype=tf.float32)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neigh_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neigh_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neigh_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
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
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        stacked_features = tf.concat([transformed_xyz, features], axis=-1)
        return stacked_features

    def init_predict_pipeline(self):
        print('Initiating prediction pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function_test, gen_types, gen_shapes = self.get_batch_gen('predict')

        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_test_data.output_types, self.batch_test_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.test_init_op = iter.make_initializer(self.batch_test_data)


def predict(filepath):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    dataset = OrchardRoad(filepath)
    file_name = filepath.split('/')[-1]

    snap_path = 'av_randlanet_scfnet/checkpoints'
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
    chosen_step = np.sort(snap_steps)[-1]
    chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))

    dataset.init_predict_pipeline()
    model = Network(dataset, cfg, file_name)

    tester = ModelTester(model, dataset, cfg, file_name, restore_snap=chosen_snap)
    print("Starting prediction...")
    tester.test(model, dataset, file_name)
