from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from UNext.utils.helper_tool import DataProcessing as DP
from UNext.utils import helper_tf_util
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        # if self.config.saving:
        #     if self.config.saving_path is None:
        #         self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        #     else:
        #         self.saving_path = self.config.saving_path
        #     makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 1]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 2]


            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.loss_type = 'sqrt'  # wce, lovas

            # self.class_weights = DP.get_class_weights(dataset.name, dataset.num_per_class, self.loss_type)
            self.class_weights = DP.get_class_weights(dataset.name)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits, self.sup_list = self.inference(self.inputs, self.is_training)

        # #####################################################################
        # # Ignore the invalid point (unlabeled) when calculating the loss #
        # #####################################################################
        # with tf.variable_scope('loss'):
        #     self.logits = tf.reshape(self.logits, [-1, config.num_classes])

        # with tf.variable_scope('optimizer'):
        #     self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
        #     self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # with tf.variable_scope('results'):
        #     self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #     self.prob_logits = tf.nn.softmax(self.logits)

        #     tf.summary.scalar('learning_rate', self.learning_rate)
        #     tf.summary.scalar('loss', 0)
        #     tf.summary.scalar('accuracy', 0)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        jin = []
        feature_list = []
        sup_list = [[], [], [], []]

        for j in range(self.config.num_layers):
            if j == 0:
                feature_list = []
                for i in range(self.config.num_layers - j):
                    # encoding[(i, j-1)]
                    f_encoder_i_j = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(j) + str(i), is_training)
                    f_sampled_i_j = self.random_sample(f_encoder_i_j, inputs['sub_idx'][i])
                    feature_list.append(f_encoder_i_j)
                    feature = f_sampled_i_j
                jin.append(feature_list)

            if 0 < j < self.config.num_layers - 1:
                feature_list = []
                for i in range(self.config.num_layers - j):
                    if i == 0:
                        # encoding[(i-1, j); (i-1, j+1)]
                        f_1 = jin[j-1][i]
                        f_2 = self.nearest_interpolation(jin[j-1][i+1], inputs['interp_idx'][i])
                        if j != self.config.num_layers - 1:
                            feature = tf.concat([f_1, f_2], axis=3)
                        if j == self.config.num_layers - 1:
                            f_3 = jin[0][i]
                            feature = tf.concat([f_1, f_2, f_3], axis=3)
                        f_decoder_i_j = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(j) + str(i), is_training)

                        f_sup = helper_tf_util.conv2d(f_decoder_i_j, self.config.num_classes, [1, 1], 'sup_' + str(j) + str(i), [1, 1], 'VALID', False, is_training, activation_fn=None)
                        f_sup = tf.squeeze(f_sup, [2])
                        sup_list[i].append(f_sup)

                        f_sampled_i_j = self.random_sample(f_decoder_i_j, inputs['sub_idx'][i])
                        feature_list.append(f_decoder_i_j)
                        feature = f_sampled_i_j

                    if i >= 1:
                        # encoding[(i, j-1); (i-1, j); (i-1, j+1)]
                        f_0 = feature
                        f_1 = jin[j - 1][i]
                        f_2 = self.nearest_interpolation(jin[j - 1][i + 1], inputs['interp_idx'][i])
                        feature = tf.concat([f_0, f_1, f_2], axis=3)
                        if j + i == self.config.num_layers - 1 and i != 3:
                            # encoding[(i, j-1); (i-1, j); (i-1, j+1); (0, j)]
                            f_3 = jin[0][i]
                            feature = tf.concat([feature, f_3], axis=3)
                        f_decoder_i_j = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(j) + str(i), is_training)

                        f_sup = helper_tf_util.conv2d(f_decoder_i_j, self.config.num_classes, [1, 1], 'sup_' + str(j) + str(i), [1, 1], 'VALID', False, is_training, activation_fn=None)
                        f_sup = tf.squeeze(f_sup, [2])
                        sup_list[i].append(f_sup)

                        f_sampled_i_j = self.random_sample(f_decoder_i_j, inputs['sub_idx'][i])
                        feature_list.append(f_decoder_i_j)
                        feature = f_sampled_i_j

                jin.append(feature_list)

            if j == self.config.num_layers - 1:  # The last Node
                feature_list = []
                i = 0
                # encoding[(i-1, j); (i-1, j+1)]
                f_1 = jin[j - 1][i]
                f_2 = self.nearest_interpolation(jin[j - 1][i + 1], inputs['interp_idx'][i])
                f_3 = jin[0][i]
                feature = tf.concat([f_1, f_2, f_3], axis=3)
                f_decoder_i_j = helper_tf_util.conv2d_transpose(feature, d_out[i], [1, 1], 'Decoder_layer_' + str(j) + str(i), [1, 1], 'VALID', bn=True, is_training=is_training)

                f_sup = helper_tf_util.conv2d(f_decoder_i_j, self.config.num_classes, [1, 1], 'sup_' + str(j) + str(i), [1, 1], 'VALID', False, is_training, activation_fn=None)
                f_sup = tf.squeeze(f_sup, [2])
                sup_list[i].append(f_sup)

                f_sampled_i_j = self.random_sample(f_decoder_i_j, inputs['sub_idx'][i])
                feature_list.append(f_decoder_i_j)
                feature = f_sampled_i_j

                jin.append(feature_list)

        f_layer_fc1 = helper_tf_util.conv2d(jin[-1][-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False, is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, sup_list


    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
