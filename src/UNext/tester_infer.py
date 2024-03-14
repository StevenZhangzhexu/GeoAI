from os import makedirs
from os.path import exists, join
from UNext.utils.helper_las import read_las, write_laz_inf #, save_coordinates
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()
import numpy as np
import time


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, config, filename, restore_snap=None):
        self.filename = filename
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # print("Shapes of tensors before creating Saver:")
        # for var in my_vars:
        #     print(var.name, var.shape)

        self.saver = tf.train.Saver(my_vars, max_to_keep=100)


        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                           for l in dataset.input_trees['pred']]

        self.config = config
        # Test saving path
        self.saving_path = time.strftime('UNext/results/Pred_%Y-%m-%d_%H-%M-%S', time.gmtime())
        makedirs(self.saving_path) if not exists(self.saving_path) else None
        # saving_path = 'UNext/results/%s/predictions/' % filename
        # self.saving_path = saving_path
        # makedirs(saving_path, exist_ok=True)
        log_file_path = join( self.saving_path+'/' , 'log_test.txt')
        if not os.path.exists(log_file_path):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_out = open(log_file_path, 'a')

    def chosen_folder_path(self):
        return self.saving_path
        
    def infer(self, model, dataset, num_votes=100, id=0):
        filename = self.filename

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5
        t0 = time.time()

        log_string(f'Prediction of Model {id}', self.log_out)

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)
                # print(ops)

                stacked_probs, point_idx, cloud_idx = self.sess.run(
                    ops, {model.is_training: False})
                # Debugging: Print shapes of tensors
                # print("Shape of stacked_probs:", stacked_probs.shape)
                # print("Shape of point_idx:", point_idx.shape)
                # print("Shape of cloud_idx:", cloud_idx.shape)
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * \
                        self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['pred'])), self.log_out)

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['pred'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(
                    epoch_id, new_min), self.log_out)

                if last_min + 1 < new_min:

                    print('Prediction done in {:.1f} s\n'.format(
                        time.time() - t0))
                    print('Saving clouds')

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(
                        int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        points = self.load_test_points(file_path)

                        # Reproject probs
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :][0]
                        # print(probs.shape)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(
                            probs, axis=1)].astype(np.uint8)

                        # Save laz
                        test_las = read_las(file_path)
                        pred_filepath = join(self.saving_path, filename)
                        write_laz_inf(pred_filepath, test_las, points, preds)
                        # save_coordinates(
                        #     pred_filepath[:-4], test_las, points, preds)
                        log_string(pred_filepath +
                                   ' has been saved.', self.log_out)

                        i_test += 1

                    t2 = time.time()
                    print(
                        'Reprojection and saving done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_test_points(file_path):
        data = read_las(file_path)
        return np.vstack((data.x, data.y, data.z)).T