import tensorflow as tf
import numpy as np
import pysc2.my_agent.layer as layer
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'


class ProbeNetwork(object):

    def __init__(self):

        self.encoder_version = "basic"
        self.encoder_model_path = "model/encoder_%s/probe" % self.encoder_version

        self.action_type_trainable = False
        self.action_type_model_path = "model/action_type/probe"

        self.action_pos_trainable = True
        self.action_pos_model_path = "model/action_pos/probe"

        self.value_trainable = True
        self.value_model_path = "model/value_net/probe"

        self.rl_training = True
        self.rl_model_path = "model/rl/probe"

        self.summary = []

        self.map_width = 64
        self.map_num = 3
        self.action_num = 5

        self.encoder_lr = 0.00001
        self.action_type_lr = 0.00001
        self.action_pos_lr = 0.000001

        self.flatten_map_size = self.map_width * self.map_width * self.map_num

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self._create_graph()

            self.sess = tf.Session(graph=self.graph)
            self._define_saver()

            # tf.summary.FileWriter("logs/", self.sess.graph)

    def _define_saver(self):

        self.encoder_var_list_save = list(set(self.encoder_var_list_train +
                                              tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Encoder")))
        self.encoder_saver = tf.train.Saver(var_list=self.encoder_var_list_save)

        self.action_type_var_list_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Action_Type")
        self.action_type_saver = tf.train.Saver(var_list=self.action_type_var_list_save)

        self.action_pos_var_list_save = list(set(self.action_pos_var_list_train +
                                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Action_Pos")))
        self.action_pos_saver = tf.train.Saver(var_list=self.action_pos_var_list_save)

        self.rl_var_list_save = self.encoder_var_list_train + self.rl_var_list_train
        self.rl_var_list_save.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RL_loss"))
        self.rl_saver = tf.train.Saver(var_list=self.rl_var_list_save)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def _create_graph(self):

        # define learning rate
        self.encoder_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="encoder_lr")
        self.action_type_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="action_type_lr")
        self.action_pos_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="action_pos_lr")

        # define input
        self.map_data = tf.placeholder(dtype=tf.float32, shape=[None, self.map_num, self.map_width, self.map_width],
                                       name="input_map_data")
        self.action_type_label = tf.placeholder(dtype=tf.float32, shape=[None, self.action_num],
                                                name="Action_Type_label")
        self.action_pos_label = tf.placeholder(dtype=tf.float32, shape=[None, self.map_width*self.map_width],
                                               name="Action_Position_label")

        # define network structure
        self.encode_data = eval("self._encoder_%s(self.map_data)" % self.encoder_version)
        self.decode_data = eval("self._decoder_%s(self.encode_data)" % self.encoder_version)

        self.action_type_predict = self._action_type_net(self.encode_data)
        self.action_pos_predict = self._action_pos_net(self.encode_data, self.action_type_predict)

        # define network loss to train
        self._define_var_list_train()
        self._define_sl_loss()

        # RL start
        self.value = self._value_net(self.encode_data, self.action_type_predict)

        self.value_target = tf.placeholder(tf.float32, [None], name='value_target')
        self.action_type_selected = tf.placeholder(dtype=tf.float32, shape=[None, self.action_num],
                                                   name="Action_Type_selected")
        self.action_pos_selected = tf.placeholder(dtype=tf.float32, shape=[None, self.map_width*self.map_width],
                                                  name="Action_Position_selected")
        self.valid_action_type = tf.placeholder(tf.float32, [None, self.action_num], name='valid_action_type')
        self.valid_action_pos = tf.placeholder(tf.float32, [None], name='valid_action_pos')

        self.rl_lr_ph = tf.placeholder(tf.float32, None, name='rl_learning_rate')

        self._define_rl_loss()

    def _define_var_list_train(self):
        # encoder
        self.encoder_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
        self.encoder_var_list_train.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder"))

        # action_type
        self.action_type_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Type")

        # action_pos
        self.action_pos_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Pos")

        # value_net
        self.value_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Value_Net")

    def _define_sl_loss(self):
        # encoder loss
        # self.encoder_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
        # self.encoder_var_list_train.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder"))
        with tf.name_scope("Encoder_loss"):
            self.encoder_loss = tf.reduce_mean(tf.squared_difference(self.map_data, self.decode_data))
            self.encoder_train_step = tf.train.AdamOptimizer(self.encoder_lr_ph).minimize(
                self.encoder_loss, var_list=self.encoder_var_list_train)

        # action type loss
        # self.action_type_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Type")
        with tf.name_scope("Action_Type_loss"):
            self.action_type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.action_type_predict, labels=self.action_type_label))
            # for batch_norm training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.action_type_train_step = tf.train.AdamOptimizer(self.action_type_lr_ph).minimize(
                    self.action_type_loss, var_list=self.action_type_var_list_train)

            self.action_type_cor_pre = tf.equal(tf.argmax(self.action_type_label, 1),
                                                tf.argmax(self.action_type_predict, 1))
            self.action_type_acu = tf.reduce_mean(tf.cast(self.action_type_cor_pre, tf.float32))

        # action pos loss
        # self.action_pos_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Pos")
        with tf.name_scope("Action_Pos_loss"):
            self.action_pos_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.action_pos_predict, labels=self.action_pos_label))
            # for batch_norm training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.action_pos_train_step = tf.train.AdamOptimizer(self.action_pos_lr_ph).minimize(
                    self.action_pos_loss, var_list=self.action_pos_var_list_train)

            self.action_pos_label_index = tf.argmax(self.action_pos_label, 1)
            self.action_pos_predict_index = tf.argmax(self.action_pos_predict, 1)
            self.action_pos_cor_pre = tf.equal(self.action_pos_label_index, self.action_pos_predict_index)
            self.action_pos_acu = tf.reduce_mean(tf.cast(self.action_pos_cor_pre, tf.float32))

    def _define_rl_loss(self):
        with tf.name_scope("RL_loss"):
            # Compute log probability
            self.action_type_prob = tf.reduce_sum(self.action_type_predict * self.action_type_selected, axis=1)
            self.valid_action_type_prob = tf.reduce_sum(self.action_type_predict * self.valid_action_type, axis=1)
            self.valid_action_type_prob = tf.clip_by_value(self.valid_action_type_prob, 1e-10, 1.)
            self.action_type_prob = self.action_type_prob / self.valid_action_type_prob
            self.action_type_log_prob = tf.log(tf.clip_by_value(self.action_type_prob, 1e-10, 1.))

            self.action_pos_prob = tf.reduce_sum(self.action_pos_predict * self.action_pos_selected, axis=1)
            self.action_pos_log_prob = tf.log(tf.clip_by_value(self.action_pos_prob, 1e-10, 1.))

            self.summary.append(tf.summary.histogram('action_type_prob', self.action_type_prob))
            self.summary.append(tf.summary.histogram('action_pos_prob', self.action_pos_prob))

            # Policy loss and value loss
            self.action_log_prob = self.valid_action_pos * self.action_pos_log_prob + self.action_type_log_prob
            self.advantage = tf.stop_gradient(self.value_target - self.value)
            self.policy_loss = -tf.reduce_mean(self.action_log_prob * self.advantage)
            self.value_loss = -tf.reduce_mean(self.value * self.advantage)

            self.summary.append(tf.summary.scalar('policy_loss', self.policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', self.value_loss))

            # TOD: policy penalty
            self.rl_loss = self.policy_loss + self.value_loss

            self.rl_var_list_train = self.action_type_var_list_train + self.action_pos_var_list_train + self.value_var_list_train
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.rl_train_step = tf.train.AdamOptimizer(self.rl_lr_ph).minimize(
                    self.rl_loss, var_list=self.rl_var_list_train)

            self.summary_op = tf.summary.merge(self.summary)

    def _encoder_basic(self, data):

        with tf.name_scope("Encoder_basic"):
            # conv1 = layer.conv2d_layer(data, 3, 10, "Conv1")
            # conv2 = layer.conv2d_layer(conv1, 3, 20, "Conv2")

            # flatten_input_shape = data.get_shape()
            # nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]
            flatten_input = tf.reshape(data, shape=[-1, self.flatten_map_size])

            d1 = layer.dense_layer(flatten_input, 4096, "DenseLayer1")
            d2 = layer.dense_layer(d1, 2048, "DenseLayer2")
            d3 = layer.dense_layer(d2, 1024, "DenseLayer3", func=None)

        return d3

    def _decoder_basic(self, encode_data):

        with tf.name_scope("Decoder_basic"):
            d1 = layer.dense_layer(encode_data, 2048, "DenseLayer1")
            d2 = layer.dense_layer(d1, 4096, "DenseLayer2")
            d3 = layer.dense_layer(d2, self.flatten_map_size, "DenseLayer3")

        data = tf.reshape(d3, shape=[-1, self.map_num, self.map_width, self.map_width])
        return data

    def _action_type_net(self, encode_data):

        with tf.name_scope("Action_Type"):
            norm_data = layer.batch_norm(encode_data, self.action_type_trainable, "Norm1")
            d1 = layer.dense_layer(norm_data, 512, "DenseLayer1")

            d1_norm = layer.batch_norm(d1, self.action_type_trainable, "Norm2")
            d2 = layer.dense_layer(d1_norm, 256, "DenseLayer2")

            d2_norm = layer.batch_norm(d2, self.action_type_trainable, "Norm3")
            d3 = layer.dense_layer(d2_norm, 128, "DenseLayer3")

            d4 = layer.dense_layer(d3, self.action_num, "DenseLayer4", func=tf.nn.softmax)

        return d4

    def _action_pos_net(self, encode_data, action_type):

        with tf.name_scope("Action_Pos"):
            encode_data_norm = layer.batch_norm(encode_data, self.action_pos_trainable, "Norm1")
            data = tf.concat(values=[encode_data_norm, action_type], axis=1)

            d1 = layer.dense_layer(data, 2048, "DenseLayer1")

            d1_norm = layer.batch_norm(d1, self.action_pos_trainable, "Norm2")
            d2 = layer.dense_layer(d1_norm, 2048, "DenseLayer2")

            d2_norm = layer.batch_norm(d2, self.action_pos_trainable, "Norm3")
            d3 = layer.dense_layer(d2_norm, 4096, "DenseLayer3")

            d3_norm = layer.batch_norm(d3, self.action_pos_trainable, "Norm4")
            d4 = layer.dense_layer(d3_norm, self.map_width * self.map_width, "DenseLayer4", func=tf.nn.softmax)

        return d4

    def _value_net(self, encode_data, action_type):

        with tf.name_scope("Value_Net"):
            encode_data_norm = layer.batch_norm(encode_data, self.value_trainable, "Norm1")
            data = tf.concat(values=[encode_data_norm, action_type], axis=1)

            d1 = layer.dense_layer(data, 512, "DenseLayer1")

            d1_norm = layer.batch_norm(d1, self.value_trainable, "Norm2")
            d2 = layer.dense_layer(d1_norm, 256, "DenseLayer2")

            d2_norm = layer.batch_norm(d2, self.value_trainable, "Norm3")
            d3 = layer.dense_layer(d2_norm, 128, "DenseLayer3")

            d3_norm = layer.batch_norm(d3, self.value_trainable, "Norm4")
            d4 = layer.dense_layer(d3_norm, 1, "DenseLayer4", func=None)

            d4 = tf.reshape(d4, [-1])

        return d4

    def SL_train(self):

        # self.saver.restore(self.sess, self.model_path)

        self.encoder_saver.restore(self.sess, self.encoder_model_path)
        # self.SL_train_encoder()

        self.action_type_saver.restore(self.sess, self.action_type_model_path)
        # self.SL_train_action_type()

        self.action_pos_saver.restore(self.sess, self.action_pos_model_path)
        self.SL_train_action_pos()

    def SL_train_encoder(self):

        map_data = np.load("map_sample.npy")
        map_num = map_data.shape[0]

        batch_size = 20
        iter_num = 5

        for iter_index in range(iter_num):
            i = 0
            while (i+1) * batch_size <= map_num:
                batch_map_data = map_data[i*batch_size:(i+1)*batch_size, :]
                batch_map_data = batch_map_data.reshape(batch_size, 4, self.map_width, self.map_width)

                feed_dict = {
                    self.map_data: batch_map_data[:, :3, :, :],
                    self.encoder_lr_ph: self.encoder_lr
                }
                self.encoder_train_step.run(feed_dict, session=self.sess)

                print("Encoder: epoch: %d/%d, batch_step: %d, loss: " % (iter_index, iter_num, i),
                      self.encoder_loss.eval(feed_dict, session=self.sess))

                i += 1

                if i % 50 == 0:
                    self.encoder_saver.save(self.sess, self.encoder_model_path)
                    print("Model have been save!")

    def SL_train_action_type(self):

        order_data = np.load("new_order_sample.npy")
        action_type_data = self.change_action_type(order_data[:, 1])

        frame_array = order_data[:, 0]

        sample_num = order_data.shape[0]  # 5000
        batch_size = 20
        iter_num = 20

        batch_map_data = np.zeros((batch_size, 3, 64, 64))
        for iter_index in range(iter_num):
            i = 0
            while (i+1) * batch_size <= sample_num:

                for j in range(batch_size):
                    batch_map_data[j, :, :, :] = np.load("../../data/demo1/minimap_%d.npy" %
                                                         int(frame_array[i * batch_size + j]))[[0, 1, 5], :, :]

                batch_action_type = action_type_data[i*batch_size:(i+1)*batch_size, :]

                feed_dict = {
                    self.map_data: batch_map_data[:, :3, :, :],
                    self.action_type_label: batch_action_type,
                    self.action_type_lr_ph: self.action_type_lr
                }

                self.action_type_train_step.run(feed_dict, session=self.sess)

                print("Action_Type: epoch: %d/%d, batch_step: %d, loss: " % (iter_index+1, iter_num, i),
                      self.action_type_loss.eval(feed_dict, session=self.sess),
                      # "predict: ", self.action_type_predict.eval(feed_dict, session=self.sess)[-1, :],
                      # "label: ", batch_action_type[-1, :]
                      )

                i += 1

                if i % 50 == 0:
                    self.action_type_saver.save(self.sess, self.action_type_model_path)
                    print("Model have been save!")

    def SL_train_action_pos(self):

        order_data = np.load("new_order_sample.npy")
        action_pos_data = order_data[:, [2, 3]].astype("int")
        frame_array = order_data[:, 0]

        pos_index = [x[1]*self.map_width+x[0] for x in action_pos_data]

        sample_num = order_data.shape[0]  # 5000
        batch_size = 20
        iter_num = 20

        batch_map_data = np.zeros((batch_size, 3, 64, 64))
        for iter_index in range(iter_num):
            i = 0
            while (i+1) * batch_size <= sample_num:

                for j in range(batch_size):
                    batch_map_data[j, :, :, :] = np.load("../../data/demo1/minimap_%d.npy" %
                                                         int(frame_array[i * batch_size + j]))[[0, 1, 5], :, :]

                batch_action_pos = np.zeros((batch_size, self.map_width * self.map_width))
                for j in range(batch_size):
                    batch_action_pos[j, pos_index[i*batch_size+j]] = 1

                feed_dict = {
                    self.map_data: batch_map_data,
                    self.action_pos_label: batch_action_pos,
                    self.action_pos_lr_ph: self.action_pos_lr
                }

                self.action_pos_train_step.run(feed_dict, session=self.sess)

                print(self.sess.run(tf.gradients(self.action_pos_loss, self.action_pos_var_list_train), feed_dict))

                print("Action_Pos: epoch: %d/%d, batch_step: %d, loss: " % (iter_index+1, iter_num, i),
                      self.action_pos_loss.eval(feed_dict, session=self.sess),
                      # "label:", self.action_pos_label_index.eval(feed_dict, session=self.sess),
                      # "predict:", self.action_pos_predict_index.eval(feed_dict, session=self.sess)
                      )

                i += 1

                if i % 50 == 0:
                    self.action_pos_saver.save(self.sess, self.action_pos_model_path)
                    print("Model have been save!")

    def restore_encoder(self):
        self.encoder_saver.restore(self.sess, self.encoder_model_path)

    def restore_action_type(self):
        self.action_type_saver.restore(self.sess, self.action_type_model_path)

    def restore_action_pos(self):
        self.action_pos_saver.restore(self.sess, self.action_pos_model_path)

    def print_decode_data(self):
        self.encoder_saver.restore(self.sess, self.encoder_model_path)

        map_data = np.load("map_sample.npy")

        batch_size = 1
        iter_num = 1

        for iter_index in range(iter_num):
            i = 0

            batch_map_data = map_data[i*batch_size:(i+1)*batch_size, :]
            batch_map_data = batch_map_data.reshape(batch_size, 4, self.map_width, self.map_width)

            feed_dict = {
                self.map_data: batch_map_data[:, :3, :, :],
                self.encoder_lr_ph: self.encoder_lr
            }

            encode_data = self.decode_data.eval(feed_dict, session=self.sess)
            np.save("encoder.npy", encode_data)

            decode_data = self.decode_data.eval(feed_dict, session=self.sess)

            # decode data seems pretty good!
            print(1)

    def test_action_type(self):
        self.encoder_saver.restore(self.sess, self.encoder_model_path)
        self.action_type_saver.restore(self.sess, self.action_type_model_path)

        map_data = np.load("map_sample.npy")

        order_data = np.load("order_sample_ap.npy")
        action_type_data = order_data[:, 3:]

        sample_num = order_data.shape[0]  # 2000
        batch_size = 2000
        iter_num = 1

        for iter_index in range(iter_num):
            i = 0
            while (i+1) * batch_size <= sample_num:
                batch_map_data = map_data[i*batch_size:(i+1)*batch_size, :]
                batch_map_data = batch_map_data.reshape(batch_size, 4, self.map_width, self.map_width)
                batch_action_type = action_type_data[i*batch_size:(i+1)*batch_size, :]

                feed_dict = {
                    self.map_data: batch_map_data[:, :3, :, :],
                    self.action_type_label: batch_action_type,
                    self.action_type_lr_ph: self.action_type_lr
                }

                print("Action_Type: epoch: %d/%d, batch_step: %d, acu: " % (iter_index+1, iter_num, i),
                      self.action_type_acu.eval(feed_dict, session=self.sess))

                i += 1

    def test_action_pos(self):
        self.encoder_saver.restore(self.sess, self.encoder_model_path)
        self.action_type_saver.restore(self.sess, self.action_type_model_path)
        self.action_pos_saver.restore(self.sess, self.action_pos_model_path)

        # order_data = np.load("../../data/demo1/order.npy")
        order_data = np.load("new_order_sample.npy")
        action_pos_data = order_data[:, [2, 3]].astype("int")
        frame_array = order_data[:, 0]

        pos_index = [x[1] * self.map_width + x[0] for x in action_pos_data]

        sample_num = order_data.shape[0]
        batch_size = 5
        iter_num = 1

        batch_map_data = np.zeros((batch_size, 3, 64, 64))
        for iter_index in range(iter_num):
            i = 0
            while (i + 1) * batch_size <= sample_num:

                for j in range(batch_size):
                    batch_map_data[j, :, :, :] = np.load("../../data/demo1/minimap_%d.npy" %
                                                         int(frame_array[i*batch_size + j]))[[0, 1, 5], :, :]

                batch_action_pos = np.zeros((batch_size, self.map_width * self.map_width))
                for j in range(batch_size):
                    batch_action_pos[j, pos_index[i * batch_size + j]] = 1

                feed_dict = {
                    self.map_data: batch_map_data,
                    self.action_pos_label: batch_action_pos,
                }

                print("Action_Pos: epoch: %d/%d, batch_step: %d, acu: " % (iter_index + 1, iter_num, i),
                      self.action_pos_acu.eval(feed_dict, session=self.sess),
                      "label:", self.action_pos_label_index.eval(feed_dict, session=self.sess),
                      "predict:", self.action_pos_predict_index.eval(feed_dict, session=self.sess)
                      )

                i += 1

    def predict(self, map_data):

        self.encoder_saver.restore(self.sess, self.encoder_model_path)
        self.action_type_saver.restore(self.sess, self.action_type_model_path)
        self.action_pos_saver.restore(self.sess, self.action_pos_model_path)

        feed_dict = {self.map_data: map_data.reshape(-1, self.map_num, self.map_width, self.map_width)}

        # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon, 4: nothing
        action_type_pro = self.action_type_predict.eval(feed_dict, session=self.sess)
        action_type = action_type_pro.argmax()

        action_pos = self.action_pos_predict_index.eval(feed_dict, session=self.sess)

        class WorldPos(object):
            x = 0
            y = 0

        WorldPos.x = action_pos % self.map_width
        WorldPos.y = action_pos // self.map_width

        return action_type_pro, action_type, WorldPos

    def change_action_type(self, action_type_array):
        data = np.zeros((action_type_array.shape[0], self.action_num))

        for i in range(action_type_array.shape[0]):
            data[i, int(action_type_array[i])] = 1

        return data

















