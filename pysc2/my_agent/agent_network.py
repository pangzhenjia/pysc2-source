import tensorflow as tf
import numpy as np
import pysc2.my_agent.layer as layer


class ProbeNetwork(object):

    def __init__(self):

        self.encoder_version = "basic"
        self.encoder_model_path = "model/encoder_%s/probe" % self.encoder_version

        self.action_type_trainable = False
        self.action_type_model_path = "model/action_type/probe"

        self.map_width = 64
        self.map_num = 3
        self.action_num = 4

        self.encoder_lr = 0.00001
        self.action_type_lr = 0.0001
        self.action_pos_lr = 0.0001

        self.flatten_map_size = self.map_width * self.map_width * self.map_num

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self._create_graph()

            self.sess = tf.Session(graph=self.graph)

            self.encoder_var_list_save = list(set(self.encoder_var_list_train +
                                                  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Encoder")))
            self.encoder_saver = tf.train.Saver(var_list=self.encoder_var_list_save)

            self.action_type_var_list_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Action_Type")
            self.action_type_saver = tf.train.Saver(var_list=self.action_type_var_list_save)

            # tf.summary.FileWriter("logs/", self.sess.graph)
            # self.sess.run(tf.global_variables_initializer())

    def _create_graph(self):

        # define learning rate
        self.encoder_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="encoder_lr")
        self.action_type_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="action_type_lr")
        self.action_pos_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="action_pos_lr")

        # define input
        self.map_data = tf.placeholder(dtype=tf.float32, shape=[None, self.map_num, self.map_width, self.map_width],
                                       name="input_map_data")

        self.action_type_label = tf.placeholder(dtype=tf.float32, shape=[None, self.action_num], name="Action_Type_label")
        self.action_pos_label = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="Action_Position_label")

        # define network structure
        self.encode_data = eval("self._encoder_%s(self.map_data)" % self.encoder_version)
        self.decode_data = eval("self._decoder_%s(self.encode_data)" % self.encoder_version)

        self.action_type_predict = self._action_type_net(self.encode_data)

        self.encode_data_action = tf.concat(values=[self.encode_data, self.action_type_predict], axis=1)
        self.action_pos_predict = self._action_pos_net(self.encode_data_action)

        # define network loss to train
        self.encoder_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
        self.encoder_var_list_train.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder"))
        with tf.name_scope("Encoder_loss"):
            self.encoder_loss = tf.reduce_mean(tf.squared_difference(self.map_data, self.decode_data))
            self.encoder_train_step = tf.train.AdamOptimizer(self.encoder_lr_ph).minimize(
                self.encoder_loss, var_list=self.encoder_var_list_train)

        self.action_type_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Type")
        with tf.name_scope("Action_Type_loss"):
            self.action_type_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.action_type_predict, labels=self.action_type_label))
            # for batch_norm training
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.action_type_train_step = tf.train.AdamOptimizer(self.action_type_lr_ph).minimize(
                    self.action_type_loss, var_list=self.action_type_var_list_train)

        self.action_pos_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Position_Regress")
        with tf.name_scope("Action_Pos_loss"):
            self.action_pos_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.action_pos_predict, labels=self.action_pos_label))
            self.action_pos_train_step = tf.train.AdamOptimizer(self.action_pos_lr_ph).minimize(
                self.action_pos_loss, var_list=self.action_pos_var_list)

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

        name = "Action_Type"
        with tf.name_scope(name):
            norm_data = layer.batch_norm(encode_data, self.action_type_trainable, "Norm1")
            d1 = layer.dense_layer(norm_data, 512, "DenseLayer1")

            d1_norm = layer.batch_norm(d1, self.action_type_trainable, "Norm2")
            d2 = layer.dense_layer(d1_norm, 256, "DenseLayer2")

            d2_norm = layer.batch_norm(d2, self.action_type_trainable, "Norm3")
            d3 = layer.dense_layer(d2_norm, 128, "DenseLayer3")

            d4 = layer.dense_layer(d3, self.action_num, "DenseLayer4", func=tf.nn.softmax)

        return d4

    def _action_pos_net(self, encode_data):

        with tf.name_scope("Position_Regress"):
            d1 = layer.dense_layer(encode_data, 2048, "DenseLayer1")
            d2 = layer.dense_layer(d1, 4096, "DenseLayer2")
            d3 = layer.dense_layer(d2, self.map_width * self.map_width, "DenseLayer3", func=tf.nn.softmax)

        return d3

    def train(self):

        # self.saver.restore(self.sess, self.model_path)

        self.encoder_saver.restore(self.sess, self.encoder_model_path)
        # self.train_encoder()

        self.action_type_saver.restore(self.sess, self.action_type_model_path)
        self.train_action_type()

        self.train_action_pos()

    def train_encoder(self):

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

    def train_action_type(self):
        map_data = np.load("map_sample.npy")

        order_data = np.load("order_sample_ap.npy")
        action_type_data = order_data[:, 3:]

        sample_num = order_data.shape[0]  # 2000
        batch_size = 20
        iter_num = 5

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

                self.action_type_train_step.run(feed_dict, session=self.sess)

                print("Action_Type: epoch: %d/%d, batch_step: %d, loss: " % (iter_index+1, iter_num, i),
                      self.action_type_loss.eval(feed_dict, session=self.sess),
                      "predict: ", self.action_type_predict.eval(feed_dict, session=self.sess)[-1, :],
                      "label: ", batch_action_type[-1, :])

                i += 1

                if self.action_type_trainable:
                    if i % 50 == 0:
                        self.action_type_saver.save(self.sess, self.action_type_model_path)
                        print("Model have been save!")

    def train_action_pos(self):
        map_data = np.load("map_sample.npy")

        order_data = np.load("order_sample_ap.npy")
        action_type_data = order_data[:, 3:]
        action_pos_data = order_data[:, [1, 2]].astype("int")

        sample_num = order_data.shape[0]  # 2000
        batch_size = 20
        iter_num = 5

        for iter_index in range(iter_num):
            i = 0
            while (i+1) * batch_size <= sample_num:
                batch_map_data = map_data[i*batch_size:(i+1)*batch_size, :]
                batch_action_type = action_type_data[i*batch_size:(i+1)*batch_size, :]
                batch_action_pos = action_pos_data[i*batch_size:(i+1)*batch_size, :]

                feed_dict = {
                    self.map_data: batch_map_data.reshape(batch_size, self.map_num, self.map_width, self.map_width),
                    self.action_type_label: batch_action_type,
                    self.action_pos_label: batch_action_pos,
                    self.action_pos_lr_ph: self.action_pos_lr
                }

                self.action_pos_train_step.run(feed_dict, session=self.sess)

                print("Action_Pos: epoch: %d/%d, batch_step: %d, loss: " % (iter_index, iter_num, i),
                      self.action_pos_loss.eval(feed_dict, session=self.sess))

                i += 1

                # if i % 50 == 0:
                #     self.saver.save(self.sess, self.model_path)
                #     print("Model have been save!")

    def print_data(self):
        self.encoder_saver.restore(self.sess, "model/test")

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














