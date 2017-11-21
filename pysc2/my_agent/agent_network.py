import tensorflow as tf
import numpy as np
import pysc2.my_agent.layer as layer
from pysc2.my_agent.utils import get_power_index
from pysc2.lib import actions as sc2_actions
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

_NO_OP = sc2_actions.FUNCTIONS.no_op.id

_MOVE_MINIMAP = sc2_actions.FUNCTIONS.Move_minimap.id
_BUILD_PYLON = sc2_actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_FORGE = sc2_actions.FUNCTIONS.Build_Forge_screen.id
_BUILD_CANNON = sc2_actions.FUNCTIONS.Build_PhotonCannon_screen.id


_ACTION_ARRAY = [_MOVE_MINIMAP, _BUILD_PYLON, _BUILD_FORGE, _BUILD_CANNON]
_ACTION_TYPE_NAME = ["move", "build_pylon", "build_forge", "build_cannon"]


class ProbeNetwork(object):

    def __init__(self):

        self.encoder_version = "basic"
        self.encoder_model_path = "model/encoder_%s/probe" % self.encoder_version

        self.action_type_trainable = False
        self.action_type_model_path = "model/action_type/probe"

        self.action_pos_trainable = False
        self.action_pos_model_path = "model/action_pos/probe"

        self.value_trainable = True
        self.value_model_path = "model/value_net/probe"

        self.sl_training = True
        self.rl_training = True

        self.rl_model_path = "model/rl/probe"
        self.epsilon = [0.2, 0.2]

        self.summary = []
        self.summary_writer = tf.summary.FileWriter("logs/")

        self.map_width = 64
        self.map_num = 3
        self.action_num = 4

        self.encoder_lr = 0.00001
        self.action_type_lr = 0.00001
        self.action_pos_lr = 0.00001

        self.flatten_map_size = self.map_width * self.map_width * self.map_num

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self._create_graph()

            self.sess = tf.Session(graph=self.graph)

            if self.sl_training:
                self._define_sl_saver()
            if self.rl_training:
                self._define_rl_saver()

            # tf.summary.FileWriter("logs/", self.sess.graph)

    def _define_sl_saver(self):
        self.encoder_var_list_save = list(set(self.encoder_var_list_train +
                                              tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Encoder") +
                                              tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Decoder")))
        self.encoder_saver = tf.train.Saver(var_list=self.encoder_var_list_save)

        self.action_type_var_list_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Action_Type")
        self.action_type_saver = tf.train.Saver(var_list=self.action_type_var_list_save)

        self.action_pos_var_list_save = list(set(self.action_pos_var_list_train +
                                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Action_Pos")))
        self.action_pos_saver = tf.train.Saver(var_list=self.action_pos_var_list_save)

    def _define_rl_saver(self):
        # self.rl_var_list_save = self.encoder_var_list_train + self.rl_var_list_train
        # self.rl_var_list_save.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RL_loss"))
        self.rl_saver = tf.train.Saver()

    def initialize(self):
        with self.graph.as_default() as g:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def _create_graph(self):

        # ################################# SL part  ##########################################
        self.encoder_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="encoder_lr")
        self.action_type_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="action_type_lr")
        self.action_pos_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="action_pos_lr")

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
        self._define_sl_var_list_train()
        if self.sl_training:
            self._define_sl_loss()

        # ################################## RL part  ############################################
        self.value = self._value_net(self.encode_data, self.action_type_predict)
        self.value_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Value_Net")

        self.value_target = tf.placeholder(tf.float32, [None], name='value_target')
        self.action_type_selected = tf.placeholder(dtype=tf.float32, shape=[None, self.action_num],
                                                   name="Action_Type_selected")
        self.action_pos_selected = tf.placeholder(dtype=tf.float32, shape=[None, self.map_width*self.map_width],
                                                  name="Action_Position_selected")
        self.valid_action_type = tf.placeholder(tf.float32, [None, self.action_num], name='valid_action_type')
        self.valid_action_pos = tf.placeholder(tf.float32, [None], name='valid_action_pos')

        self.rl_lr_ph = tf.placeholder(tf.float32, None, name='rl_learning_rate')

        if self.rl_training:
            self._define_rl_loss()

    def _define_sl_var_list_train(self):
        # encoder
        self.encoder_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Encoder")
        self.encoder_var_list_train.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Decoder"))

        # action_type
        self.action_type_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Type")

        # action_pos
        self.action_pos_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Action_Pos")

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

    def predict(self, obs):

        map_data = obs.observation["minimap"][[0, 1, 5], :, :]

        feed_dict = {self.map_data: map_data.reshape(-1, self.map_num, self.map_width, self.map_width)}

        # action_type: 0 : move, 1 : build_pylon, 2 : build_forge, 3: build_cannon
        action_type_prob = self.action_type_predict.eval(feed_dict, session=self.sess)
        action_type = action_type_prob.argmax()

        action_pos_prob = self.action_pos_predict.eval(feed_dict, session=self.sess).reshape(-1)

        # get the valid pos for building forge and cannon
        if action_type > 1:
            index_list = get_power_index(obs)
            if np.sum(index_list) > 1:
                new_action_pos_prob = np.zeros(action_pos_prob.shape)
                new_action_pos_prob[index_list] = action_pos_prob[index_list]
                action_pos = new_action_pos_prob.argmax()
            else:
                action_pos = action_pos_prob.argmax()
        else:
            action_pos = action_pos_prob.argmax()

        x = action_pos % self.map_width
        y = action_pos // self.map_width

        # Epsilon greedy exploration
        if self.rl_training and np.random.rand() < self.epsilon[0]:
            action_type = np.random.choice(np.arange(self.action_num))
        if self.rl_training and np.random.rand() < self.epsilon[1]:
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            x = max(0, min(self.map_width - 1, x + dx))
            y = max(0, min(self.map_width - 1, y + dy))

        return action_type_prob, action_type, x, y

    def update(self, rbs, disc, lr, cter):
        # Compute R, which is value of the last observation
        if not self.rl_training:
            return

        obs = rbs[-1][2]
        if obs.last():
            R = 0
        else:
            map_data = obs.observation["minimap"][[0, 1, 5], :, :]
            feed_dict = {self.map_data: map_data.reshape(-1, self.map_num, self.map_width, self.map_width)}
            R = self.value.eval(feed_dict, session=self.sess)

        sample_num = len(rbs)
        map_data_batch = np.zeros((sample_num, self.map_num, self.map_width, self.map_width))

        valid_action_pos = np.zeros(sample_num)
        action_pos_selected = np.zeros((sample_num, self.map_width**2))
        valid_action_type = np.ones((sample_num, self.action_num))
        action_type_selected = np.zeros((sample_num, self.action_num))

        value_target = np.zeros([len(rbs)], dtype=np.float32)
        value_target[-1] = R

        rbs.reverse()
        for i, [obs, action, next_obs, reward] in enumerate(rbs):

            # map data
            map_data_batch[i] = obs.observation["minimap"][[0, 1, 5], :, :]

            # action data
            act_id = action.function
            act_args = action.arguments
            action_type_index = _ACTION_ARRAY.index(act_id)
            action_type_selected[i, action_type_index] = 1

            args = sc2_actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    pos_index = act_arg[1] * self.map_width + act_arg[0]
                    if obs.observation["minimap"][5, act_arg[0], act_arg[1]] == 0:
                        valid_action_pos[i] = 1
                    action_pos_selected[i, pos_index] = 1

            # value data
            value_target[i] = reward + disc * value_target[i - 1]

        feed_dict = {
            self.map_data: map_data_batch,
            self.value_target: value_target,
            self.valid_action_type: valid_action_type,
            self.action_type_selected: action_type_selected,
            self.valid_action_pos: valid_action_pos,
            self.action_pos_selected: action_pos_selected,
            self.rl_lr_ph: lr
        }

        # self.RL_update_encoder(map_data_batch)
        _, summary = self.sess.run([self.rl_train_step, self.summary_op], feed_dict)
        self.summary_writer.add_summary(summary, cter)

    def RL_update_encoder(self, map_data):
        map_num = map_data.shape[0]

        batch_size = 20
        iter_num = 1

        for iter_index in range(iter_num):
            i = 0
            while (i + 1) * batch_size <= map_num:
                batch_map_data = map_data[i * batch_size:(i + 1) * batch_size]

                feed_dict = {
                    self.map_data: batch_map_data,
                    self.encoder_lr_ph: self.encoder_lr
                }
                self.encoder_train_step.run(feed_dict, session=self.sess)

                print("Encoder: epoch: %d/%d, batch_step: %d, loss: " % (iter_index, iter_num, i),
                      self.encoder_loss.eval(feed_dict, session=self.sess))

                i += 1

                # if i % 50 == 0:
                #     self.encoder_saver.save(self.sess, self.encoder_model_path)
                #     print("Model have been save!")

    def SL_train(self):

        # self.restore_rl_model()

        self.encoder_saver.restore(self.sess, self.encoder_model_path)
        # self.SL_train_encoder()

        self.action_type_saver.restore(self.sess, self.action_type_model_path)
        # self.SL_train_action_type()

        # self.action_pos_saver.restore(self.sess, self.action_pos_model_path)
        self.SL_train_action_pos()

    def SL_train_encoder(self):

        frame_num = 4500
        frame_array = np.arange(2, frame_num+2, 2)

        file_num = 6

        batch_size = 50
        iter_num = 5

        batch_map_data = np.zeros((batch_size, 3, 64, 64))
        count = 0
        for iter_index in range(iter_num):
            batch_step = 0
            for file_index in [2, 3, 4, 5, 6]:

                i = 0
                while (i+1) * batch_size <= frame_num / 2:

                    for j in range(batch_size):
                        batch_map_data[j, :, :, :] = np.load("../../data/demo%d/minimap_%d.npy" %
                                                             (file_index, int(frame_array[i * batch_size + j])))[[0, 1, 5], :, :]

                    feed_dict = {
                        self.map_data: batch_map_data,
                        self.encoder_lr_ph: self.encoder_lr
                    }
                    self.encoder_train_step.run(feed_dict, session=self.sess)

                    print("Encoder: epoch: %d/%d, batch_step: %d, loss: " % (iter_index, iter_num, batch_step),
                          self.encoder_loss.eval(feed_dict, session=self.sess))

                    i += 1
                    count += 1
                    batch_step += 1

                    if count % 50 == 0:
                        self.encoder_saver.save(self.sess, self.encoder_model_path)
                        print("Model have been save!")

    def SL_train_action_type(self):

        sample_num = 2000
        batch_size = 100
        iter_num = 20

        batch_map_data = np.zeros((batch_size, 3, 64, 64))
        count = 0
        for iter_index in range(iter_num):
            batch_step = 0
            for file_index in [1, 2, 3, 4, 5, 6]:

                order_data = np.load("data/new_order_sample_%d.npy" % file_index)
                action_type_data = self.change_action_type(order_data[:, 1])

                frame_array = order_data[:, 0]

                i = 0
                while (i+1) * batch_size <= sample_num:

                    for j in range(batch_size):
                        batch_map_data[j, :, :, :] = \
                            np.load("C:/Users/chensy/Desktop/pysc2 source/data/demo1/minimap_%d.npy" %
                                    int(frame_array[i * batch_size + j]))[[0, 1, 5], :, :]

                    batch_action_type = action_type_data[i*batch_size:(i+1)*batch_size, :]

                    feed_dict = {
                        self.map_data: batch_map_data,
                        self.action_type_label: batch_action_type,
                        self.action_type_lr_ph: self.action_type_lr
                    }

                    self.action_type_train_step.run(feed_dict, session=self.sess)

                    print("Action_Type: epoch: %d/%d, batch_step: %d, loss: " % (iter_index+1, iter_num, batch_step),
                          self.action_type_loss.eval(feed_dict, session=self.sess),
                          # "predict: ", self.action_type_predict.eval(feed_dict, session=self.sess)[-1, :],
                          # "label: ", batch_action_type[-1, :]
                          )

                    i += 1
                    count += 1
                    batch_step += 1

                    if count % 50 == 0:
                        self.action_type_saver.save(self.sess, self.action_type_model_path)
                        print("Model have been save!")

    def SL_train_action_pos(self):

        sample_num = 2000
        batch_size = 100
        iter_num = 20

        batch_map_data = np.zeros((batch_size, 3, 64, 64))
        count = 0
        for iter_index in range(iter_num):
            batch_step = 0
            for file_index in [1, 2, 3, 4, 5, 6]:

                order_data = np.load("data/new_order_sample_%d.npy" % file_index)
                action_pos_data = order_data[:, [2, 3]].astype("int")
                frame_array = order_data[:, 0]

                pos_index = [x[1] * self.map_width + x[0] for x in action_pos_data]

                i = 0
                while (i+1) * batch_size <= sample_num:

                    for j in range(batch_size):
                        batch_map_data[j, :, :, :] = \
                            np.load("C:/Users/chensy/Desktop/pysc2 source/data/demo1/minimap_%d.npy" %
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
                    print("Action_Pos: epoch: %d/%d, batch_step: %d, loss: " % (iter_index+1, iter_num, batch_step),
                          self.action_pos_loss.eval(feed_dict, session=self.sess),
                          # "label:", self.action_pos_label_index.eval(feed_dict, session=self.sess),
                          # "predict:", self.action_pos_predict_index.eval(feed_dict, session=self.sess)
                          )

                    i += 1
                    count += 1
                    batch_step += 1

                    if count % 50 == 0:
                        self.action_pos_saver.save(self.sess, self.action_pos_model_path)
                        print("Model have been save!")

    def restore_encoder(self):
        self.encoder_saver.restore(self.sess, self.encoder_model_path)

    def restore_action_type(self):
        self.action_type_saver.restore(self.sess, self.action_type_model_path)

    def restore_action_pos(self):
        self.action_pos_saver.restore(self.sess, self.action_pos_model_path)

    def restore_sl_model(self):
        self.restore_encoder()
        self.restore_action_type()
        self.restore_action_pos()

    def restore_rl_model(self):
        self.rl_saver.restore(self.sess, self.rl_model_path)

    def save_rl_model(self):
        self.rl_saver.save(self.sess, self.rl_model_path)

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

    def change_action_type(self, action_type_array):
        data = np.zeros((action_type_array.shape[0], self.action_num))

        for i in range(action_type_array.shape[0]):
            data[i, int(action_type_array[i])] = 1

        return data

















