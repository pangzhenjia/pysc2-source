import tensorflow as tf
import numpy as np
import pysc2.my_agent.layer as layer
import pysc2.my_agent.agent_network as Network

if __name__ == "__main__":

    # with tf.name_scope('some_scope1'):
    #     a = tf.Variable(1, 'a')
    #     b = tf.Variable(2, 'b')
    #     c = tf.Variable(3, 'c')
    #
    # with tf.name_scope('some_scope2'):
    #     d = tf.Variable(4, 'd')
    #     e = tf.Variable(5, 'e')
    #     f = tf.Variable(6, 'f')
    #
    # h = tf.Variable(8, 'h')
    #
    # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
    #     print(i)

    # order_data = np.load("order_sample.npy")
    # type_array = order_data[:, 1]
    #
    # sample_num = order_data.shape[0]
    # type_data = np.zeros((sample_num, 4))
    #
    # for i in range(sample_num):
    #     type_data[i, int(type_array[i])] = 1
    #
    # order_data_ap = np.zeros((sample_num, 3 + 4))
    # order_data_ap[:, :3] = order_data[:, [0, 2, 3]]
    # order_data_ap[:, 3:] = type_data
    #
    # np.save("order_sample_ap.npy", order_data_ap)

    probe_net = Network.ProbeNetwork()
    probe_net.train()
