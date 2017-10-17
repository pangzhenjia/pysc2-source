import tensorflow as tf
import numpy as np
import pysc2.my_agent.layer as layer
import pysc2.my_agent.agent_network as Network

if __name__ == "__main__":

    # with tf.name_scope('some_scope1'):
    #     a = tf.Variable(111, 'a')
    #     b = tf.Variable(222, 'b')
    #     c = tf.Variable(333, 'c')
    #
    # with tf.name_scope('some_scope2'):
    #     d = tf.Variable(444, 'd')
    #     e = tf.Variable(555, 'e')
    #     f = tf.Variable(666, 'f')
    #
    # h = tf.Variable(8, 'h')
    #
    # some_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope1')
    # sess = tf.Session()
    # saver = tf.train.Saver(var_list=some_1)
    # sess.run(tf.global_variables_initializer())
    #
    # saver.restore(sess, "test_model/test")
    #
    # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
    #     # print(i.eval(session=sess))
    #     print(i)

    # train model
    probe_net = Network.ProbeNetwork()
    probe_net.test_action_pos()
