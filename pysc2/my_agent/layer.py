import tensorflow as tf
import numpy as np


def dense_layer(data, out_dim, name, ret_vars=False, func=tf.nn.relu):
    in_dim = data.get_shape().as_list()[-1]
    shape = [in_dim, out_dim]
    d = 1.0 / np.sqrt(in_dim)

    with tf.name_scope(name):
        w_init = tf.random_uniform(shape, -d, d)
        b_init = tf.random_uniform([out_dim], -d, d)

        w = tf.Variable(w_init, name="weights", dtype=tf.float32)
        b = tf.Variable(b_init, name="bias", dtype=tf.float32)

        output = tf.matmul(data, w) + b
        if func is not None:
            output = func(output)

    return output


def conv2d_layer(data, filter_size, out_dim, name, ret_vars=False, strides=[1, 1, 1, 1], func=tf.nn.relu):
    in_dim = data.get_shape().as_list()[-1]
    shape = [filter_size, filter_size, in_dim, out_dim]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)

    with tf.name_scope(name):
        w_init = tf.random_uniform(shape, -d, d)
        b_init = tf.random_uniform([out_dim], -d, d)

        w = tf.Variable(w_init, name="kernel_weights", dtype=tf.float32)
        b = tf.Variable(b_init, name="bias", dtype=tf.float32)

        output = tf.nn.conv2d(data, w, strides==strides, padding='SAME', data_format="NCHW") + b
        if func is not None:
            output = func(output)

    return output
