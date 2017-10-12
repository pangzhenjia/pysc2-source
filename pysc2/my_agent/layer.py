import tensorflow as tf
import numpy as np


def dense_layer(data, out_dim, name, func=tf.nn.relu):
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


def conv2d_layer(data, filter_size, out_dim, name, strides=[1, 1, 1, 1], func=tf.nn.relu):
    in_dim = data.get_shape().as_list()[-1]
    shape = [filter_size, filter_size, in_dim, out_dim]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)

    with tf.name_scope(name):
        w_init = tf.random_uniform(shape, -d, d)
        b_init = tf.random_uniform([out_dim], -d, d)

        w = tf.Variable(w_init, name="kernel_weights", dtype=tf.float32)
        b = tf.Variable(b_init, name="bias", dtype=tf.float32)

        output = tf.nn.conv2d(data, w, strides=strides, padding='SAME', data_format="NCHW") + b
        if func is not None:
            output = func(output)

    return output


def batch_norm(data, is_training, name):

    out_size = data.get_shape().as_list()[-1]

    with tf.name_scope(name):

        fc_mean, fc_var = tf.nn.moments(
            data,
            axes=[0],           # the dimension you wanna normalize, here [0] for batch
                                # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        )

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.8)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        if is_training:
            mean, var = mean_var_with_update()
        else:
            mean = tf.Variable(tf.zeros([out_size, ]), trainable=False, name="moments/Squeeze/ExponentialMovingAverage")
            var = tf.Variable(tf.zeros([out_size, ]), trainable=False, name="moments/Squeeze_1/ExponentialMovingAverage")

        scale = tf.Variable(tf.ones([out_size]), trainable=False, name="scale")
        shift = tf.Variable(tf.zeros([out_size]), trainable=False, name="shift")
        epsilon = 0.001

        data_norm = tf.nn.batch_normalization(data, mean, var, shift, scale, epsilon)

    return data_norm
