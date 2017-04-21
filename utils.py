import tensorflow as tf


def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
