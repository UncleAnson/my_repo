import tensorflow as tf
# model helper 初始化函数

def _get_bias_initializer():
    return tf.zeros_initializer()

def _get_weight_initializer():
    return tf.random_normal_initializer(mean=0.0, stddev=0.05)