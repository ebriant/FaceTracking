import tensorflow as tf


def build_models(x, bias, dropout):
    x = tf.reshape(x, shape=[])

    x = tf.nn.conv2d(x, 64, 3, activation=tf.nn.relu, name="conv_1_1")
    x = tf.nn.conv2d(x, 64, 3, activation=tf.nn.relu, name="conv_1_2")
    x = tf.nn.max_pool(x, 2, 2)
    x = tf.nn.conv2d(x, 128, 3, activation=tf.nn.relu, name="conv_2_1")
    x = tf.nn.conv2d(x, 128, 3, activation=tf.nn.relu, name="conv_2_2")
    x = tf.nn.max_pool(x, 2, 2)
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_3_1")
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_3_2")
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_3_3")
    x = tf.nn.max_pool(x, 2, 2)
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_4_1")
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_4_2")
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_4_3")
    x = tf.nn.max_pool(x, 2, 2)
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_5_1")
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_5_2")
    x = tf.nn.conv2d(x, 512, 3, activation=tf.nn.relu, name="conv_5_3")
    x = tf.nn.max_pool(x, 2, 2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 4096)

