from __future__ import division
import argparse
import pickle
from pyspark import SparkContext, SparkConf
from tensorflowonspark import TFCluster
from io import BytesIO

def main_fun(argv, ctx):
    # this will be executed/imported on the executors.
    import sys, time
    from datetime import datetime
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    from tensorflowonspark import TFNode
    import numpy as np

    sys.argv = argv
    num_workers = len(ctx.cluster_spec['worker'])
    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    # the cluster has no GPUs
    cluster, server = TFNode.start_cluster_server(ctx, num_gpus=0)
    # Create generator for Spark data feed
    tf_feed = ctx.get_data_feed(argv.mode == 'inference')

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,
                                                      cluster=cluster)):
            def lrelu(x):
                return tf.maximum(x * 0.2, x)

            def upsample_and_concat(x1, x2, output_channels, in_channels):
                pool_size = 2
                deconv_filter = tf.Variable(
                    tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
                deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

                deconv_output = tf.concat([deconv, x2], 3)
                deconv_output.set_shape([None, None, None, output_channels * 2])

                return deconv_output

            def network(input):
                conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
                conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
                pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

                conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
                conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
                pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

                conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
                conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
                pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

                conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
                conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
                pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

                conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
                conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

                up6 = upsample_and_concat(conv5, conv4, 256, 512)
                conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
                conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

                up7 = upsample_and_concat(conv6, conv3, 128, 256)
                conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
                conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

                up8 = upsample_and_concat(conv7, conv2, 64, 128)
                conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
                conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

                up9 = upsample_and_concat(conv8, conv1, 32, 64)
                conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
                conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
                out = tf.depth_to_space(conv10, 2)
                return out
                