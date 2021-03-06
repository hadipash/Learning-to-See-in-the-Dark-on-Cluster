# Original code: https://github.com/cchen156/Learning-to-See-in-the-Dark

# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division


def main_fun(argv, ctx):
    # this will be executed/imported on the executors.
    import time
    from datetime import datetime
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    from tensorflowonspark import TFNode
    import numpy as np

    num_workers = len(ctx.cluster_spec['worker'])
    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # the cluster has no GPUs
    cluster, server = TFNode.start_cluster_server(ctx, num_gpus=0)

    def feed_dict(batch):
        input_patch = np.zeros((1, 512, 512, 4))

        if batch:
            input_patch = np.array(batch[0][0])
            input_patch = np.expand_dims(input_patch, axis=0)
        return input_patch

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

            in_image = tf.placeholder(tf.float32, [None, None, None, 4])
            gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

            out_image = network(in_image)

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = ctx.absolute_path(argv.model)
        print("tensorflow model path: {0}".format(logdir))

        sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                 logdir=logdir,
                                 summary_op=None,
                                 saver=saver,
                                 stop_grace_secs=300,
                                 save_model_secs=0)

        with sv.managed_session(server.target) as sess:
            print("{0} session ready".format(datetime.now().isoformat()))

            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            tf_feed = TFNode.DataFeed(ctx.mgr, argv.mode == "train")
            while not sv.should_stop() and not tf_feed.should_stop() and step < argv.steps:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                # using feed_dict
                batch_xs = feed_dict(tf_feed.next_batch(1))

                if len(batch_xs) > 0:
                    output = sess.run(out_image, feed_dict={in_image: batch_xs})
                    output = np.minimum(np.maximum(output, 0), 1)
                    tf_feed.batch_results(output)

            if sv.should_stop() or step >= argv.steps:
                tf_feed.terminate()

        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()
