# Original code: https://github.com/cchen156/Learning-to-See-in-the-Dark

# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
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
    tf_feed = ctx.get_data_feed(args.mode == 'train')

    def rdd_generator():
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)

            if len(batch) == 0:
                return

            row = batch[0]
            input_patch = row[0]
            gt_patch = row[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input_patch = np.flip(input_patch, axis=1)
                gt_patch = np.flip(gt_patch, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input_patch = np.flip(input_patch, axis=2)
                gt_patch = np.flip(gt_patch, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            input_patch = np.minimum(input_patch, 1.0)

            yield (input_patch, gt_patch)

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

            # Dataset for input data
            ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (
                tf.TensorShape([None, None, None, 4]), tf.TensorShape([None, None, None, 3]))).batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()
            in_image, gt_image = iterator.get_next()

            out_image = network(in_image)

            G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))

            t_vars = tf.trainable_variables()
            lr = tf.placeholder(tf.float32)
            G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = ctx.absolute_path(args.model)
        print("tensorflow model path: {0}".format(logdir))

        epoch = 0

        hooks = [tf.train.StopAtStepHook(last_step=(args.epochs / args.batch_size))] if args.mode == "train" else []
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               scaffold=tf.train.Scaffold(init_op=init_op, saver=saver),
                                               checkpoint_dir=logdir,
                                               hooks=hooks) as sess:
            print("{} session ready".format(datetime.now().isoformat()))

            while not sess.should_stop() and not tf_feed.should_stop():
                # TODO: split training and testing stages
                g_loss = np.zeros((5000, 1))

                st = time.time()

                learning_rate = 1e-4
                if epoch > 2000:
                    learning_rate = 1e-5

                _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                                feed_dict={in_image: in_image, gt_image: gt_image, lr: learning_rate})
                output = np.minimum(np.maximum(output, 0), 1)

                epoch += 1

        print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

        if sess.should_stop() or epoch >= args.epochs:
            tf_feed.terminate()

        # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
        # wait for all other nodes to complete (via done files)
        done_dir = "{}/{}/done".format(ctx.absolute_path(args.model), args.mode)
        print("Writing done file to: {}".format(done_dir))
        tf.gfile.MakeDirs(done_dir)
        with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
            done_file.write("done")

        for i in range(60):
            if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
                print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
                time.sleep(1)
            else:
                print("{} All nodes done".format(datetime.now().isoformat()))
                break


if __name__ == '__main__':
    sc = SparkContext(conf=SparkConf().setAppName("See in the Dark (train Sony)"))
    executors = sc.getConf().get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    # arguments for Spark and TFoS
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--tensorboard", help="launch tensorboard process", default=False)
    parser.add_argument("--driver_ps_nodes", help="""run tensorflow PS node on driver locally.
        You will need to set cluster_size = num_executors + num_ps""", default=False)
    parser.add_argument("--mode", help="train|inference", default="train")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
    parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                        default='hdfs://gpu10:9000/Sony_dataset_pkl/Sony_model')
    parser.add_argument("--input-dir", help="HDFS path to training set",
                        default='hdfs://gpu10:9000/Sony_dataset_pkl/Sony_dataset_pkl/image_data_list')
    parser.add_argument("--gt-dir", help="HDFS path to ground truth training set",
                        default='hdfs://gpu10:9000/Sony_dataset_pkl/Sony_dataset_pkl/gt_data')
    args = parser.parse_args()

    in_images = sc.binaryFiles(args.input_dir, 1000).sortByKey(ascending=True).map(lambda (k, v): (pickle.load(BytesIO(v))))
    gt_images = sc.binaryFiles(args.gt_dir, 1000).sortByKey(ascending=True).map(lambda (k, v): pickle.load(BytesIO(v)))

    print("zipping input and ground truth images")
    dataRDD = in_images.zip(gt_images)

    cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, args.num_ps, args.tensorboard,
                            TFCluster.InputMode.SPARK, driver_ps_nodes=args.driver_ps_nodes)

    if args.mode == 'train':
        cluster.train(dataRDD, args.epochs)

    cluster.shutdown()
