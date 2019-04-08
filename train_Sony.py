# Original code: https://github.com/cchen156/Learning-to-See-in-the-Dark

# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import argparse
from pyspark import SparkContext, SparkConf
from tensorflowonspark import TFCluster


def main_fun(argv, ctx):
    # this will be executed/imported on the executors.
    import sys, os, time, scipy.io
    from datetime import datetime
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    from tensorflowonspark import TFNode
    import numpy as np
    import rawpy
    import glob

    sys.argv = argv
    num_workers = len(ctx.cluster_spec['worker'])
    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index

    # the cluster has no GPUs
    cluster_spec, server = TFNode.start_cluster_server(ctx, num_gpus=0)
    # Create generator for Spark data feed
    tf_feed = ctx.get_data_feed(args.mode == 'train')

    def rdd_generator():
        while not tf_feed.should_stop():
            # TODO: read images and labels
            # batch = tf_feed.next_batch(1)
            # if len(batch) == 0:
            #     return
            # row = batch[0]
            # image = np.array(row[0]).astype(np.float32) / 255.0
            # label = np.array(row[1]).astype(np.int64)
            # yield (image, label)
            pass

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):

            input_dir = 'hdfs://gpu10-cluster/datasets/Sony/Sony/short/'
            gt_dir = 'hdfs://gpu10-cluster/datasets/Sony/Sony/long/'
            checkpoint_dir = 'hdfs://gpu10-clusters/result_Sony/'
            result_dir = 'hdfs://gpu10-cluster/result_Sony/'

            # TODO: probably we want to read ids from txt files
            # get train IDs
            train_fns = glob.glob(gt_dir + '0*.ARW')
            train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

            ps = 512  # patch size for training

            DEBUG = 0
            if DEBUG == 1:
                train_ids = train_ids[0:5]

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

            # TODO: have we done this stage?
            def pack_raw(raw):
                # pack Bayer image to 4 channels
                im = raw.raw_image_visible.astype(np.float32)
                im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

                im = np.expand_dims(im, axis=2)
                img_shape = im.shape
                H = img_shape[0]
                W = img_shape[1]

                out = np.concatenate((im[0:H:2, 0:W:2, :],
                                      im[0:H:2, 1:W:2, :],
                                      im[1:H:2, 1:W:2, :],
                                      im[1:H:2, 0:W:2, :]), axis=2)
                return out

            # TODO: change dataset parameters
            # TODO: this is old code
            # sess = tf.Session()
            # in_image = tf.placeholder(tf.float32, [None, None, None, 4])
            # gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
            # TODO: this is new code
            # Dataset for input data
            ds = tf.data.Dataset.from_generator(rdd_generator, (tf.float32, tf.float32), (
                tf.TensorShape([0 * 0]), tf.TensorShape([10]))).batch(args.batch_size)
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

        hooks = [tf.train.StopAtStepHook(last_step=(args.epochs / args.batch_size))] if args.mode == "train" else []
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               scaffold=tf.train.Scaffold(init_op=init_op, saver=saver),
                                               checkpoint_dir=logdir,
                                               hooks=hooks) as sess:
            print("{} session ready".format(datetime.now().isoformat()))

            epoch = 0
            while not sess.should_stop() and not tf_feed.should_stop():
                # TODO: prepare the dataset
                # TODO: split training and testing stages

                # TODO: do we need this if we are going to use RDD?
                # Raw data takes long time to load. Keep them in memory after loaded.
                gt_images = [None] * 6000
                input_images = {}
                input_images['300'] = [None] * len(train_ids)
                input_images['250'] = [None] * len(train_ids)
                input_images['100'] = [None] * len(train_ids)

                g_loss = np.zeros((5000, 1))

                # TODO: do we need this if we are going to use RDD?
                allfolders = glob.glob('./result/*0')
                lastepoch = 0
                for folder in allfolders:
                    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

                learning_rate = 1e-4
                if epoch > 2000:
                    learning_rate = 1e-5

                cnt = 0
                for ind in np.random.permutation(len(train_ids)):
                    # TODO: prepare data batch for each epoch?
                    # get the path from image id
                    train_id = train_ids[ind]
                    in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
                    in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
                    in_fn = os.path.basename(in_path)

                    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
                    gt_path = gt_files[0]
                    gt_fn = os.path.basename(gt_path)
                    in_exposure = float(in_fn[9:-5])
                    gt_exposure = float(gt_fn[9:-5])
                    ratio = min(gt_exposure / in_exposure, 300)

                    st = time.time()
                    cnt += 1

                    if input_images[str(ratio)[0:3]][ind] is None:
                        raw = rawpy.imread(in_path)
                        input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

                        gt_raw = rawpy.imread(gt_path)
                        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                        gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

                    # crop
                    H = input_images[str(ratio)[0:3]][ind].shape[1]
                    W = input_images[str(ratio)[0:3]][ind].shape[2]

                    xx = np.random.randint(0, W - ps)
                    yy = np.random.randint(0, H - ps)
                    input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
                    gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

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

                    if args.mode == "train":
                        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                                        feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                                   lr: learning_rate})
                        output = np.minimum(np.maximum(output, 0), 1)
                        g_loss[ind] = G_current
                        print("%d %d Loss=%.3f Time=%.3f" % (
                            epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

                epoch += 1


if __name__ == '__main__':
    sc = SparkContext(conf=SparkConf().setAppName("See in the Dark"))
    executors = sc.getConf().get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    # arguments for Spark and TFoS
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--tensorboard", help="launch tensorboard process", default=False)
    parser.add_argument("--driver_ps_nodes", help="""run tensorflow PS node on driver locally.
        You will need to set cluster_size = num_executors + num_ps""", default=False)
    (args, rem) = parser.parse_known_args()

    cluster = TFCluster.run(sc, main_fun, rem, args.cluster_size, args.num_ps, args.tensorboard,
                            TFCluster.InputMode.SPARK, driver_ps_nodes=args.driver_ps_nodes)
    cluster.shutdown()
