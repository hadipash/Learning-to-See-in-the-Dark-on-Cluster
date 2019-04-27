import argparse
import pickle
from pyspark import SparkContext, SparkConf
from tensorflowonspark import TFCluster
from io import BytesIO

import train_Sony, inference_Sony, inference_Sony_our


if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.kryoserializer.buffer.max.mb", "1024")
    conf.setAppName('See in the Dark')

    sc = SparkContext(conf=conf)
    executors = sc.getConf().get("spark.executor.instances")
    num_executors = int(executors) if executors is not None else 1

    # arguments for Spark and TFoS
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=2)
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
    parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
    parser.add_argument("--tensorboard", help="launch tensorboard process", default=False)
    parser.add_argument("--mode", help="train|inference", default="train")
    parser.add_argument("--inference", help="pretrained|our", default="our")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--steps", help="maximum number of steps", type=int, default=1400)
    parser.add_argument("--save_steps", help="model saving frequency", type=int, default=3000)
    parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                        default='hdfs://gpu10:9000/Sony_model')
    parser.add_argument("--input-dir", help="HDFS path to training set",
                        default='hdfs://gpu10:9000/Sony_pickle/image_data')
    parser.add_argument("--gt-dir", help="HDFS path to ground truth training set",
                        default='hdfs://gpu10:9000/Sony_pickle/gt_data')
    parser.add_argument("--outputfile", help="local file for output",
                        default='./numpy.pkl')
    parser.add_argument("--inputfile", help="Input File",
                        default='hdfs://gpu10:9000/Sony_pickle_test/image_data/00001_00_0.1s.pkl')
    args = parser.parse_args()

    if args.mode == 'train':
        in_images = sc.binaryFiles(args.input_dir, 560).sortByKey(ascending=True).map(
            lambda (k, v): (pickle.load(BytesIO(v))))
        gt_images = sc.binaryFiles(args.gt_dir, 560).sortByKey(ascending=True).map(
            lambda (k, v): pickle.load(BytesIO(v)))
        dataRDD = in_images.zip(gt_images)
        dataRDD = dataRDD.cache()

        cluster = TFCluster.run(sc, train_Sony.main_fun, args, args.cluster_size, args.num_ps, args.tensorboard,
                                TFCluster.InputMode.SPARK)

        cluster.train(dataRDD, args.epochs)

    else:   # inference
        imageRDD = sc.binaryFiles(args.inputfile).sortByKey(ascending=True).map(
            lambda (k, v): (pickle.load(BytesIO(v))))
        inputfile = imageRDD.collect()

        print(inputfile)
        print(inputfile[0].shape)

        if args.inference == 'pretrained':
            cluster = TFCluster.run(sc, inference_Sony.main_fun, args, args.cluster_size, args.num_ps, args.tensorboard,
                                    TFCluster.InputMode.SPARK)
        else:
            cluster = TFCluster.run(sc, inference_Sony_our.main_fun, args, args.cluster_size, args.num_ps,
                                    args.tensorboard, TFCluster.InputMode.SPARK)

        print('inference starting.....................................')
        labelRDD = cluster.inference(imageRDD)
        print('inference finished.....................................')

        output = labelRDD.collect()
        print(output)
        with open(args.outputfile, 'wb') as f:
            pickle.dump(output, f)

    cluster.shutdown()
