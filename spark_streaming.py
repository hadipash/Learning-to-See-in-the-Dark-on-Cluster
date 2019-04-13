from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import argparse
import numpy as np

from tensorflowonspark import TFCluster
import inference_Sony

#Spark config
conf = SparkConf()
conf.setMaster('yarn-client')
conf.set('spark.yarn.dist.files', 'file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/pyspark.zip,file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/py4j-0.10.7-src.zip')
conf.setExecutorEnv('PYTHONPATH', 'pyspark.zip:py4j-0.10.7-src.zip')
conf.setAppName('inference_spark')

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(conf=conf)
sc.setLogLevel("FATAL")
ssc = StreamingContext(sc, 1)

executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

# arguments for Spark and TFoS
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
parser.add_argument("--tensorboard", help="launch tensorboard process", default=False)
parser.add_argument("--driver_ps_nodes", help="""run tensorflow PS node on driver locally.You will need to set cluster_size = num_executors + num_ps""", default=False)
parser.add_argument("--mode", help="train|inference", default="inference")
parser.add_argument("--model", help="HDFS path to save/load model during train/inference", default='hdfs://gpu10:9000/Sony_model')
parser.add_argument("--input", help="HDFS path to inference image", default='hdfs://gpu10:9000/somepath')
parser.add_argument("--output", help="HDFS path to save test/inference output", default="hdfs://gpu10:9000/somepath")
args = parser.parse_args()

#imageRDD = ssc.textFileStream(args.input-dir)

# Create a DStream that will connect to hostname:port, like localhost:9999
imageRDD = ssc.socketTextStream("gpu10", 9999)

def parse(input):
    output = np.fromstring(input, dtype=np.float32)
    return output

#parse the string rdd to numpy rdd
imageRDD = imageRDD.map(lambda x: parse(imageRDD))

cluster = TFCluster.run(sc, inference_Sony.main_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)

labelRDD = cluster.inference(imageRDD)
labelRDD.saveAsTextFile(args.output)
cluster.shutdown()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

print('stopped')