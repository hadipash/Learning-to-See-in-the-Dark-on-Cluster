
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import numpy as np
import sys
from io import BytesIO
from StringIO import StringIO
from tensorflowonspark import TFCluster
import argparse
import inference_Sony_our
from datetime import datetime

conf = SparkConf()
conf.setMaster('yarn-client')
conf.set('spark.yarn.dist.files', 'file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/pyspark.zip,file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/py4j-0.10.7-src.zip')
conf.setExecutorEnv('PYTHONPATH', 'pyspark.zip:py4j-0.10.7-src.zip')
conf.setAppName('spark-streaming')
conf.set("spark.dynamicAllocation.enabled", "false")
# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(conf=conf)
#sc.setLogLevel("FATAL")

executors = sc.getConf().get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1

# arguments for Spark and TFoS
parser = argparse.ArgumentParser()
parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
parser.add_argument("--tensorboard", help="launch tensorboard process", default=False)
parser.add_argument("--mode", help="train|inference", default="inference")
parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("--steps", help="maximum number of steps", type=int, default=1)
parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                    default='hdfs://gpu10:9000/Sony_model/')
parser.add_argument("--output", help="HDFS path to save output file",
                    default='hdfs://gpu10:9000/Sony_output/')
args = parser.parse_args()

print("args:", args)

print("{0} ===== Start".format(datetime.now().isoformat()))

ssc = StreamingContext(sc, 30)

hdfs_path = 'hdfs://gpu10:9000/textnumpy/'

rawtextRDD = ssc.textFileStream(hdfs_path)

def string2numpy(input):
    output = StringIO(input.encode('latin1'))
    output = np.loadtxt(output, dtype=np.float32)
    return np.array(output.reshape(512,512,4))

#parse the string rdd to numpy rdd
imageRDD = rawtextRDD.map(lambda content: string2numpy(content))
imageRDD.pprint()

cluster = TFCluster.run(sc, inference_Sony_our.main_fun, args, args.cluster_size, args.num_ps, args.tensorboard,TFCluster.InputMode.SPARK)
labelRDD = cluster.inference(imageRDD)

labelRDD.pprint()

labelRDD.saveAsTextFiles(args.output)

cluster.shutdown()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

print('stopped')