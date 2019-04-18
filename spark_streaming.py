
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import numpy as np
import sys
from io import BytesIO
from StringIO import StringIO
from tensorflowonspark import TFCluster
import argparse

conf = SparkConf()
conf.setMaster('yarn-client')
conf.set('spark.yarn.dist.files', 'file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/pyspark.zip,file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/py4j-0.10.7-src.zip')
conf.setExecutorEnv('PYTHONPATH', 'pyspark.zip:py4j-0.10.7-src.zip')
conf.setAppName('spark-streaming')

num_executors = int(executors) if executors is not None else 1
# arguments for Spark and TFoS
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="number of records per batch", type=int, default=1)
parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
parser.add_argument("--num_ps", help="number of ps nodes", type=int, default=1)
parser.add_argument("--tensorboard", help="launch tensorboard process", default=False)
parser.add_argument("--driver_ps_nodes", help="""run tensorflow PS node on driver locally.
    You will need to set cluster_size = num_executors + num_ps""", default=False)
parser.add_argument("--mode", help="train|inference", default="train")
parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
parser.add_argument("--model", help="HDFS path to save/load model during train/inference",
                    default='hdfs://gpu10:9000/checkpoint_pretrained/')
args = parser.parse_args()


# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(conf=conf)
sc.setLogLevel("FATAL")
ssc = StreamingContext(sc, 2)

hdfs_path = 'hdfs://gpu10:9000/textnumpy/'

#rawtextRDD = sc.wholeTextFiles(hdfs_path)
rawtextRDD = ssc.textFileStream(hdfs_path)
# Create a DStream that will connect to hostname:port, like localhost:9999
#rawtextRDD = ssc.socketTextStream("gpu10", 9999)

#rawtextRDD.pprint()
#convert string to numpy array with specific shape

def string2numpy(input):
    #content = BytesIO(input)
    #input = input.decode('utf-16')
    input = input.replace('x', '\n')
    output = np.array(input)
    return output

#parse the string rdd to numpy rdd
imageRDD = rawtextRDD.map(lambda filename,content: string2numpy(content))
#words = rawtextRDD.flatMap(lambda line: line.split(" "))
imageRDD.pprint()
'''
def sendRecord(rdd):
    connection = createNewConnection()  # executed at the driver
    rdd.foreach(lambda record: connection.send(record))
    connection.close()

imageRDD.foreachRDD(sendRecord)
'''

'''
#pairs = words.map(lambda word: (word, 1))
#wordCounts = pairs.reduceByKey(lambda x, y: x + y)
cluster = TFCluster.run(sc, inference_Sony.main_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)

labelRDD = cluster.inference(imageRDD)
labelRDD.saveAsTextFile(args.output)
cluster.shutdown()
'''
ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

print('stopped')