
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

conf = SparkConf()
conf.setMaster('yarn-client')
conf.set('spark.yarn.dist.files', 'file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/pyspark.zip,file:/usr/local/lib/python2.7/dist-packages/pyspark/python/lib/py4j-0.10.7-src.zip')
conf.setExecutorEnv('PYTHONPATH', 'pyspark.zip:py4j-0.10.7-src.zip')
conf.setAppName('NetworkWordCount')

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(conf=conf)
sc.setLogLevel("FATAL")
ssc = StreamingContext(sc, 1)

# Create a DStream that will connect to hostname:port, like localhost:9999
lines = ssc.socketTextStream("gpu10", 9999)

# Split each line into words
words = lines.flatMap(lambda line: line.split(" "))

# Count each word in each batch
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate