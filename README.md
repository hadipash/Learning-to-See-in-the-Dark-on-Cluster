# Cluster implementation of "Learning to See in the Dark"

source code: https://github.com/cchen156/Learning-to-See-in-the-Dark

### Install TensorflowOnSpark

1. Run `pip install tensorflow tensorflowonspark` on all the machines (Dom0, VM1 - VM8)

2. Add the following lines to /etc/profile file:  
   `export QUEUE=default`  
   `export LIB_HDFS=$HADOOP_HOME/lib/native`  
   `export LIB_JVM=$JAVA_HOME/jre/lib/amd64/server`  
   `export SPARK_HOME=/opt/spark-2.4.0-bin-hadoop2.7`  
   `export LD_LIBRARY_PATH=${PATH}`

### Run training

1. Test run (6 images, 10 epochs, batch size 2). Input directory with the test dataset is `hdfs://gpu10:9000/Sony_pickle_test/`,
   model output is `hdfs://gpu10:9000/Sony_model_test`.\
   `${SPARK_HOME}/bin/spark-submit` \\  
   `--master yarn` \\  
   `--deploy-mode cluster` \\  
   `--num-executors 15` \\  
   `--driver-memory 3G` \\  
   `--executor-memory 3G` \\  
   `--py-files /home/hduser/see-in-the-dark/train_Sony.py,/home/hduser/see-in-the-dark/inference_Sony.py,/home/hduser/see-in-the-dark/inference_Sony_our.py` \\  
   `--conf spark.dynamicAllocation.enabled=false` \\  
   `--conf spark.yarn.maxAppAttempts=1` \\  
   `--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS` \\  
   `--conf spark.driver.memory=3G` \\  
   `--conf spark.executor.memory=3G` \\  
   `--conf spark.driver.maxResultSize=2G` \\  
   `--conf spark.executor.cores=1` \\  
   `--conf spark.task.cpus=1` \\  
   `/home/hduser/see-in-the-dark/script.py` \\  
   `--batch_size 2` \\  
   `--steps 30` \\  
   `--model hdfs://gpu10:9000/Sony_model_test` \\  
   `--input-dir hdfs://gpu10:9000/Sony_pickle_test/image_data` \\  
   `--gt-dir hdfs://gpu10:9000/Sony_pickle_test/gt_data`\
   To run in a client mode replace the following lines:  
   `--deploy-mode client` \\  
   `--driver-memory 1G` \\  
   `--conf spark.yarn.am.memory=1G` \\

2. Full dataset. Input directory with the full dataset is `hdfs://gpu10:9000/Sony_pickle/`,
   model output is `hdfs://gpu10:9000/Sony_model`.\
   `${SPARK_HOME}/bin/spark-submit` \\  
   `--master yarn` \\  
   `--deploy-mode cluster` \\  
   `--num-executors 15` \\  
   `--driver-memory 3G` \\  
   `--executor-memory 3G` \\  
   `--py-files /home/hduser/see-in-the-dark/train_Sony.py,/home/hduser/see-in-the-dark/inference_Sony.py,/home/hduser/see-in-the-dark/inference_Sony_our.py` \\  
   `--conf spark.dynamicAllocation.enabled=false` \\  
   `--conf spark.yarn.maxAppAttempts=1` \\  
   `--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS` \\  
   `--conf spark.driver.memory=3G` \\  
   `--conf spark.executor.memory=3G` \\  
   `--conf spark.driver.maxResultSize=2G` \\  
   `--conf spark.executor.cores=1` \\  
   `--conf spark.task.cpus=1` \\  
   `/home/hduser/see-in-the-dark/script.py`

### Run inference

${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue ${QUEUE} \
--num-executors 15 \
--driver-memory 3G \
--executor-memory 3G \
--py-files /tmp/pycharm_rustam/train_Sony.py,/tmp/pycharm_rustam/inference_Sony.py,/tmp/pycharm_rustam/inference_Sony_our.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
--conf spark.driver.memory=3G \
--conf spark.executor.memory=3G \
--conf spark.driver.maxResultSize=2G \
--conf spark.executor.cores=1 \
--conf spark.task.cpus=1 \
/tmp/pycharm_rustam/script.py \
--mode inference \
--steps 1 \
--model hdfs://gpu10:9000/Sony_model \
--inference our
--inputfile hdfs://gpu10:9000/predict_images/20005_01_0.1s.ARW20190418-150337.pkl --outputfile testResult.pkl

### Run server

1. To start flask application, do the following commands:
   `cd flask_app` \\  
   `source flaskapp/bin/activate` \\  
   `export FLASK_APP=flask_app.py` \\  
   `flask run --host=0.0.0.0 --port=6000`
2. Connect to vpn.cs.hku.hk, use browser to connect http://202.45.128.135:22610/
3. Upload ARW image to the cluster via the web applciation. Image uploading and processing might take 2-4 minutes depending on file size and network speed.
