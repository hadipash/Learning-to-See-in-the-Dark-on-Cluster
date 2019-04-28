import os, sys, time
from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import glob
import numpy as np
import rawpy
import pydoop.hdfs as hdfs
import scipy.misc
import pickle
import PIL

app = Flask(__name__)

UPLOAD_FOLDER = '/home/hduser/flask_app/uploads'
IMAGE_FOLDER = '/home/hduser/flask_app/uploads/'
TARGET_FOLDER = '/home/hduser/flask_app/pickles/'
RESULT_FOLDER = '/home/hduser/spark-streaming/results/'
PNG_FOLDER = '/home/hduser/flask_app/result_png/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = '/home/hduser/flask_app/result_png'
UL_FOLDER = os.path.join('static', 'im_upload')
IM_FOLDER = os.path.join('static', 'im_result')
app.config['IM_UPLOAD'] = UL_FOLDER
app.config['IM_RESULT'] = IM_FOLDER
ALLOWED_EXTENSIONS = set(['arw'])

ps = 512


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


def process_raw(filepath):
    in_path = filepath
    in_fn = os.path.basename(in_path)
    in_exposure = float(in_fn[9:-5])
    gt_exposure = 10
    # ratio = 300
    ratio = min(gt_exposure / in_exposure, 300)
    print(in_path)

    raw = rawpy.imread(in_path)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    input_full = np.minimum(input_full, 1.0)

    #add to filename to avoid same name
    timestr = time.strftime("%Y%m%d-%H%M%S")

    from_path = TARGET_FOLDER + in_fn + timestr + '.pkl'
    to_path = 'hdfs://gpu10:9000/predict_images/' + in_fn + timestr + '.pkl'


    # input_full = input_full[:, :ps, :ps,:]

    with open(from_path, 'wb+') as pfile:
        pickle.dump(input_full, pfile)

    try:
        hdfs.put(from_path, to_path)
    except:
        print("File maybe already there")
    print(from_path)
    print("ARW processed - " + to_path)

    pkl_fn = in_fn + timestr + '.pkl'

#     CMD_LINE = """${SPARK_HOME}/bin/spark-submit \
# --master yarn \
# --deploy-mode client \
# --queue ${QUEUE} \
# --num-executors 2 \
# --driver-memory 1G \
# --executor-memory 6G \
# --conf spark.dynamicAllocation.enabled=false \
# --conf spark.yarn.maxAppAttempts=1 \
# --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS \
# --conf spark.yarn.am.memory=1G \
# --conf spark.executor.memory=6G \
# --conf spark.executor.cores=1 \
# --conf spark.task.cpus=1 \
# ~/spark-streaming/inference_Sony_our.py \
# --inputfile hdfs://gpu10:9000/predict_images/""" + pkl_fn + " --outputfile " + RESULT_FOLDER + pkl_fn

    CMD_LINE = """${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode client \
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
--inference our \
--inputfile hdfs://gpu10:9000/predict_images/""" + pkl_fn + " --outputfile " + RESULT_FOLDER + pkl_fn

    print(CMD_LINE)

    os.system(CMD_LINE)

    print("Prediction done.")

    return pkl_fn


def retrieve_output(result_fn):
    in_path = RESULT_FOLDER + result_fn

    print(in_path)
    while not os.path.exists(in_path):
        print("Waiting for file in " + in_path)
        time.sleep(1)

    if os.path.isfile(in_path):
        with open(in_path, "rb") as result_file:
            print("Loading pickle...")
            result_np = pickle.load(result_file)

            output = np.minimum(np.maximum(result_np, 0), 1)
            output = output[0, :, :, :]

            png_path = os.path.join(app.config['IM_RESULT'], result_fn[:-4] + '.png')
            print("Converting to image...")
            scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(png_path)

            return png_path
    else:
        return


@app.route("/")
def home():
    return render_template("home.html")

@app.route('/upload', methods=['GET', 'POST'])
def jump():
    dummyImPath = "../static/image/dummy.png"
    return render_template('upload.html')


@app.route("/uploadImage", methods=['POST'])
def uploadImage():
    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file part')
        file = request.files['image']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return render_template("upload.html")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('Image received - ' + filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            fn = process_raw(IMAGE_FOLDER + filename)
            raw = rawpy.imread(IMAGE_FOLDER + filename)
            rgb = raw.postprocess(no_auto_bright=True)
            ip_filename = os.path.join(app.config['IM_UPLOAD'], filename[:-4] + ".png")
            PIL.Image.fromarray(rgb).save(ip_filename, quality=90)

            op_filename = retrieve_output(fn)

            # ip_filename = os.path.join(app.config['IM_UPLOAD'], '20005_01_0.1s.ARW')
            print(ip_filename)
            # op_filename = os.path.join(app.config['IM_RESULT'], '20005_00_0.1s.ARW20190419-073656.png')
            print(op_filename)

            return render_template('upload.html', input_image = ip_filename, output_image = op_filename)


if __name__ == '__main__':
    app.run('0.0.0.0', port=6000)
