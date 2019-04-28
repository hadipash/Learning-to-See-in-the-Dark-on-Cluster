from __future__ import division
# import glob
import numpy as np
import rawpy
import os, sys

import pydoop.hdfs as hdfs



ps = 512  # patch size for training

UPLOAD_FOLDER = '/home/hduser/Sony/Sony_test/'
filename = '20005_00_0.04s.ARW'
TARGET_FOLDER = '/home/hduser/flask_app/textfiles/'

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
    ps = 512

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

    input_str = str(input_full).replace("\n", "x")
    print(input_str)

    with open(TARGET_FOLDER + in_fn + '.txt', "w+") as target_file:
        target_file.write(input_str)
        target_file.close()

    from_path = TARGET_FOLDER + in_fn + '.txt'
    to_path = 'hdfs://gpu10:9000/predict_images/' + in_fn + '.txt'
    hdfs.put(from_path, to_path)
    print(from_path)
    print(to_path)


in_path = UPLOAD_FOLDER + filename
process_raw(in_path)
# in_fn = os.path.basename(in_path)
# in_exposure = float(in_fn[9:-5])
# gt_exposure = 10
# # ratio = 300
# ratio = min(gt_exposure / in_exposure, 300)
# print(in_path)
#
# raw = rawpy.imread(in_path)
# input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
# input_full = np.minimum(input_full, 1.0)
#
# # print(str(input_full))
#
# input_str = str(input_full).replace("\n", "x")
#
# print(input_str)

