from __future__ import division
import glob
import numpy as np
import rawpy
import pickle
import sys, os

input_dir = "/home/hduser/Sony/short/"
gt_dir = '/home/hduser/Sony/long/'
input_data_dir = '/home/hduser/Sony/image_data/'
gt_data_dir = '/home/hduser/Sony/gt_data/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

print("Total: " + str(len(train_fns)))

ps = 512  # patch size for training

input_list = []
gt_list = []

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


# gt_images = [None] * 6000
# input_images = {}
# input_images['300'] = [None] * len(train_ids)
# input_images['250'] = [None] * len(train_ids)
# input_images['100'] = [None] * len(train_ids)

count = 0

# for ind in np.random.permutation(len(train_ids)):
for ind in range(len(train_ids)):
    # get the path from image id
    train_id = train_ids[ind]
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
    print(in_files)
    in_path1 = in_files[np.random.random_integers(0, len(in_files) - 1)]
    in_fn1 = os.path.basename(in_path1)

    gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
    gt_path = gt_files[0]
    gt_fn = os.path.basename(gt_path)
    in_exposure1 = float(in_fn1[9:-5])
    gt_exposure = float(gt_fn[9:-5])

    ratio1 = min(gt_exposure / in_exposure1, 300)

    # # if input_images[str(ratio)[0:3]][ind] is None:
    raw1 = rawpy.imread(in_path1)
    tmp_image1 = np.expand_dims(pack_raw(raw1), axis=0) * ratio1

    gt_raw = rawpy.imread(gt_path)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    tmp_gt = np.expand_dims(np.float32(im / 65535.0), axis=0)


    # crop
    H = tmp_image1.shape[1]
    W = tmp_image1.shape[2]

    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)

    for f in in_files:
        in_fn = os.path.basename(f)
        in_exposure = float(in_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        raw = rawpy.imread(f)
        tmp_image = np.expand_dims(pack_raw(raw), axis=0) * ratio

        input_patch = tmp_image[:, yy:yy + ps, xx:xx + ps, :]

        in_filename = os.path.basename(f)[:-4]

        print(input_data_dir + in_filename + '.pkl')

        with open(input_data_dir + in_filename + '.pkl', "w+") as input_file:
            pickle.dump(input_patch, input_file)
        input_file.close()

    # input_patch = tmp_image[:, yy:yy + ps, xx:xx + ps, :]
    gt_patch = tmp_gt[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]


    # in_filename = os.path.basename(in_path)[:-4]
    gt_filename = os.path.basename(gt_path)[:-4]


    print(gt_data_dir+gt_filename+'.pkl')

    with open(gt_data_dir+gt_filename+'.pkl', "w+") as gt_file:
        pickle.dump(gt_patch, gt_file)
    gt_file.close()

    count += 1
    print(count)

print('Ended')


