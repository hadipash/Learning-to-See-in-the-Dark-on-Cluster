import scipy.misc
import PIL
import pickle
import numpy as np
import rawpy

in_path = "/home/hduser/flask_app/uploads/20005_01_0.1s.ARW"

print(in_path)
#
# raw = rawpy.imread(in_path)
# rgb = raw.postprocess(no_auto_bright=True)
# PIL.Image.fromarray(rgb).save('image.jpg', quality=90)

print(in_path[:-4])

