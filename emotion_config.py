import os
NUM_OF_CLASSES = 6
BASEPATH = "/home/aniket/datasets/fer2013"
OUTPUTPATH = os.path.sep.join([BASEPATH, "output"])

BATCH_SIZE = 128

INPUT_PATH = os.path.sep.join([BASEPATH, "fer2013/fer2013.csv"])

TRAIN_HDF5 = os.path.sep.join([BASEPATH, "hdf5/train.hdf5"])
TEST_HDF5 = os.path.sep.join([BASEPATH, "hdf5/test.hdf5"])
VAL_HDF5 = os.path.sep.join([BASEPATH, "hdf5/val.hdf5"])
