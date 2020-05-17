from DLWP.io import hdf5datasetwriter
import numpy as np
from emotion_config import *



file = open(INPUT_PATH)
file.__next__()

(trainImages, trainLabels) = ([], [])
(testImages, testLabels) = ([], [])
(valImages, valLabels) = ([], [])

print("[INFO] Loading Dataset.......")

for line in file:
    (label, image, usage) = line.strip().split(",")
    label = int(label)


    if NUM_OF_CLASSES == 6:
        if label==0:
            label=1

        if label > 0:
            label-=1

    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))


    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    else:
        testImages.append(image)
        testLabels.append(label)



datasets = [
    (trainImages, trainLabels, TRAIN_HDF5),
    (testImages, testLabels, TEST_HDF5),
    (valImages, valLabels, VAL_HDF5)
]
print("[INFO] Writing data.....")
for (images, labels, outputPath) in datasets:
    print("[INFO] Building {}".format(outputPath))
    writer = hdf5datasetwriter.HDF5DatasetWriter((len(images), 48, 48), outputPath)

    for (image, label) in zip(images, labels):
        writer.add([image], [label])



    writer.close()


file.close()
