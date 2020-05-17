import emotion_config as config
from DLWP.preprocessing import imagetoarraypreprocessor
from cnn_architectures import emotionvggnet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from DLWP.io import hdf5DatasetGenerator
import keras.backend as K




trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1/255.0, fill_mode='nearest')
valAug = ImageDataGenerator(rescale=1/255.0)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()



trainGen = hdf5DatasetGenerator.HDF5DatasetGenerator(config.TRAIN_HDF5, batchSize=config.BATCH_SIZE, preprocessors =[iap], aug =trainAug, classes=config.NUM_OF_CLASSES)
valGen = hdf5DatasetGenerator.HDF5DatasetGenerator(config.VAL_HDF5, batchSize=config.BATCH_SIZE, preprocessors =[iap], aug =valAug, classes=config.NUM_OF_CLASSES)


print("[INFO] Compiling Model..........")
model = emotionvggnet.EmotionVGGNet.build(48, 48, depth=1, classes=config.NUM_OF_CLASSES)
opt = Adam(lr=1e-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
                    validation_data=valGen.generator(), validation_steps=valGen.numImages // config.BATCH_SIZE, epochs=65,
                    max_queue_size=config.BATCH_SIZE * 2, verbose=1)

model.save("emo.hdf5")
trainGen.close()
valGen.close()
