import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from skimage import io
from PIL import Image

from src.losses import focal_tversky, tversky
from src.resunet import resblock, upsample_concat

clf_model = keras.models.load_model("./models/seg_cls_res.hdf5")

seg_model = tf.keras.models.load_model(
    "./models/ResUNet-weights.hdf5",
    custom_objects={
        "resblock": resblock,
        "upsample_concat": upsample_concat,
        "focal_tversky": focal_tversky,
        "tversky": tversky,
    },
)

def predict(image, save_name):
    img = io.imread(image)
        # normalizing
    img = img * 1.0 / 255.0
        # reshaping
    img = cv2.resize(img, (256, 256))
        # converting img into array
    img = np.array(img, dtype=np.float64)
        # reshaping the image from 256,256,3 to 1,256,256,3
    img = np.reshape(img, (1, 256, 256, 3))

        # making prediction for tumor in image
    is_defect = clf_model.predict(img)

        # if tumour is not present we append the details of the image to the list
    if np.argmax(is_defect) == 0:
        return "No mask."
    else:
        # Creating a empty array of shape 1,256,256,1
        X = np.empty((1, 256, 256, 3))
        # read the image
        img = io.imread(image)
        # resizing the image and coverting them to array of type float64
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)

        # standardising the image
        img -= img.mean()
        img /= img.std()
        # converting the shape of image from 256,256,3 to 1,256,256,3
        X[
            0,
        ] = img

        # make prediction of mask
        predict = seg_model.predict(X)

        # if sum of predicted mask is 0 then there is not tumour
        if predict.round().astype(int).sum() == 0:
            return "No mask :)"
        else:
            im = Image.fromarray(predict)
            im.save(f"{save_name}")
            return predict