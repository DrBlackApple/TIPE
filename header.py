import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import databuilder as dbld
import joblib
import display
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *

USE_SAVED = True
PATH = 'cache/'
SAMPLES_FILE = PATH+'samples.save'
METAS_FILE = PATH+'metas.save'
Y_FILE = PATH+'Y.save'

TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.2

#récupère les samples
samples, Y, metas = dbld.collectData('./data/', SAMPLES_FILE, METAS_FILE, Y_FILE, True, 1)

#split les données
cut = int(len(samples)*TRAIN_PERCENT)
cut2 = int(len(samples)*(TRAIN_PERCENT + VAL_PERCENT))
X_train, X_val, X_test = samples[:cut], samples[cut:cut2], samples[cut2:]
Y_train, Y_val, Y_test = Y[:cut], Y[cut:cut2], Y[cut2:]

cb = [ModelCheckpoint('model_best.hdf5', save_best_only=True),EarlyStopping(monitor='val_loss', patience=9, min_delta=0.00001)]

N_EPOCH = 100
lr = 0.01

opt =  Adam(lr)
lss = 'categorical_crossentropy'