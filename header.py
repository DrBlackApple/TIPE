import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import databuilder as dbld
import joblib
import display
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, Softmax, Concatenate, concatenate
from tensorflow.keras import Model

USE_SAVED = True
PATH = 'cache/'
SAMPLES_FILE = PATH+'samples.save'
METAS_FILE = PATH+'metas.save'
Y_FILE = PATH+'Y.save'

PERCENT = 0.8

samples, Y, metas = dbld.collectData('data/', SAMPLES_FILE, METAS_FILE, Y_FILE)

cut = int(len(samples)*PERCENT)
X_train, X_test = samples[:cut], samples[cut:]
Y_train, Y_test = Y[:cut], Y[cut:]