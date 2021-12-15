import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import databuilder as dbld
import joblib
import display

USE_SAVED = True
PATH = 'cache/'
SAMPLES_FILE = PATH+'samples.save'
METAS_FILE = PATH+'metas.save'
Y_FILE = PATH+'Y.save'

samples, Y, metas = dbld.collectData('data/', SAMPLES_FILE, METAS_FILE, Y_FILE)

cut = int(len(samples)*0.9)
X_train, X_test = samples[:cut], samples[cut:]
Y_train, Y_test = Y[:cut], Y[cut:]