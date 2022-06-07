#!/usr/bin/env python3
from header import *

"""
    ########### Modèle ############
"""

m = keras.Sequential([
        Dense(12, input_shape=(1000, 12), activation='relu'),
        BatchNormalization(),
        Dense(3, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(3, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(12, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(len(Y_train[0]), activation='softmax')
])

m.summary()
#keras.utils.plot_model(m, 'model.png', show_shapes=True)
m.compile(optimizer=opt, loss=lss, metrics=['accuracy'])
m.fit(X_train, Y_train, batch_size=len(X_train), epochs=N_EPOCH, validation_data=(X_val,Y_val), callbacks=cb)
hist = m.evaluate(X_test, Y_test)
#print(hist)
#joblib.dump(hist, 'hist_graph.dat')
#m.save('model_final.hdf5')

print(m.predict(X_test[0].reshape((1, 1000, 12))), Y_test[0])