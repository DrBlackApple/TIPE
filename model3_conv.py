#!/usr/bin/env python3
from header import *

"""
    Modèle numéro 3 :
    1 entrée par signal
    puis perceptrons
    puis concaténation et résultat
"""

m = keras.Sequential([
    Input(shape=(1000, 12)),
    Conv1D(12, 12, activation="relu"),
    MaxPooling1D(6),
    Conv1D(6, 1, activation='relu'),
    Conv1D(3, 1, activation='relu'),
    Flatten(),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(len(Y_train[0]), activation='softmax')
])
##########

#h.display.showECG(h.X_train[0], h.metas[0])

m.summary()
#keras.utils.plot_model(m, 'train/model3_conv.png', show_shapes=True)
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = m.fit(X_train, Y_train, batch_size=len(X_train), epochs=N_EPOCH)
m.save('cache/m3')

print(m.predict(X_test[0].reshape((1, 12, 1000))), Y_test[0])