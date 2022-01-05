#!/usr/bin/env python3
import header as h

def euclidian_distance(y_true, y_pred):
    return h.tf.math.reduce_euclidean_norm(y_true - y_pred)

"""
    ########### Modèle ############
"""

N_EPOCH = 30
model = h.keras.Sequential([
    h.keras.layers.Dense(500,input_shape=(h.metas[0]['n_sig'], h.metas[0]['sig_len']), activation='relu'),
    h.keras.layers.BatchNormalization(),
    h.keras.layers.Dropout(0.2),
    h.keras.layers.Dense(128, activation='relu'),
    h.keras.layers.BatchNormalization(),
    h.keras.layers.Flatten(),
    h.keras.layers.Dense(256, activation='relu'),
    h.keras.layers.BatchNormalization(),
    h.keras.layers.Dropout(0.2),
    h.keras.layers.Dense(len(h.Y_train[0]), activation='sigmoid'),
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(h.X_train, h.Y_train, epochs=N_EPOCH)

h.display.showHistory(hist)

print(model.predict(h.X_test[0].reshape(1, 12, 1000)), h.Y_test[0])