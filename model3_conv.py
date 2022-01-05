#!/usr/bin/env python3
import header as h

N_EPOCH = 100

"""
    Modèle numéro 3 :
    1 entrée par signal
    puis perceptrons
    puis concaténation et résultat
"""

m = h.keras.Sequential([
    h.Input(shape=(12, 1000)),
    h.Conv1D(512, 12, activation='relu'),
    h.Conv1D(25, 1, activation='relu'),
    h.Conv1D(10, 1, activation='relu'),
    h.Flatten(),
    h.Dense(256, activation='relu'),
    h.Dense(len(h.Y_train[0]), activation='sigmoid')
])
##########

#h.display.showECG(h.X_train[0], h.metas[0])

m.summary()
h.keras.utils.plot_model(m, 'train/model3_conv.png', show_shapes=True)
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = m.fit(h.X_train, h.Y_train, batch_size=len(h.X_train), epochs=N_EPOCH)
m.save('cache/m3')

h.display.showHistory(hist)

print(m.predict([h.X_test[0]]), h.Y_test[0])