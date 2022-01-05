#!/usr/bin/env python3
import header as h

N_EPOCH = 170

"""
    Modèle numéro 2 :
    1 entrée par signal
    puis perceptrons
    puis concaténation et résultat
"""

class signalLayers(h.keras.layers.Layer):
    """Classe pour créer un sous réseau pour un signal
    """
    def __init__(self, siglen, dropout=0.2, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.siglen = siglen
        self.dense = h.keras.layers.Dense(256, activation='relu')
        self.bn = h.keras.layers.BatchNormalization()
        self.dropout = h.keras.layers.Dropout(dropout)

    def call(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

def createBranchs(units, siglen, dropout=0.2):
    """Construit les couches pour chaque signal

    Args:
        units (int): nombre de signals
        siglen (int): Longueur du signal
        dropout (float, optional): Le taux de dropout. Defaults to 0.2.

    Returns:
        List[Model]: liste des modèles
        List[Tensor]: liste des sorties des modèles
    """
    mods = []
    out = []
    inp = []
    for i in range(units):
        inputs = h.keras.layers.Input(shape=(siglen))
        x = signalLayers(siglen, dropout, name='Signal_treatment_'+str(i+1))(inputs)

        mods.append(h.Model(inputs, x))
        out.append(mods[-1].output)
        inp.append(mods[-1].inputs)
    return mods,out,inp

mods, out, inps = createBranchs(h.metas[0]['n_sig'], h.metas[0]['sig_len'])

conc = h.concatenate(out)

x = h.Dense(512, activation='relu', name='Couche_intermediaire')(conc)
x = h.Dense(len(h.Y_train[0]), activation='sigmoid', name='Sortie')(x)

m = h.Model(inputs=inps, outputs=x)

m.summary()
h.keras.utils.plot_model(m, 'train/model2_multiinput.png', show_shapes=True)
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = m.fit(list(h.X_train.swapaxes(0,1)), h.Y_train, batch_size=len(h.X_train), epochs=N_EPOCH)
m.save('cache/m2')

h.display.showHistory(hist)

print(m.predict(list(h.X_test[0].reshape(12, 1, 1000))), h.Y_test[0])