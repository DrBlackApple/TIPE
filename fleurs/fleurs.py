import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import *
from mpl_toolkits.mplot3d import Axes3D
import time as t

def get_model(n_in, n_out:int):
    return keras.Sequential([
        Input(n_in),
        Dense(14, activation='sigmoid'),
        Dense(n_out, activation='sigmoid')
    ])

def showFlower(df:pd.DataFrame):

    fig = plt.figure()
    ax = Axes3D(fig)

    for name, idx in df.groupby('species').groups.items():
        ax.scatter(*df.iloc[idx, [0, 2, 3]].T.values, label=name)

    ax.set_xlabel('Longueur sépal')
    ax.set_ylabel('Longueur pétal')
    ax.set_zlabel('Largeur pétal')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('iris.txt')
    #showFlower(df)
    labels = df.species.unique()

    enc = OneHotEncoder()
    enc.fit(labels.reshape(-1, 1))
    df['label'] = df['species'].apply(lambda x: list(enc.transform([[x]]).toarray()[0])) #One hot encode

    data_in = df.sample(frac=1).reset_index(drop=True) #mélange les données

    #split
    cut = int(len(data_in)*0.8)
    X_train, X_test = np.array(data_in.iloc[:cut, :4]), np.array(data_in.iloc[cut:, :4])
    Y_train, Y_test = np.array(data_in.label[:cut].tolist()), np.array(data_in.label[cut:].tolist())

    m = get_model(len(X_train[0]), len(Y_train[0]))
    start = t.perf_counter()
    #keras.utils.plot_model(m, 'fleurs_network.png', show_layer_names=False, show_shapes=True)
    m.compile(optimizer=Adam(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()
    m.fit(X_train, Y_train, epochs=50)
    m.evaluate(X_test, Y_test)

    print('Temps execution : {}s'.format(t.perf_counter()-start))
    print(m.predict(X_test[3].reshape(1, 4)), Y_test[3])