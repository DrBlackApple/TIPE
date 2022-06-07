import tensorflow.keras as k
import tensorflow as tf
import argparse
import pandas as pd
import wfdb
import numpy as np
import ast
import matplotlib.pyplot as plt
import joblib
import matplotlib.pyplot as plt
import pickle

PATH_TO_DATA = '../data/'
DATABASE = 'ptbxl_database.csv'
STATEMENTS = 'scp_statements.csv'

def showECG(ecg:np.ndarray, meta:dict):
    fig, axs = plt.subplots(meta['n_sig'], sharex=True, sharey=True)
    plt.xlabel('Temps (ms)')
    for i in range(meta['n_sig']):
        axs[i].plot(ecg[i], 'r-')
        axs[i].set_title(meta['sig_name'][i])
        if i == meta['n_sig'] // 2:
            axs[i].set_ylabel('Amplitude (µV)')
    plt.show()

def loadDatabase(csv:str) -> pd.DataFrame:
    """
        Renvoie un Dataframe à partir du fichier base de donnée csv ptbxl
    """
    df = pd.read_csv(csv, index_col='ecg_id', header=0)
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x)) # eval json array
    return df

def loadSamplesById(path:str, df: pd.DataFrame, id:list, samplerate=100):
    """
        Renvoie un tableau de tableau des signaux de tous les patients et les metadonnées
        [[patient1], [patient2], ...]
        [patient1] = [[I], [II], ..., [V3]]
    """
    if samplerate == 100:
        data = [wfdb.rdsamp(path+df.loc[i].filename_lr) for i in id]
    elif samplerate == 500:
        data = [wfdb.rdsamp(path+df.loc[i].filename_hr) for i in id]
    else:
        return Exception('Invalid parameter ' + str(samplerate))
    raw = []
    metas = []
    for signal, meta in data:
        #raw.append(np.array(signal).flatten(order='F').reshape(meta['n_sig'], meta['sig_len']))
        raw.append(np.array(signal))
        metas.append(meta)
    return (np.array(raw), np.array(metas))

def givePredArray(pred):
    ii = tf.argmax(pred, 0) #index of max
    out = np.zeros(len(pred))
    out[ii] = 1
    return out

def loadStatement(csv:str):
    """
        Load statement
    """
    df = pd.read_csv(csv, index_col=None)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prédiction d'un batch de donnée à partir du modèle choisi")
    parser.add_argument('model_path', help="Path to the file model or the dir model for keras", type=str)
    parser.add_argument('-Wid', '--wfdb-ecgs-id', type=int, nargs='+', help='Ecgs id in format wfdb')
    parser.add_argument('--data-path', type=str, default=PATH_TO_DATA, help='Path to the ptbxl database')
    parser.add_argument('-e', '--encoder', type=str, required=True, help='Path to the encoder to decode the statement')
    parser.add_argument('-d', '--dat', type=str, help='The dat file of an ecg at pickle format')

    args = parser.parse_args()

    #load everything
    m = k.models.load_model(args.model_path)
    df = loadDatabase(args.data_path+DATABASE)
    stats = loadStatement(args.data_path+STATEMENTS)
    ecgs = []
    if args.wfdb_ecgs_id:
        ecg, metas = loadSamplesById(args.data_path, df, args.wfdb_ecgs_id)
        ecgs.extend(ecg)
        #showECG(ecg[0].flatten(order='F').reshape(12, 1000), metas[0])
    if args.dat:
        with open(args.dat, 'rb') as f:
            ecgs.append(np.array(pickle.load(f)))
    ecgs = np.array(ecgs)

    enc = joblib.load(args.encoder) # a onehotencoder

    preds = m.predict(ecgs)
    for i in range(len(preds)):
        j = np.argmax(preds[i])
        final = np.zeros(len(preds[0]))
        final[j] = 1
        #pred = enc.inverse_transform([final])[0][0]
        pred = ''
        if args.wfdb_ecgs_id:
            id = args.wfdb_ecgs_id[i]
            print("{}: {}/{} ({})".format(id, pred, df.loc[id].scp_codes, preds[i]))
        if args.dat:
            print("{}: {} ({})".format(i, pred,preds[i]))