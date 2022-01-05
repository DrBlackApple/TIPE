import ast
import numpy as np
import wfdb
import numba
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os
import joblib

def loadDatabase(csv:str) -> pd.DataFrame:
    """
        Renvoie un Dataframe à partir du fichier base de donnée csv ptbxl
    """
    df = pd.read_csv(csv, index_col='ecg_id', header=0)
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x)) # eval json array
    return df

def loadSample(path:str, df: pd.DataFrame, samplerate=100):
    """
        Renvoie un tableau de tableau des signaux de tous les patients et les metadonnées
        [[patient1], [patient2], ...]
        [patient1] = [[I], [II], ..., [V3]]
    """
    if samplerate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    elif samplerate == 500:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    else:
        return Exception('Invalid parameter ' + str(samplerate))
    raw = []
    metas = []
    for signal, meta in data:
        raw.append(np.array(signal).flatten(order='F').reshape(meta['n_sig'], meta['sig_len']))
        metas.append(meta)
    return (np.array(raw), np.array(metas))

def loadStatement(csv:str):
    """
        Load statement
    """
    df = pd.read_csv(csv, index_col=None)
    return df

def createYArray(statement:pd.DataFrame, db:pd.DataFrame, save_dir='cache/', use_only_diag=True) -> np.array:
    """One-hot encode les diagnostics -> un vecteur de dimension 71 pour chaque diag
        Sauvergarde l'encodeur pour décoder la prédiction

    Args:
        statement (pd.DataFrame): Les diagnostics
        db (pd.DataFrame): La base de donnés en csv
        save_dir (str, optional): Le dossier de sauvegarde. Defaults to 'cache/'.

    Returns:
        np.array: Un tableau de tableau de vecteurs de dimension len(statement)
    """
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    if use_only_diag:
        statement = statement[statement.diagnostic == 1]

    enc = OneHotEncoder()
    labels = statement['code'].to_numpy()
    enc.fit(labels.reshape(-1, 1))

    joblib.dump(enc, save_dir+'encoder.save')

    Y = []
    for codes in db.scp_codes:
        tmp = np.zeros(len(labels))
        for key in codes.keys():
            if key in labels:
                tmp = np.add(tmp, enc.transform([[key]]).toarray()[0] * codes[key] / 100)
        Y.append(tmp)

    return np.array(Y)

def normalize(samples:np.array, metas:np.array, save_dir='cache/', use_existing=False) -> np.array:
    """Normalise les donnés avec MinMaxScaler

    Args:
        samples (np.array): les samples
        metas (np.array): les métadonnés
        save_dir (str, optional): le dossier de sauvegarde. Defaults to 'cache/'.
        use_existing (bool, optional): Utilise ou non la sauvegarde. Defaults to False.

    Returns:
        np.array: les samples normalisés entre 0 et 1
    """

    names = metas[0]['sig_name']
    for i in range(len(names)):
        data = samples[i,:]

        file = save_dir+'scaler_'+names[i]+'.save'

        if use_existing:
            scaler = joblib.load(file)
            scaler.transform(data)
        else:
            scaler = MinMaxScaler(copy=False)
            scaler.fit_transform(data)
            joblib.dump(scaler, file)

    return samples

def collectData(db_dir, samples_file, metas_file, y_file, use_saved=True):
    """Charge les données à partir de celles sauvergardés ou en les créant

    Args:
        db_dir (str): dossier de la base de donnée
        samples_file (str): emplacement du fichier de sauvegarde des samples
        metas_file (str): emplacement du fichier de sauvegarde des metadonnés
        y_file (str): emplacement du fichier de sauvergarde des labels
        use_saved (bool, optional): Utilise ou non la sauvegarde. Defaults to True.

    Returns:
        Tuple: Renvoie les samples, labels et métadonnés
    """
    db = loadDatabase(db_dir + 'ptbxl_database.csv')
    stat = loadStatement(db_dir + 'scp_statements.csv')

    # Charge les données déjà traitées si elles existent
    if use_saved and os.path.exists(samples_file):
        print('Chargement des données ', end='', flush=True)
        samples = joblib.load(samples_file)
        metas = joblib.load(metas_file)
        Y = joblib.load(y_file)
        print('[OK]')
    else:
        print('Création des données ', end='', flush=True)
        samples, metas = loadSample('data/', db)
        #samples = normalize(samples, metas)
        Y = createYArray(stat, db)

        joblib.dump(samples, samples_file, compress=3)
        joblib.dump(metas, metas_file)
        joblib.dump(Y, y_file, compress=3)
        print('[OK]')
    ######################################################################

    return samples, Y, metas