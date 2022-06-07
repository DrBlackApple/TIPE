from typing import Tuple
from scipy.stats import truncnorm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import random as r
import os
import time as t

def truncated_normal(moy:float, et:float, borne_inf:float, borne_sup:float):
    a, b = (borne_inf - moy) / et, (borne_sup - moy) / et
    return truncnorm(a, b, moy, et)

def peuplerDistribution(distribution:scipy.stats._distn_infrastructure.rv_frozen, nombreEchantillons:int) -> np.ndarray:
    return np.array(distribution.rvs(nombreEchantillons))

def initMatricePoids(nbNoeudEntree:int, nbNoeudSortie:int, moy:float, et:float) -> np.ndarray:
    a = -1/np.math.sqrt(nbNoeudEntree)
    b = -a
    tab  = peuplerDistribution(truncated_normal(moy, et, a, b), nbNoeudEntree*nbNoeudSortie)
    return tab.reshape(nbNoeudSortie, nbNoeudEntree)

def initMatricePoids_biais(nbNoeudEntree:int, nbNoeudSortie:int, moy:float, et:float, biais:bool) -> np.ndarray:
    a = -1/np.math.sqrt(nbNoeudEntree)
    b = -a
    if biais:
        nbNoeudEntree +=1

    tab  = peuplerDistribution(truncated_normal(moy, et, a, b), nbNoeudEntree*nbNoeudSortie)
    return tab.reshape(nbNoeudSortie, nbNoeudEntree)

def sigmoide(x:np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-1*x))

def lance(vecteurEntree:np.array, pie:np.array, psi:np.array, fonctiondactivation) -> np.ndarray:
    x = fonctiondactivation(np.matmul(pie, vecteurEntree))
    return fonctiondactivation(np.matmul(psi, x))

def lance_biais(vecteurEntree:np.array, pie:np.array, psi:np.array, fonctiondactivation, biais:bool) -> np.ndarray:
    x = np.concatenate((vecteurEntree, [[1]] if biais else [[0]]), axis=0)
    x = fonctiondactivation(pie@x)
    return fonctiondactivation(psi@x)

def apprentissage(vecteurEntree:np.array, vecteurObjectif:np.array, pie:np.array, psi:np.array, vitesse:float,
                    fonctiondactivation):
    y = fonctiondactivation(np.matmul(pie, vecteurEntree))
    z = fonctiondactivation(np.matmul(psi, y))
    Ez = (vecteurObjectif - z)
    Ey = psi.transpose()@Ez
    psi1 = psi + vitesse*((Ez*z*(1-z))@y.transpose())
    pie1 = pie + vitesse*((Ey*y*(1-y))@vecteurEntree.transpose())
    return pie1, psi1

def apprentissage_biais(vecteurEntree:np.array, vecteurObjectif:np.array, pie:np.array, psi:np.array, vitesse:float, fonctiondactivation, biais:bool):
    x = np.concatenate((vecteurEntree,[[1]] if biais else [[0]]),axis=0)
    y = fonctiondactivation(pie@x)
    z = fonctiondactivation(psi@y)
    Ez = (vecteurObjectif - z)
    Ey = psi.transpose()@Ez
    psi1 = psi + vitesse*((Ez*z*(1-z))@y.transpose())
    pie1 = pie + vitesse*((Ey*y*(1-y))@x.transpose())
    return pie1, psi1, Ez

def lectureFichier(csv:str) -> list:
    data = []
    with open(csv, 'r') as f:
        col_name = f.readline()
        lignes = f.readlines()
    for ligne in lignes:
        tmp = []
        for c in ligne.strip().split(','):
            try:
                c = float(c)
            except ValueError:
                c = str(c)
            tmp.append(c)
        data.append(tmp)
    return data

def choixCouleur(typ:str) -> str:
    if typ=='setosa':
        return 'r'
    elif typ=='versicolor':
        return 'b'
    elif typ=='virginica':
        return 'm'
    else:
        return 'b1'

def affichage(critereUn:int, critereDeux:int, echantillons:list):
    fig,ax = plt.subplots(1, 1)
    for e in echantillons:
        ax.plot(e[critereUn], e[critereDeux], '.', color=choixCouleur(e[-1]), markersize=12)

    return fig

def echantillonsEntrainementCompetition(echantillons:list,pourcent:float):
    cut = int(len(echantillons)*pourcent)
    train = r.sample(echantillons, cut)
    test = list(filter(lambda x: x not in train, echantillons))
    return train, test

def choixObjectif(typ:str):
    if typ=='setosa':
        return [[1],[0],[0]]
    elif typ=='versicolor':
        return [[0],[1],[0]]
    elif typ=='virginica':
        return [[0],[0],[1]]
    else:
        return [[0],[0],[0]]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def entrainement(listeEntrainement:list, pieInit:np.array, psiInit:np.array, fonctionActivation, nbIterations:int, vitesse:float):
    pie = pieInit
    psi = psiInit
    for i in range(nbIterations):
        for item in listeEntrainement:
            x, y = np.array(item[:len(item)-1]), np.array(choixObjectif(item[-1]))
            pie, psi = apprentissage(x.reshape(len(x), 1), y, pie, psi, vitesse, fonctionActivation)
        printProgressBar(i, nbIterations, 'Entrainement')
    return pie, psi


def evaluationPerformance(listeCompetition:list, pie:np.array, psi:np.array, fonctionActivation):
    out = np.array([0]*2)
    for item in listeCompetition:
        x, y = np.array(item[:len(item)-1]), choixObjectif(item[-1])
        ty = lance(x.reshape(len(x), 1), pie, psi, fonctionActivation)
        if np.linalg.norm(y-ty) < 0.1:
            out[0] += 1
        else:
            out[1] += 1
    return (out/len(listeCompetition))*100

def evaluationPerformance_biais(listeCompetition:list, pie:np.array, psi:np.array, fonctionActivation, biais:bool = True):
    out = np.array([0]*2)
    for item in listeCompetition:
        x, y = np.array(item[:len(item)-1]), choixObjectif(item[-1])
        ty = lance_biais(x.reshape(len(x), 1), pie, psi, fonctionActivation, biais)
        if np.linalg.norm(y-ty) < 0.1:
            out[0] += 1
        else:
            out[1] += 1
    return (out/len(listeCompetition))*100

def entrainement_biais(listeEntrainement:list, listeEvaluation:list, pieInit:np.array, psiInit:np.array, fonctionActivation, nbIterations:int, vitesse:float, biais:bool = True):
    pie = pieInit
    psi = psiInit
    e = []
    test = []
    plt.figure()
    for i in range(nbIterations):
        moy_e = 0
        for item in listeEntrainement:
            x, y = np.array(item[:len(item)-1]), np.array(choixObjectif(item[-1]))
            pie, psi, err = apprentissage_biais(x.reshape(len(x), 1), y, pie, psi, vitesse, fonctionActivation, biais)
            moy_e += np.linalg.norm(err)
        e.append(moy_e/len(listeEntrainement))

        for item in listeEvaluation:
            x, y = np.array(item[:len(item)-1]), np.array(choixObjectif(item[-1]))
            err = lance_biais(x.reshape(len(x), 1), pie, psi, fonctionActivation, biais)
            moy_e += np.linalg.norm(y - err)
        test.append(moy_e/len(listeEvaluation))

        printProgressBar(i, nbIterations, 'Entrainement')

    plt.plot(e, 'r', label='erreur entraînement')
    plt.plot(test, 'b', label='test erreur')
    plt.legend()
    #plt.show()

    return pie, psi

pie = initMatricePoids_biais(4, 14, 0, 0.2, True)
psi = initMatricePoids_biais(14, 3, 0, 0.2, False)
e = lectureFichier('iris.txt')
e1, e2 = echantillonsEntrainementCompetition(e, 0.8)
start = t.perf_counter()
pie,psi = entrainement_biais(e1, e2, pie, psi, sigmoide, 5000, 0.1, True)
print('Temps execution : {}s'.format(t.perf_counter()-start))
print(evaluationPerformance_biais(e2, pie, psi, sigmoide, True))
