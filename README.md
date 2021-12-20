# TIPE

Projet de tipe pour la PT* 2021-2022.


# But

Développer un résaux de neurones afin de diagnostiquer une Electrocardiogramme (ECG).

# Base de données :

On utilise la [PTBXL](https://physionet.org/content/ptb-xl/1.0.1/) contenant plus de 20000 ECG associé à un diagnostique.
Les ECG sont normalisés entre 0 et 1. Chaque modèle les traitera différemment.

Il faut télécharger la base est l'extraire dans un dossier `data/`.

# Modèles :

## 1) Linéaire

Modèle purement séquentiel qui traite directemment l'entrée avec des couches de perceptrons (Dense layer).

## 2) Multi-entrée

Modèle traitant chaque signal de façon indépendante puis fusion de chaque modèle

## 3) Convolutionnel

On a une courbe qui représente le coeur en 3 dimensions, pourquoi pas une convolution ?

# Librairies
- Tensorflow
- Panda
- Numpy
- Sklearn
- joblib
- matplotlib
