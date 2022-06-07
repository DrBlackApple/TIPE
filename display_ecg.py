import display as d
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pickle
from scipy.signal import resample

def reduceSampling(data:list, sample_out=1000):
    out = []
    for lead in data:
        out.append(resample(lead, sample_out)) #resample avec la transformée de fourrier
    return np.array(out), len(out[0])

def convertData(data:list, n_sig, sig_len):

    return np.array(data).flatten(order='F').reshape(sig_len, n_sig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Affiche l'ecg au format xml")
    parser.add_argument('ecg_file', type=str, help="Chemin d'accès au fichier xml de l'ecg")

    args = parser.parse_args()

    tree = ET.parse(args.ecg_file)
    root = tree.getroot()

    #search for strip data
    print('--> {}'.format(root.tag))
    strip_data = root.findall('StripData')[0]
    print('\t|\n\t--> {}'.format(strip_data.tag))
    metas = {'n_sig': int(strip_data[0].text), 'sig_len': int(strip_data[2].text), 'sig_name': []}
    data = []
    for wave in strip_data.findall('WaveformData'):
        metas['sig_name'].append(wave.attrib['lead'])
        data.append(list(map(int, wave.text.strip('\t').split(','))))

    print(metas)
    #d.showECG(data, metas)
    cd, metas['sig_len'] = reduceSampling(data, 1000)
    cd /= 500
    d.showECG(cd, metas)
    cd = convertData(cd, metas['n_sig'], metas['sig_len'])
    with open('johan.dat', 'wb') as f:
        pickle.dump(cd, f)