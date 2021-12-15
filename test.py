import numpy as np
import wfdb
import matplotlib.pyplot as plt

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr[:100]]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

#plt.subplot

signal, meta = wfdb.rdsamp('data/records100/00000/00001_lr')
fs = meta['fs']
sig_len = meta['sig_len']
downsample_factor = {'sample': 1, 'seconds':fs, 'minutes':fs * 60,
                             'hours':fs * 3600}
t = np.linspace(0, sig_len-1, sig_len) / downsample_factor['sample']
signal = np.array(signal).flatten(order='F').reshape((meta['n_sig'], sig_len))

fig, axs = plt.subplots(meta['n_sig'], sharex=True, sharey=True)
for i in range(meta['n_sig']):
    axs[i].plot(t, signal[i])
    axs[i].set_title(meta['sig_name'][i])
plt.show()
