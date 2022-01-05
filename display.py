import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import header as h

class CallbackDisplay(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        fig, self.ax1 = plt.subplots()
        self.ax1.set_ylabel('Accuracy')
        self.ax1.set_xlabel("Batch step")

        self.ax2 = self.ax1.twinx()
        self.ax2.set_ylabel('Loss')

        #self.loss_line = ax2.plot([], 'r-', label="Loss")[0]
        #self.acc_line = ax1.plot([], 'g-', label="Accuracy")[0]

        self.accuracy = []
        self.loss = []
        self.tick = [1]

    def on_train_batch_end(self, batch, logs=None):
        self.accuracy.append(logs['accuracy'])
        self.loss.append(logs['loss'])
        self.tick.append(self.tick[-1]+1)

        #self.loss_line.set_data(self.tick[:len(self.tick)-1], self.loss)
        #self.acc_line.set_data(self.tick[:len(self.tick)-1], self.accuracy)
        #plt.pause(0.0001)

    def on_train_end(self, logs=None):
        self.ax2.plot(self.tick[:len(self.tick)-1], self.loss, 'r-', label="Loss")
        self.ax1.plot(self.tick[:len(self.tick)-1], self.accuracy, 'g-', label="Accuracy")
        self.ax2.legend()
        self.ax1.legend(loc='upper left')
        plt.show()


def showHistory(hist:h.keras.callbacks.History, save=True):
    fig, ax = plt.subplots(1,1)
    ax.plot(hist.history['loss'], label='Loss')
    ax.plot(hist.history['accuracy'], label='Accuracy')
    ax.legend()
    plt.show()

def showECG(ecg:h.np.ndarray, meta:dict):
    fig, axs = plt.subplots(meta['n_sig'], sharex=True, sharey=True)
    for i in range(meta['n_sig']):
        axs[i].plot(ecg[i], 'r-')
        axs[i].set_title(meta['sig_name'][i])
    plt.show()