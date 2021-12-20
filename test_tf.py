import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.compat.v1.disable_eager_execution()

x = np.cos(np.linspace(0, 2*np.pi))
i = tf.constant(x, dtype=tf.float32, name='i')
k = tf.constant([-4, 3, 1, 0.2], dtype=tf.float32, name='k')

print(i, '\n', k, '\n')

data   = tf.reshape(i, [1, int(i.shape[0]), 1], name='data')
kernel = tf.reshape(k, [int(k.shape[0]), 1, 1], name='kernel')

print(data, '\n', kernel, '\n')

res = tf.squeeze(tf.nn.conv1d(data, kernel, 2, 'VALID'))
with tf.compat.v1.Session() as sess:
    t = sess.run(res)
    plt.plot(x, label='normal')
    plt.plot(t, label='Convolution')
    plt.legend()
    plt.show()