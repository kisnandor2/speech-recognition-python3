from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

data = []
_, d = wavfile.read('audio/apple/apple01.wav')
data.append(d)
data = np.array(data)

def stft(x, fs=8000, framesz=0.008, hop=0.004):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) for i in range(0, len(x)-framesamp, hopsamp)])
    return X[:, :(framesamp//2)]

a = stft(data[0,:], fs = 8000)

x = data[0,:]
w = scipy.hanning(x.shape[0])
X = scipy.fft(x*w)
X = X[:(X.shape[0]//2)]

plt.figure(1)

plt.subplot(311)
plt.plot(a)

plt.subplot(312)
plt.plot(data[0,:])

plt.subplot(313)
plt.plot(X)

plt.show()

