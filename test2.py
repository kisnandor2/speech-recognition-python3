import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal as signal

n = 100	
to = np.pi*2
step = to/n
x = np.array([np.sin(i) for i in np.arange(0, to, step)])

def stft(x, fftsize=64, overlap_pct=.5):   
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]    
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x)-fftsize, hop)])
    return raw[:, :(fftsize // 2)]

# for i in range(0, len(x) - fftsize, hop):
# 	a = np.fft.rfft(w*x[i:i + fftsize])
# 	a = a[:(fftsize//2)]

# print(a.shape)

a, b, c = signal.stft(x, nperseg=64)
print(a,b,c.shape)
c = stft(x)
print(c.shape)