import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

fpaths = []
labels = []
spoken = []
for f in os.listdir('audio'):
	for w in os.listdir('audio/' + f):
		fpaths.append('audio/' + f + '/' + w)
		labels.append(f)
		if f not in spoken:
			spoken.append(f)
print('Words spoken:', spoken)

data = np.zeros((len(fpaths), 32000))
maxsize = -1
for n,file in enumerate(fpaths):
	_, d = wavfile.read(file)
	data[n, :d.shape[0]] = d
	if d.shape[0] > maxsize:
		maxsize = d.shape[0]
data = data[:, :maxsize]

import scipy

def stft(x, fftsize=64, overlap_pct=.5):   
	#Modified from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
	hop = int(fftsize * (1 - overlap_pct))
	w = scipy.hanning(fftsize + 1)[:-1]
	raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
	
	for i in range(0, len(x) - fftsize, hop):
		a = w*x[i:i + fftsize]
		print(a.shape)
		b = np.fft.rfft(a)
		print(b.shape)

	return raw[:, :(fftsize // 2)]

# for row in data:
r = stft(data[0,:])
print(data[0,:].shape)