from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

data = []
_, d = wavfile.read('audio/banana/banana05.wav')
data.append(d)
data = np.array(data)

def stft(x, fs=8000, framesz=0.008, hop=0.004):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) for i in range(0, len(x)-framesamp, hopsamp)])
    return X[:, :(framesamp//2)]

def peakFind(x, numberOfPeaks=6, leftSize=3, centerSize=3, rightSize=3, fn=np.mean):
	windowSize = leftSize + centerSize + rightSize
	def isPeak(x):
		leftMean = fn(x[:leftSize])
		center = x[leftSize:leftSize+centerSize]
		rightMean = fn(x[-rightSize:])
		centered = np.argmax(x) == (leftSize + centerSize//2)
		if centered and x[leftSize + centerSize//2] > leftMean and x[leftSize + centerSize//2] > rightMean:
			return leftSize + centerSize//2, x[leftSize + centerSize//2]
		else:
			return -1, -1
	r = []
	for i in range(x.shape[0] - windowSize + 1):
		location, value = isPeak(x[i:i+windowSize])
		if location > -1:
			location += i
		r.append((location, value))
	r.sort(key=lambda tup: tup[1])
	r = r[::-1]
	locations, values = zip(*r)
	return np.array(locations)[:numberOfPeaks], np.array(values)[:numberOfPeaks]
		
# fig, axs = plt.subplots(10,1)

# for i in range(10):
# 	print(i)
# 	p = a[i,:]
# 	locations, values = peakFind(p)
# 	locations = locations[values > -1]
# 	values = values[values > -1]
# 	axs[i].plot(p)
# 	axs[i].plot(locations, values, 'x', color='red')
# plt.show()

a = stft(data[0,:], fs = 8000)

# x = data[0,:]
# w = scipy.hanning(x.shape[0])
# X = scipy.fft(x*w)
# X = X[:(X.shape[0]//2)]

plt.figure(1)

plt.subplot(211)
plt.plot(a[0:100,:])

plt.subplot(212)
plt.plot(data[0,:])

# plt.subplot(313)
# plt.plot(X)

plt.show()

