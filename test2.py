import numpy as np
import matplotlib.pyplot as plt
import scipy

n = 100
to = np.pi*2
step = to/n
a = np.array([np.sin(i) for i in np.arange(0, to, step)])
# plt.plot(a)
# plt.show()

x = a
fftsize = 10
hop = 5
w = scipy.hanning(fftsize + 1)[:-1]
# fft = np.array([np.fft.rfft(w*x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
# fft = fft[:, :(fftsize//2)]
# print(fft.shape)
# plt.plot(fft[0,:])
# plt.show()

for i in range(0, len(x) - fftsize, hop):
	a = np.fft.rfft(w*x[i:i + fftsize])
	a = a[:(fftsize//2)]
	plt.figure()
	plt.plot(a)
	plt.show()
