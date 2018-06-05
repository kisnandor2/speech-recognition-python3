import numpy as np
import scipy
from scipy.io import wavfile
from peakutils import indexes as findPeaks

import matplotlib.pyplot as plt

def buffer(x, n, p=0, opt=None):
	'''Mimic MATLAB routine to generate buffer array

	MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

	Args
	----
	x:   signal array
	n:   number of data segments
	p:   number of values to overlap
	opt: initial condition options. default sets the first `p` values
		 to zero, while 'nodelay' begins filling the buffer immediately.
	'''

	if p >= n:
		raise ValueError('p ({}) must be less than n ({}).'.format(p,n))

	# Calculate number of columns of buffer array
	cols = int(np.ceil(len(x)/float((n-p))))

	# Check for opt parameters
	if opt == 'nodelay':
		# Need extra column to handle additional values left
		cols += 1
	elif opt != None:
		raise SystemError('Only `None` (default initial condition) and '
						  '`nodelay` (skip initial condition) have been '
						  'implemented')
	# Create empty buffer array
	b = np.zeros((n, cols))
	# Fill buffer by column handling for initial condition and overlap
	j = 0
	for i in range(cols):
		# Set first column to n values from x, move to next iteration
		if i == 0 and opt == 'nodelay':
			b[0:n,i] = x[0:n]
			continue
		# set first values of row to last p values
		elif i != 0 and p != 0:
			b[:p, i] = b[-p:, i-1]
		# If initial condition, set p elements in buffer array to zero
		else:
			b[:p, i] = 0
		# Get stop index positions for x
		k = j + n - p
		# Get stop index position for b, matching number sliced from x
		n_end = p+len(x[j:k])
		# Assign values to buffer array from x
		b[p:n_end,i] = x[j:k]
		# Update start index location for next iteration of x
		j = k
	return b

def nextPowerOf2(n):
	p = 1
	i = 0
	while (p < n):
		p = p*2
		i += 1
	return i;

def extractFeatures(x):
	fs=8000
	framesize=64
	overlap=32
	D=6

	power = nextPowerOf2(framesize)
	frames= buffer(x, framesize, overlap)
	w = scipy.hanning(framesize)
	frameFrequencies = np.zeros((D, frames.shape[1]))

	i = 0
	NFFT = np.power(2, power)
	for frame in frames.T:
		x = frame * w
		X = scipy.fft(x, n=NFFT)/framesize
		f = fs/2*np.linspace(0,1,NFFT//2+1)

		# locs, vals = peakFind(np.abs(X[0:(NFFT//2+1)]))
		X = np.abs(X[0:(NFFT//2+1)])
		locs = findPeaks(X, thres=0.02)
		realLocs = np.zeros((D,1))
		for j in range(locs.shape[0]):
			count = 0
			loc1 = locs[j]
			for k in range(locs.shape[0]):
				loc2 = locs[k]
				if X[loc1] < X[loc2]:
					count += 1
			if count < D:
				realLocs[count] = loc1
		realLocs = realLocs.flatten().astype(int)
		# if i > 83:
		# 	print(realLocs)
		# 	print(X[realLocs])
		# 	print(f[realLocs])
		# 	plt.plot(X)
		# 	plt.show()
		frameFrequencies[:,i] = f[realLocs]
		i += 1
	return frameFrequencies

def peakFind(x, numberOfPeaks=6, leftSize=1, centerSize=1, rightSize=1, fn=np.mean):
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
		else:
			location = 0
			value = 0
		r.append((location, value))
	r.sort(key=lambda tup: tup[1])
	r = r[::-1]
	locations, values = zip(*r)
	return np.array(locations)[:numberOfPeaks], np.array(values)[:numberOfPeaks]

# data = np.zeros((1, 8000))
# dmax = 0
# _, d = wavfile.read('apple01.wav')
# data[0, :d.shape[0]] = d
# if d.shape[0] > dmax:
# 	dmax = d.shape[0]
# data = data[:, :dmax]
# f = extractFeatures(data[0])
# print(f.shape)

# plt.subplot(211)
# plt.plot(data[0])
# plt.subplot(212)
# plt.plot(np.abs(a[0]))
# plt.plot(p[:,0], v[:,0], 'x', color='red')
# plt.show()
