from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import os
from forward_backward import hmm

def peakFind(x, numberOfPeaks=6, leftSize=3, centerSize=3, rightSize=3, fn=np.mean):
	windowSize = leftSize + centerSize + rightSize
	def isPeak(x):
		leftMean = fn(x[:leftSize])
		center = x[leftSize:leftSize+centerSize]
		rightMean = fn(x[-rightSize:])
		centered = np.argmax(x) == (leftSize + centerSize//2)
		if centered and fn(center) > leftMean and fn(center) > rightMean:
			return leftSize + centerSize//2, x[leftSize + centerSize//2]
		else:
			return -1, -1
	r = []
	for i in range(x.shape[0] - windowSize + 1):
		location, value = isPeak(x[i:i+windowSize])
		if location > -1:
			location += i
		else:
			value = 1
		r.append((location, value))
	r.sort(key=lambda tup: tup[1])
	r = r[::-1]
	locations, values = zip(*r)
	return np.array(locations)[:numberOfPeaks], np.array(values)[:numberOfPeaks]

def stft(x, fs=8000, framesz=0.008, hop=0.004):
	# Converting the audio wave into multiple FFT transformed signals to be used for feature selection 
	# For more see whyUseStft.py
	framesamp = int(framesz*fs)
	hopsamp = int(hop*fs)
	w = scipy.hanning(framesamp)
	X = scipy.array([scipy.fft(w*x[i:i+framesamp]) for i in range(0, len(x)-framesamp, hopsamp)])
	return X[:, :(framesamp//2)]

if __name__ == '__main__':
	# Read files
	files = []
	labels = []
	spoken = []
	for f in os.listdir('audio'):
		for w in os.listdir('audio/' + f):
			files.append('audio/' + f + '/' + w)
			labels.append(f)
			if f not in spoken:
				spoken.append(f)
	# Store files into data, each row in data is an audio wave
	data = np.zeros((len(files), 8000))
	dmax = 0
	for i,file in enumerate(files):
		_, d = wavfile.read(file)
		data[i, :d.shape[0]] = d
		if d.shape[0] > dmax:
			dmax = d.shape[0]
	data = data[:, :dmax]

	# Get labels
	print('Number of files total:', data.shape[0])
	all_labels = np.zeros(data.shape[0])
	for n, l in enumerate(set(labels)):
	    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
		
	print('Labels and label indices', all_labels)

	# Get observations
	obsFileName = 'obs.npy'
	if (os.path.isfile(obsFileName)):
		allObservations = np.load(obsFileName)
	else:	
		allObservations = []
		for i in range(data.shape[0]):
			p = np.abs(stft(data[i, :]))
			n = 6
			obs = np.zeros((n, p.shape[0]))
			for j in range(p.shape[0]):
				_, t = peakFind(p[j, :], numberOfPeaks=n)
				obs[:, j] = t.copy()
			if i % 10 == 0:
				print("Processed obs %s" % i)
			allObservations.append(obs)
		
		allObservations = np.array(allObservations)
		# Stochasticise observations(sum col = 1)
		for i in range(allObservations.shape[0]):
			allObservations[i] /= allObservations[i].sum(axis=0)
		np.save('obs', allObservations)

	from sklearn.cross_validation import StratifiedShuffleSplit
	sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)

	all_obs = allObservations
	for train_index, test_index in sss:
	    X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
	    y_train, y_test = all_labels[train_index], all_labels[test_index]

	ys = set(all_labels)
	# ms = [hmm(6) for y in ys]
	# _ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
	# ps = [m.transform(X_test) for m in ms]
	# res = np.vstack(ps)
	# predicted_labels = np.argmax(res, axis=0)
	# missed = (predicted_labels != y_test)
	# print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))

	from hmmlearn import hmm
	h = hmm.GMMHMM(n_components=6)
	d = X_train[y_train == 1, :, :]
	t = d[0]
	for i in range(d.shape[0]):
		if i > 0:
			t = np.concatenate((t, d[i]), axis=1)
	print(t.shape)
	h.fit(X_train[0])
	print(h.predict(X_train[30]))

