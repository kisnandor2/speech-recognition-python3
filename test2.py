from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import os
from forward_backward import hmm
from peakFind import extractFeatures

if __name__ == '__main__':
	# Read files
	files = []
	for f in os.listdir('audio'):
		for w in os.listdir('audio/' + f):
			files.append('audio/' + f + '/' + w)

	# Store files into data, each row in data is an audio wave
	data = np.zeros((len(files), 8000))
	length = np.zeros((len(files), 1))
	dmax = 0
	for i,file in enumerate(files):
		_, d = wavfile.read(file)
		data[i, :d.shape[0]] = d
		length[i] = d.shape[0]
		if d.shape[0] > dmax:
			dmax = d.shape[0]
	data = data[:, :dmax]
	# Store the length of each audio wave
	length = length.flatten().astype(int)


	db = 1
	err = 0
	# Test it for a lot of times
	for counter in range(db):
		if (counter % 5 == 0):
			print(counter)
		# Random generate test and train data indexes
		rand = np.random
		n = 15
		testCount = 3
		randTrain = np.zeros((n-testCount, 7))
		randTest = np.zeros((testCount, 7))
		for i in range(7):
			randTest[:,i] = rand.choice(np.arange(n), size=testCount, replace=False)
			randTrain[:,i] = np.delete(np.arange(n), randTest[:,i])
		randTrain = randTrain.astype(int)
		randTest = randTest.astype(int)

		# Train the hmms
		# 7 hmm = number of words
		# hmm(6) = number of features(6)
		hmms = []
		for i in range(7):
			T = data[i*n,:length[i*n]]
			h = hmm(6, i)
			for j in randTrain[:,i]:
				k = i*n+j
				# Concatenate the observations; why is it better? (it is)
				T = np.concatenate((T, data[k,:length[k]]), axis=0)
			# Get features
			obs = extractFeatures(T)
			# Train the model with these features
			h.fit(obs)
			hmms.append(h)

		# Test the model
		miss = 0
		ok = 0
		for i in range(7): # Get a word from the i'th group
			for j in randTest[:,i]: # ex. apple01
				k = i*n+j
				maxLikelihood = -np.inf
				index = -1
				# Get the max likelihood for all classes
				obs = extractFeatures(data[k,:length[k]])
				for l in range(7):
					h = hmms[l]
					likelihood = h.transform(obs)
					# Select the class wihch gives the highest likelihood
					if likelihood > maxLikelihood:
						maxLikelihood = likelihood
						index = l
				# If chosen class is the same as the word belongs to
				if index == i:
					ok += 1
				else:
					miss += 1
		print('Error:', miss/(ok+miss))
		err += miss/(ok+miss)
	print(err/db)
