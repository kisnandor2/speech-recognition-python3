# Forward Backward algorithm
# A = (a_ij) state transition matrix
# B = (b_ij) probablity of observation i at state j (this can be declared for each time T as well, but it will be the same for all T for now)
# pi = (pi_i) probabilty of starting with state_i (t = 1)


# alfa(t,i) = P(obs_1:t, state at t = state_i)

import numpy as np

# ex.
states = {0: 'rainy', 1: 'sunny'}
observations = {0: 'umbrella', 1: 'no umbrella'}
A = [
	[0.7, 0.3],
	[0.4, 0.6]
]
B = [
	[0.9, 0.8],
	[0.1, 0.2]
]
pi = [0.3, 0.6]


class hmm:

	def __init__(self, A, B, pi):
		self.A = np.array(A)
		self.B = np.array(B)
		self.pi = np.array(pi)
		self.numberOfStates = self.A.shape[0]

	def _calculateAlfa(self, obs):
		# Forward alg.
		T = len(obs)
		alfa = np.zeros((T, self.numberOfStates))
		for t in range(T):
			if t == 0:
				alfa[t,:] = self.B[:, obs[t]] * self.pi
			else:
				alfa[t, :] = self.B[:, obs[t]] * np.dot(self.A, alfa[t-1,:])
			alfa[t,:] = alfa[t,:] / np.sum(alfa[t,:])
		return alfa

	def _calculateBeta(self, obs):
		# Backward alg.
		T = len(obs)
		beta = np.zeros((T, self.numberOfStates))
		beta[-1, :] = np.ones(self.A.shape[0])
		for t in range(T-2, -1, -1):
			beta[t, :] = np.dot(self.A, (self.B[:, obs[t+1]] * beta[t+1, :]))
			beta[t, :] = beta[t, :] / np.sum(beta[t, :])
		beta[-1, :] = beta[-1, :] / np.sum(beta[-1, :])
		return beta

	def _emStep(self, obs):
		T = obs.shape[0]

		xi = np.zeros((T-1, self.numberOfStates, self.numberOfStates)) # T db n*n-es matrix
		gamma = np.zeros((self.numberOfStates, T))

		alfa = self._calculateAlfa(obs)
		beta = self._calculateBeta(obs)

		xi_denominator = np.zeros(T-1)
		for t in range(T - 1):	
			for i in range(self.numberOfStates):
				for j in range(self.numberOfStates):
					xi[t, i, j] = alfa[t, i] * self.A[i, j] * self.B[j, obs[t+1]] * beta[t+1, j]
					xi_denominator[t] += xi[t,i,j]
			xi[t] /= xi_denominator[t]
			
		gamma = xi.sum(axis=2)		
		gammaSum = gamma.sum(axis=0)
		# Calculate new A matrix based on xi and gamma
		for i in range(self.numberOfStates):
			for j in range(self.numberOfStates):
				self.A[i, j] = xi[:, i, j].sum() / gammaSum[i]
		print(self.A)
		# Calculate new pi vector based on gamma
		self.pi = gamma[0,:]
		# Calculate new B matrix based on gamma
		# Better: calculate new B based on N dimensional Gauss Distribution(mu vector and cov matrix needed)!!		
		
h = hmm(A, B, pi)
obs = np.array([0,1,0])
# for i in range(obs.shape[0]):
h._emStep(obs)

