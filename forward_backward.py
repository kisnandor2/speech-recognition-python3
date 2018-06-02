# Forward Backward algorithm
# A = (a_ij) state transition matrix
# B = (b_ij) probablity of observation i at state j (this can be declared for each time T as well, but it will be the same for all T for now)
# pi = (pi_i) probabilty of starting with state_i (t = 1)


# alfa(t,i) = P(obs_1:t, state at t = state_i)

import numpy as np
import scipy.stats as stat

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

	def __init__(self, numberOfStates):
		self.rand = np.random.RandomState(0)
		self.numberOfStates = numberOfStates

		self.pi = self._normalize(self.rand.rand(1, self.numberOfStates))[0,:]
		self.A = self._stochasticize(self.rand.rand(self.numberOfStates, self.numberOfStates))

		self.mu = None
		self.cov = None
		self.nDim = None

	def _normalize(self, x):
		return (x + (x == 0)) / np.sum(x)
	
	def _stochasticize(self, x):
		return (x + (x == 0)) / np.sum(x, axis=1)

	def _calculateAlfa(self, B):
		# Forward alg.
		logLikelihood = 0
		T = B.shape[1]
		alfa = np.zeros((T, self.numberOfStates))
		for t in range(T):
			if t == 0:
				alfa[t,:] = B[:, t] * self.pi
			else:
				alfa[t, :] = B[:, t] * np.dot(self.A, alfa[t-1,:])
			alfaSum = np.sum(alfa[t,:])
			alfa[t,:] = alfa[t,:] / alfaSum
			logLikelihood += np.log(alfaSum)
		return alfa, logLikelihood

	def _calculateBeta(self, B):
		# Backward alg.
		T = B.shape[1]
		beta = np.zeros((T, self.numberOfStates))
		beta[-1, :] = np.ones(B.shape[0])
		for t in range(T-2, -1, -1):
			beta[t, :] = np.dot(self.A, (B[:, t+1] * beta[t+1, :]))
			beta[t, :] = beta[t, :] / np.sum(beta[t, :])
		beta[-1, :] = beta[-1, :] / np.sum(beta[-1, :])
		return beta

	def _initB(self, obs):
		B = np.zeros((self.numberOfStates, obs.shape[1]))
		for i in range(self.numberOfStates):
			np.random.seed(self.rand.randint(1))
			B[i, :] = stat.multivariate_normal.pdf(obs.T, mean=self.mu[:, i].T, cov=self.cov[:, :, i].T)
		return B

	def _emInit(self, obs):
		# Obs should be a 2D array
		if self.nDim is None:
			self.nDim = obs.shape[0]
		if self.cov is None:
			self.cov = np.zeros((self.nDim, self.nDim, self.numberOfStates))
			for i in range(self.numberOfStates):
				self.cov[:,:,i] += np.diag(np.diag(np.cov(obs)))
		if self.mu is None:
			r = self.rand.choice(np.arange(self.nDim), size=self.numberOfStates, replace=False)
			self.mu = obs[:, r]


	def _emStep(self, obs):
		T = obs.shape[1]
		B = self._initB(obs)

		xi = np.zeros((T-1, self.numberOfStates, self.numberOfStates)) # T db n*n-es matrix
		gamma = np.zeros((self.numberOfStates, T))

		alfa, logLikelihood = self._calculateAlfa(B)
		beta = self._calculateBeta(B)

		xi_denominator = np.zeros(T-1)
		
		for t in range(T - 1):	
			for i in range(self.numberOfStates):
				for j in range(self.numberOfStates):
					xi[t, i, j] = alfa[t, i] * self.A[i, j] * B[j, t+1] * beta[t+1, j]
					xi_denominator[t] += xi[t,i,j]
			xi[t] /= xi_denominator[t]
	
		gamma = xi.sum(axis=2)		
		gammaSum = gamma.sum(axis=0)
		# Calculate new A matrix based on xi and gamma
		for i in range(self.numberOfStates):
			for j in range(self.numberOfStates):
				self.A[i, j] = xi[:, i, j].sum() / gammaSum[i]
		# Calculate new pi vector based on gamma
		self.pi = gamma[0,:]
		# Calculate new B matrix based on gamma
		# Better: calculate new B based on N dimensional Gauss Distribution(mu vector and cov matrix needed)!!
		gamma = gamma.T
		expected_mu = np.zeros((self.nDim, self.numberOfStates))
		expected_covs = np.zeros((self.nDim, self.nDim, self.numberOfStates))
		
		gamma_state_sum = np.sum(gamma, axis=1)
		#Set zeros to 1 before dividing
		gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
		
		obs = obs[:, :-1]
		for s in range(self.numberOfStates):
			gamma_obs = obs * gamma[s, :]
			expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
			partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
			#Symmetrize
			partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
		for i in range(self.numberOfStates):
			expected_covs[:,:,i] += .01 * np.eye(self.nDim)
		self.mu = expected_mu
		self.cov = expected_covs

	def fit(self, obs):
		B = self._initB(obs)
		_, log_likelihood = self._calculateAlfa(B)
		return log_likelihood