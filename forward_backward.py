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
		self.rand = np.random#.RandomState(0)
		self.numberOfStates = numberOfStates

		self.pi = self._normalize(self.rand.rand(1, self.numberOfStates))[0,:]
		self.A = self._stochasticize(self.rand.rand(self.numberOfStates, self.numberOfStates))
		
		self.mu = None
		self.cov = None
		self.nDim = None

	def _normalize(self, x):
		return (x + (x == 0)) / np.sum(x)
	
	def _stochasticize(self, x):
		for i in range(x.shape[0]):
			s = x[i,:].sum()
			s = s + (s == 0)
			x[i,:] /= s
		return x

	def _calculateAlfa(self, B):
		# Forward alg.
		logLikelihood = 0
		T = B.shape[1]
		alfa = np.zeros((T, self.numberOfStates))
		for t in range(T):
			if t == 0:
				alfa[t,:] = B[:, t] * self.pi				
			else:
				# TODO!! check if self.A or self.A.T
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
		# beta[-1, :] = beta[-1, :] / np.sum(beta[-1, :])
		return beta

	def _initB(self, obs):
		B = np.zeros((self.numberOfStates, obs.shape[1]))
		for i in range(self.numberOfStates):
			# np.random.seed(self.rand.randint(1))
			B[i, :] = stat.multivariate_normal.pdf(obs.T, mean=self.mu[:, i].T, cov=self.cov[:, :, i])
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

		# xi = np.zeros((T-1, self.numberOfStates, self.numberOfStates)) # T db n*n-es matrix
		# gamma = np.zeros((self.numberOfStates, T))

		alfa, logLikelihood = self._calculateAlfa(B)
		alfa = alfa.T
		beta = self._calculateBeta(B).T

		xi_sum = np.zeros((self.numberOfStates, self.numberOfStates))
		gamma = np.zeros((self.numberOfStates, T))
		
		for t in range(T - 1):
			partial_sum = self.A * np.dot(alfa[:, t], (beta[:, t] * B[:, t + 1]).T)
			xi_sum += self._normalize(partial_sum)
			partial_g = alfa[:, t] * beta[:, t]
			gamma[:, t] = self._normalize(partial_g)
			  
		partial_g = alfa[:, -1] * beta[:, -1]
		gamma[:, -1] = self._normalize(partial_g)
		
		expected_prior = gamma[:, 0]
		expected_A = self._stochasticize(xi_sum)
		
		expected_mu = np.zeros((self.nDim, self.numberOfStates))
		expected_covs = np.zeros((self.nDim, self.nDim, self.numberOfStates))
		
		gamma_state_sum = np.sum(gamma, axis=1)
		#Set zeros to 1 before dividing
		gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
		
		for s in range(self.numberOfStates):
			gamma_obs = obs * gamma[s, :]
			expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
			partial_cov = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:,s], expected_mu[:,s].T)
			expected_covs[:,:,s] = np.triu(partial_cov) + np.triu(partial_cov).T - np.diag(partial_cov)
		
		#Ensure positive semidefinite by adding diagonal loading
		for i in range(self.numberOfStates):
			expected_covs[:,:,i] += .01 * np.eye(self.nDim)

		self.prior = expected_prior
		self.mu = expected_mu
		self.covs = expected_covs
		self.A = expected_A
		return logLikelihood

	def fit(self, obs, n_iter=15):
		if len(obs.shape) == 2:
			for i in range(n_iter):
				self._emInit(obs)
				log_likelihood = self._emStep(obs)
		elif len(obs.shape) == 3:
			count = obs.shape[0]
			T = obs[0]
			for n in range(count):
				if n > 0:
					T = np.concatenate((T, obs[n]), axis=1)
			self._emInit(T)
			log_likelihood = self._emStep(T)
		return self
	
	def transform(self, obs):      
		if len(obs.shape) == 2:
			B = self._initB(obs)
			_, log_likelihood = self._calculateAlfa(B)
			return log_likelihood
		elif len(obs.shape) == 3:
			count = obs.shape[0]
			out = np.zeros((count,))
			for n in range(count):
				B = self._initB(obs[n, :, :])
				_, log_likelihood = self._calculateAlfa(B)
				out[n] = log_likelihood
			return out