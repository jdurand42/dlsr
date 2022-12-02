import numpy as np
from math import sqrt

def add_intercept(x):
	"""Adds a column of 1’s to the non-empty numpy.array x.
	Args:
		x: has to be an numpy.array, a vector of shape m * 1.
	Returns:
		x as a numpy.array, a vector of shape m * 2.
		None if x is not a numpy.array.
		None if x is a empty numpy.array.
	Raises:
	This function should not raise any Exception.
	"""

	return np.insert(x, 0, np.ones(x.shape[0],), axis=1)

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
		x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
		The sigmoid value as a numpy.ndarray of shape (m, 1).
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	x = x.astype(float)
	sig = 1 / (1 + np.exp(-x))
	return sig

class LogisticRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000, stochastic=False, lambda_=1, ridge_reg=True):
		self.alpha = alpha
		self.thetas = thetas
		self.max_iter = max_iter
		self.losses = []
		self.r2s = []



	def fit_(self, x, y, early_stopping = None, prescision = 5):
		x_prime = add_intercept(x)
		min_loss = np.inf
		count = 0
		for i in range(0, self.max_iter):
			y_pred = self.predict_(x)
			loss = self.loss_(y, y_pred)
			self.losses.append(loss)
			self.r2s.append(self.r2_(y,y_pred))
			if early_stopping is not None:
				if np.round(loss,prescision) < np.round(min_loss,prescision):
					min_loss = loss
					count = 0
				else :
					count += 1
					if count > early_stopping:
						print('early stopping at iteration : ', i)
						break
			j =  ((np.dot(np.transpose(x_prime), y_pred - y))) / len(y)
			self.thetas = self.thetas - self.alpha * j

	def predict_(self, x):
		x_prime = add_intercept(x)
		return sigmoid_(np.dot(x_prime, self.thetas))

	def loss_elem_(self, y, y_hat):
		"""
		Description:
			Calculates all the elements (y_pred - y)^2 of the loss function.
		Args:
			y: has to be an numpy.array, a vector.
			y_hat: has to be an numpy.array, a vector.
		Returns:
			J_elem: numpy.array, a vector of dimension (number of the training examples,1).
			None if there is a dimension matching problem between y and y_hat.
			None if y or y_hat is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""

		eps = 1e-15
		j = (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - (y_hat + eps)))
		j = -j
		return j


	def loss_(self, y, y_hat):
		"""
		Description:
			Calculates the value of loss function.
		Args:
			y: has to be an numpy.array, a vector.
			y_hat: has to be an numpy.array, a vector.
		Returns:
			J_value : has to be a float.
			None if there is a shape matching problem between y or y_hat.
			None if y or y_hat is not of the expected type.
		Raises:
			This function should not raise any Exception.
		"""
		eps = 1e-15
		ones = np.ones(y.shape)
		j = (np.dot(np.transpose(y), np.log(y_hat + eps)) + \
		(np.dot(np.transpose(ones - y), np.log(ones - y_hat + eps))))

		j = j / -len(y)
		return j.mean()

	def score_(self, y, y_hat):
		eq = np.equal(y, y_hat)
		count = 0
		for i in range(0, len(eq)):
			if y[i][0] == y_hat[i][0]:
				count +=1
		return count / len(y)

	def r2_(self, y, y_hat):
		r2 = 1 - (((y - y_hat) * (y - y_hat)).sum() / (((y - y.mean()) * (y - y.mean())).sum()))
		return r2
