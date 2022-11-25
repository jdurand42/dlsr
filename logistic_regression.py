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
		self.stochastic = stochastic
		self.lambda_ = lambda_
		self.ridge_reg = ridge_reg

	def fit_(self, x, y):

		# lambda_ = 0.5
		self.stochastic = False
		self.ridge_reg = False

		x_prime = add_intercept(x)
		if self.stochastic == False:
			for i in range(0, self.max_iter):
				y_pred = self.predict_(x)
				# thetas = self.thetas.copy()
				# thetas[0][0] = 0
				# print(np.dot(np.transpose(x_prime), y_pred - y))
				j =  ((np.dot(np.transpose(x_prime), y_pred - y))) / len(y)
				if self.ridge_reg == True:
					ridge = self.lambda_ + j[1:] + np.square(self.thetas[1:]).sum()
					self.thetas[1:] = self.thetas[1:] - self.alpha * ridge
					# self.thetas[0][0] = self.thetas[0][0] - self.alpha * j[0]
				else:
					self.thetas = self.thetas - self.alpha * j
				# if i == 0 or i == 1:
				# 	print(j)
				# # 
		else:
			x_prime = add_intercept(x)
			for i in range(0, self.max_iter):
				for k in range(0, len(x)):
					y_pred = self.predict_(x[k:k+1])
					j = (np.dot(np.transpose(x_prime[k:k+1]), y_pred - y[k:k+1])) / 1
					self.thetas = self.thetas - self.alpha * j
		# print(self.thetas)
		return self.thetas

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
		"""
			gives r2 score
		"""
		eq = np.equal(y, y_hat)
		count = 0
		for i in range(0, len(eq)):
			if y[i][0] == y_hat[i][0]:
				count +=1
		return count / len(y)
	
	def r2_(self, y, y_hat):
		r2 = 1 - (((y - y_hat) * (y - y_hat)).sum() / (((y - y.mean()) * (y - y.mean())).sum()))
		return r2
