
"""
Implementation of logistic regression.

Inputs:
	limit: int
	x: list<numpy array>
	y: list<float>
"""

import numpy as np

from classifiers.util import shuffle_examples, sigmoid, gamma

class LogisticRegression:
	def __init__(self, limit=1):
		self.L = limit

	def train(self, x, y):
		self.w = np.zeros(len(x[0]) + 1)

		for epoch in range(10):
			x, y = shuffle_examples(x, y)

			for i in range(len(x)):
				xi = np.append(x[i], [1])
				error = sigmoid(self.L, np.dot(xi, self.w)) - y[i]
				self.w -= error * gamma(epoch) * xi

	def predict(self, x):
		xc = np.append(x, [1])
		return sigmoid(self.L, np.dot(xc, self.w))
