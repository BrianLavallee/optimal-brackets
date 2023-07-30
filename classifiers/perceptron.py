
"""
Implementation of perceptron classifier.

Inputs:
	x: list<numpy array>
	y: list<float>
"""

import numpy as np

from classifiers.util import shuffle_examples

class Perceptron:
	def __init__(self):
		pass

	def train(self, x, y):
		self.w = [np.zeros(len(x[0]) + 1)]
		self.c = [0]

		for epoch in range(10):
			x, y = shuffle_examples(x, y)

			for i in range(len(x)):
				xi = np.append(x[i], [1])
				yi = y[i] * 2 - 1
				pi = np.sign(np.dot(xi, self.w[-1]))

				if pi != yi:
					self.w.append(self.w[-1] + yi * xi)
					self.c.append(0)

				self.c[-1] += 1

	def predict(self, x):
		xc = np.append(x, [1])

		p = 0
		t = 0
		for i in range(len(self.w)):
			pi = np.dot(xc, self.w[i])
			p += self.c[i] if pi > 0 else 0
			t += self.c[i]

		return p / t
