
import numpy as np

from models.util import one_hot, binary_mov, pythagorean_mov
from classifiers import LogisticRegression, Perceptron

class Model2:
	# perceptron should only be used with binary_mov
	def __init__(self, mov=pythagorean_mov, classifier=LogisticRegression):
		self.internal_model = _Model2_Internal()
		self.classifier = classifier()
		self.mov = mov

	def train(self, games, teams):
		self.internal_model.train(games, teams)

		x = []
		y = []
		for game in games:
			Ma = self.internal_model.predict(game[0]["name"], game[1]["name"], game["site"], teams)
			Mb = self.internal_model.predict(game[1]["name"], game[0]["name"], game["site"] * -1, teams)
			x.append(np.array([Ma, Mb]))
			x.append(np.array([Mb, Ma]))

			Pa = game[0]["pts"]
			Pb = game[1]["pts"]
			y.append(self.mov(Pa, Pb))
			y.append(self.mov(Pb, Pa))

		self.classifier.train(x, y)

	def predict(self, Ta, Tb, teams):
		Ma = self.internal_model.predict(Ta, Tb, 0, teams)
		Mb = self.internal_model.predict(Tb, Ta, 0, teams)

		a = self.classifier.predict(np.array([Ma, Mb]))
		b = 1 - self.classifier.predict(np.array([Mb, Ma]))
		return (a + b) / 2

	def __str__(self):
		return f"Model-2 ({self.mov.__name__}, {type(self.classifier).__name__})"

class _Model2_Internal:
	def __init__(self):
		# assumes teams won't average over 2 points/possession
		self.classifier = LogisticRegression(2)

	def train(self, games, teams):
		x = []
		y = []
		for game in games:
			Ta = game[0]["name"]
			Tb = game[1]["name"]
			site = game["site"]
			x.append(one_hot(Ta, Tb, site, teams))
			x.append(one_hot(Tb, Ta, site * -1, teams))

			aposs = game[0]["fga"] - game[0]["or"] + game[0]["to"] + 0.475 * game[0]["fta"]
			bposs = game[1]["fga"] - game[1]["or"] + game[1]["to"] + 0.475 * game[1]["fta"]

			Pa = game[0]["pts"]
			Pb = game[1]["pts"]
			y.append(Pa / aposs)
			y.append(Pb / bposs)

		self.classifier.train(x, y)

	def predict(self, Ta, Tb, site, teams):
		return self.classifier.predict(one_hot(Ta, Tb, site, teams))
