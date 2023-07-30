
from models.util import one_hot, binary_mov, pythagorean_mov
from classifiers import LogisticRegression, Perceptron

class Model1:
	# perceptron should only be used with binary_mov
	def __init__(self, mov=pythagorean_mov, classifier=LogisticRegression):
		self.classifier = classifier()
		self.mov = mov

	def train(self, games, teams):
		x = []
		y = []
		for game in games:
			Ta = game[0]["name"]
			Tb = game[1]["name"]
			site = game["site"]
			x.append(one_hot(Ta, Tb, site, teams))
			x.append(one_hot(Tb, Ta, site * -1, teams))

			Pa = game[0]["pts"]
			Pb = game[1]["pts"]
			y.append(self.mov(Pa, Pb))
			y.append(self.mov(Pb, Pa))

		self.classifier.train(x, y)

	def predict(self, Ta, Tb, teams):
		a = self.classifier.predict(one_hot(Ta, Tb, 0, teams))
		b = 1 - self.classifier.predict(one_hot(Tb, Ta, 0, teams))
		return (a + b) / 2

	def __str__(self):
		return f"Model-1 ({self.mov.__name__}, {type(self.classifier).__name__})"
