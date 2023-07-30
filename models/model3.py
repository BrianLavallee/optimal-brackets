
import numpy as np

from models.util import one_hot, binary_mov, pythagorean_mov
from classifiers import LogisticRegression, Perceptron

def _two_mov(game):
	Aa = game[0]["fga"] - game[0]["3a"]
	Ma = game[0]["fgm"] - game[0]["3m"]
	Ab = game[1]["fga"] - game[1]["3a"]
	Mb = game[1]["fgm"] - game[1]["3m"]

	return Ma / Aa, Mb / Ab

def _three_mov(game):
	return game[0]["3m"] / game[0]["3a"], game[1]["3m"] / game[1]["3a"]

def _reb_mov(game):
	Oa = game[0]["or"]
	Da = game[0]["dr"]
	Ob = game[1]["or"]
	Db = game[1]["dr"]

	return Oa / (Oa + Db), Ob / (Da + Ob)

def _tov_mov(game):
	aposs = game[0]["fga"] - game[0]["or"] + game[0]["to"] + 0.475 * game[0]["fta"]
	bposs = game[1]["fga"] - game[1]["or"] + game[1]["to"] + 0.475 * game[1]["fta"]

	return game[0]["to"] / aposs, game[1]["to"] / bposs

class Model3:
	# perceptron should only be used with binary_mov
	def __init__(self, mov=pythagorean_mov, classifier=LogisticRegression):
		self.two_model = _Model3_Internal(_two_mov)
		self.three_model = _Model3_Internal(_three_mov)
		self.reb_model = _Model3_Internal(_reb_mov)
		self.tov_model = _Model3_Internal(_tov_mov)

		self.classifier = classifier()
		self.mov = mov

	def train(self, games, teams):
		self.two_model.train(games, teams)
		self.three_model.train(games, teams)
		self.reb_model.train(games, teams)
		self.tov_model.train(games, teams)

		x = []
		y = []
		for game in games:
			Ta = game[0]["name"]
			Tb = game[1]["name"]
			site = game["site"]

			TWa = self.two_model.predict(Ta, Tb, site, teams)
			TWb = self.two_model.predict(Tb, Ta, site * -1, teams)

			THa = self.three_model.predict(Ta, Tb, site, teams)
			THb = self.three_model.predict(Tb, Ta, site * -1, teams)

			Ra = self.reb_model.predict(Ta, Tb, site, teams)
			Rb = self.reb_model.predict(Tb, Ta, site * -1, teams)

			TOa = self.tov_model.predict(Ta, Tb, site, teams)
			TOb = self.tov_model.predict(Tb, Ta, site * -1, teams)

			x.append(np.array([TWa, TWb, THa, THb, Ra, Rb, TOa, TOb]))
			x.append(np.array([TWb, TWa, THb, THa, Rb, Ra, TOb, TOa]))

			Pa = game[0]["pts"]
			Pb = game[1]["pts"]
			y.append(self.mov(Pa, Pb))
			y.append(self.mov(Pb, Pa))

		self.classifier.train(x, y)

	def predict(self, Ta, Tb, teams):
		TWa = self.two_model.predict(Ta, Tb, 0, teams)
		TWb = self.two_model.predict(Tb, Ta, 0, teams)

		THa = self.three_model.predict(Ta, Tb, 0, teams)
		THb = self.three_model.predict(Tb, Ta, 0, teams)

		Ra = self.reb_model.predict(Ta, Tb, 0, teams)
		Rb = self.reb_model.predict(Tb, Ta, 0, teams)

		TOa = self.tov_model.predict(Ta, Tb, 0, teams)
		TOb = self.tov_model.predict(Tb, Ta, 0, teams)

		a = self.classifier.predict(np.array([TWa, TWb, THa, THb, Ra, Rb, TOa, TOb]))
		b = 1 - self.classifier.predict(np.array([TWb, TWa, THb, THa, Rb, Ra, TOb, TOa]))
		return (a + b) / 2

	def __str__(self):
		return f"Model-3 ({self.mov.__name__}, {type(self.classifier).__name__})"

class _Model3_Internal:
	# mov behaves slightly differently than those in util
	def __init__(self, mov):
		self.classifier = LogisticRegression()
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

			Ma, Mb = self.mov(game)
			y.append(Ma)
			y.append(Mb)

		self.classifier.train(x, y)

	def predict(self, Ta, Tb, site, teams):
		return self.classifier.predict(one_hot(Ta, Tb, site, teams))
