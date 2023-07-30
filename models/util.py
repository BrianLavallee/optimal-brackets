
import numpy as np

def one_hot(Ta, Tb, site, teams):
	x = np.zeros(len(teams) * 2 + 1)
	x[teams[Ta]] = 1
	x[teams[Tb] + len(teams)] = 1
	x[len(teams) * 2] = site
	return x

def binary_mov(Pa, Pb):
	return 1 if Pa > Pb else 0

def pythagorean_mov(Pa, Pb):
	za = Pa ** 10.25
	zb = Pb ** 10.25
	return za / (za + zb)
