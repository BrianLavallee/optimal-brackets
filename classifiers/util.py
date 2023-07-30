
import random
import math

def shuffle_examples(x, y):
	temp = list(zip(x, y))
	random.shuffle(temp)
	x, y = zip(*temp)
	return x, y

def sigmoid(L, x):
	return L / (1 + math.e ** -x)

def gamma(t):
	return 1 / (1 + t)
