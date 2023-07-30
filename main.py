
from data import load_season
from models import Model1, Model2, Model3
from bracket_dp import optimal_bracket

def accuracy(model):
	print(f"Testing accuracy of {model}")

	correct = 0
	total = 0
	for year in range(2003, 2023):
		print(f"generating models: {year}", end='\r')

		teams, reg, cha = load_season(year)
		model.train(reg, teams)

		for game in cha:
			p = model.predict(game[0]["name"], game[1]["name"], teams)
			y = 1 if game[0]["pts"] > game[1]["pts"] else 0

			total += 1
			if (p >= 0.5) == y:
				correct += 1

	print("generating models: done")
	print(f"accruacy: {correct}/{total} = {correct / total}")

def calibration(model):
	print(f"Testing calibration of {model}")

	predictions = []
	for year in range(2003, 2023):
		print(f"generating models: {year}", end='\r')

		teams, reg, cha = load_season(year)
		model.train(reg, teams)

		for game in cha:
			p = model.predict(game[0]["name"], game[1]["name"], teams)
			y = 1 if game[0]["pts"] > game[1]["pts"] else 0

			if p < 0.5:
				p = 1 - p
				y = -1 * y + 1

			predictions.append((p, y))

	print("generating models: done")
	print("calibration:")

	predictions.sort()
	for q_int in range(51, 100):
		q = q_int / 100
		w = 0.03
		if q - 0.5 < w:
			w = q - 0.5
		elif 1 - q < w:
			w = 1 - q

		correct = 0
		total = 0
		for p, y in predictions:
			if p < q - w:
				continue
			elif p > q + w:
				break

			total += 1
			correct += y

		if total > 0:
			print(f"{q:.2f}: {correct:3d}/{total:3d} = {correct / total:.3f}")
		else:
			print(f"{q:.2f}: {correct:3d}/{total:3d}")

def predict_bracket(year, model):
	print(f"Creating bracket using {model}")
	print(f"generating model: {year}", end='\r')

	teams, reg, cha = load_season(year)
	model.train(reg, teams)

	print("generating model: done")
	print("generating bracket")

	def predict_func(Ta, Tb):
		return model.predict(Ta, Tb, teams)

	bracket = [
		"Alabama",
		"TAM C. Christi",
		"Maryland",
		"West Virginia",
		"San Diego St",
		"Col Charleston",
		"Virginia",
		"Furman",
		"Creighton",
		"NC State",
		"Baylor",
		"UC Santa Barbara",
		"Missouri",
		"Utah St",
		"Arizona",
		"Princeton",
		"Purdue",
		"F Dickinson",
		"Memphis",
		"FL Atlantic",
		"Duke",
		"Oral Roberts",
		"Tennessee",
		"Louisiana",
		"Kentucky",
		"Providence",
		"Kansas St",
		"Montana St",
		"Michigan St",
		"USC",
		"Marquette",
		"Vermont",
		"Houston",
		"N Kentucky",
		"Iowa",
		"Auburn",
		"Miami FL",
		"Drake",
		"Indiana",
		"Kent",
		"Iowa St",
		"Pittsburgh",
		"Xavier",
		"Kennesaw",
		"Texas A&M",
		"Penn St",
		"Texas",
		"Colgate",
		"Kansas",
		"Howard",
		"Arkansas",
		"Illinois",
		"St Mary's CA",
		"VCU",
		"Connecticut",
		"Iona",
		"TCU",
		"Arizona St",
		"Gonzaga",
		"Grand Canyon",
		"Northwestern",
		"Boise St",
		"UCLA",
		"UNC Asheville"
	]

	optimal_bracket(bracket, [0, 1, 2, 4, 8, 16, 32], predict_func)

model = Model1()
# accuracy(model)
# calibration(model)
predict_bracket(2023, model)
