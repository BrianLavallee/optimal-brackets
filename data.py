
from csv import DictReader

def load_season(year):
	teams = {}
	with open("boxscores/teams.csv", "r") as f:
		reader = DictReader(f)
		for row in reader:
			teams[row["TeamID"]] = row["TeamName"]

	reg = []
	with open("boxscores/regular/{}.csv".format(year), "r") as f:
		reader = DictReader(f)
		for game in reader:
			reg.append(_parse(game, teams))

	cha = []
	with open("boxscores/tourney.csv", "r") as f:
		reader = DictReader(f)
		for game in reader:
			if int(game["Season"]) == year:
				cha.append(_parse(game, teams))

	teamid = {}
	i = 0
	for id in teams:
		teamid[teams[id]] = i
		i += 1

	return teamid, reg, cha

def _parse(game, teams):
	g = {}
	g["site"] = 0
	if game["WLoc"] == "H":
		g["site"] = 1
	elif game["WLoc"] == "A":
		g["site"] = -1

	Wbox = {}
	Wbox["name"] = teams[game["WTeamID"]]
	Wbox["pts"] = int(game["WScore"])
	Wbox["fgm"] = int(game["WFGM"])
	Wbox["fga"] = int(game["WFGA"])
	Wbox["3m"] = int(game["WFGM3"])
	Wbox["3a"] = int(game["WFGA3"])
	Wbox["ftm"] = int(game["WFTM"])
	Wbox["fta"] = int(game["WFTA"])
	Wbox["or"] = int(game["WOR"])
	Wbox["dr"] = int(game["WDR"])
	Wbox["to"] = int(game["WTO"])

	Lbox = {}
	Lbox["name"] = teams[game["LTeamID"]]
	Lbox["pts"] = int(game["LScore"])
	Lbox["fgm"] = int(game["LFGM"])
	Lbox["fga"] = int(game["LFGA"])
	Lbox["3m"] = int(game["LFGM3"])
	Lbox["3a"] = int(game["LFGA3"])
	Lbox["ftm"] = int(game["LFTM"])
	Lbox["fta"] = int(game["LFTA"])
	Lbox["or"] = int(game["LOR"])
	Lbox["dr"] = int(game["LDR"])
	Lbox["to"] = int(game["LTO"])

	if Wbox["name"] < Lbox["name"]:
		g[0] = Wbox
		g[1] = Lbox
	else:
		g[1] = Wbox
		g[0] = Lbox
		g["site"] *= -1

	return g
