
"""
Dynamic programming algorithm to optimize expected points in a single-elimination bracket prediction pool.
Computes osb[r, i] = the optimal sub-bracket on the first r rounds such that team ti wins.

Inputs:
    tournament tree: list<teamID>
    points per round: list<int>
    prediction model: func(teamID, teamID)

Reference:
    Kaplan, Garstka. March Madness and the Office Pool. 2001.
"""

# returns range of teams that could play ti in round r
def _legal_opponents(i, r):
    # shift amount
    sa = r - 1

    # index of oppenent in round r
    opp = (i >> sa) ^ 1

    # range of teams with correct index in round r
    return range(opp << sa, (opp << sa) + (1 << sa))

# main function
def optimal_bracket(teams, ppr, predict_func):
    # osb table stores 3 values:
    # expected points of optimal sub-bracket
    # P(ti wins first r games)
    # best opponent reference
    osb = {}

    # base case
    for i in range(len(teams)):
        osb[0, i] = (0.0, 1.0, -1)

    rounds = len(teams).bit_length() - 1

    # compute osb values
    for r in range(1, rounds + 1):
        for i in range(len(teams)):
            # P(ti wins round r | ti won round r-1)
            win_chance = 0.0

            # best opponent to choose to lose to ti
            best_points = 0.0
            best_opp = -1

            for opp in _legal_opponents(i, r):
                # P(beat opp) * P(opp reaches round r)
                win_chance += predict_func(teams[i], teams[opp]) * osb[r-1, opp][1]

                if osb[r-1, opp][0] > best_points:
                    best_points = osb[r-1, opp][0]
                    best_opp = opp

            chance = win_chance * osb[r-1, i][1]
            points = osb[r-1, i][0] + best_points + (chance * ppr[r])

            osb[r, i] = (points, chance, best_opp)

    # extract picks from osb
    best_points = 0.0
    best_team = -1
    for i in range(len(teams)):
        if osb[rounds, i][0] > best_points:
            best_points = osb[rounds, i][0]
            best_team = i

    picks = [[best_team]]
    for r in range(rounds-1):
        picks.insert(0, [])
        for i in picks[1]:
            picks[0].append(i)
            picks[0].append(osb[rounds-r, i][2])

        picks[0].sort()

    # printing picks
    print()
    for r in range(len(picks)):
        print("Round {}:".format(r+1))
        for i in picks[r]:
            print("\t{}".format(teams[i]))

        print()
