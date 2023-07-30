
## Optimal Brackets

Implementation of the dynamic programming algorithm given in "March Madness and the Office Pool" (Kaplan 2001) which computes a bracket optimizing expected points given a win probability model.
The algorithm can be used interchangeably with the following simple power ratings.

* Model 1: Directly predicts probability of victory from margin of victory (mov).
* Model 2: Predicts points/possession for each team and converts to a win probability.
* Model 3: Predicts shooting, rebounding, and turnovers and converts to a win probability.
