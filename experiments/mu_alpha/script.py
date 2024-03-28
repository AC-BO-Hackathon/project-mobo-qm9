from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[2]))

from src.mobo_qm9 import MOBOQM9, MOBOQM9Parameters
from src.utils import plot_results

params = MOBOQM9Parameters(featurizer="CM",
                           kernel="Matern",
                           surrogate_model="GaussianProcess",
                           targets=["mu", "alpha"],
                           target_bools=[True, True],
                           num_total_points=1000,
                           num_seed_points=50,
                           n_iters=50,
                           num_candidates=1)

moboqm9 = MOBOQM9(params)
moboqm9.run_optimization()
fig = plot_results(moboqm9.dataframe, [True, True])
fig.tight_layout()
plt.savefig("mu_alpha_results.png")
plt.show()
moboqm9.dataframe.to_csv("mu_alpha_results.csv")

