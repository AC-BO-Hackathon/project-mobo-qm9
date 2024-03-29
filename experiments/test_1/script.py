from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[2]))

from src.mobo_qm9 import MOBOQM9, MOBOQM9Parameters
from src.utils import plot_results

params = MOBOQM9Parameters(featurizer="CM",
                           kernel="RBF",
                           surrogate_model="GaussianProcess",
                           targets=["gap", "mu"],
                           target_bools=[True, True],
                           num_total_points=1000,
                           num_seed_points=100,
                           n_iters=50,
                           num_candidates=1)

moboqm9 = MOBOQM9(params)
moboqm9.run_optimization()
fig = plot_results(moboqm9.dataframe, [True, True])
fig.tight_layout()
plt.savefig("moboqm9_results_soap.png")
plt.show()
moboqm9.dataframe.to_csv("moboqm9_results_soap.csv")

