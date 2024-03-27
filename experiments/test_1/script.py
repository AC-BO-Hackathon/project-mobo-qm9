from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))

from src.mobo_qm9 import MOBOQM9, MOBOQM9Parameters
from src.utils import plot_results

params = MOBOQM9Parameters(featurizer="CM",
                           kernel="RBF",
                           surrogate_model="GaussianProcess",
                           targets=["gap", "mu"],
                           target_bools=[True, True],
                           num_total_points=100,
                           num_seed_points=10,
                           n_iters=100,
                           num_candidates=10)

moboqm9 = MOBOQM9(params)
moboqm9.run_optimization()
plt = plot_results(moboqm9.dataframe)
# plt.savefig("figures/moboqm9_results.png")
plt.show()

"""
panel1 => iteration vs hv [qEHVI qNEHVI random]
panel2 => pareto front for qEHVI
panel3 => pareto front for qNEHVI
panel4 => pareto front for random
"""

