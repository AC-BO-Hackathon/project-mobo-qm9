from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))

from src.utils import plot_results

df = pd.read_csv("mu_G_results.csv")
fig = plot_results(df, ["mu", "G_free_energy"], [True, False])
fig.tight_layout()
# plt.savefig("mu_G_results.png")
plt.show()

