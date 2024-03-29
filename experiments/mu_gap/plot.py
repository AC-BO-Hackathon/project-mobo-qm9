from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parents[2]))

from src.utils import plot_results

df = pd.read_csv("mu_gap_results.csv")
fig = plot_results(df, ["mu", "gap"], [True, False])
fig.tight_layout()
plt.savefig("mu_gap_results.png")
plt.show()

