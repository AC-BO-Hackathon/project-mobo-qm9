from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
import matplotlib.pyplot as plt
import torch
import numpy as np
from botorch.utils.multi_objective.pareto import is_non_dominated


# Create
def plot_results(df, target_bool):
    """
    Plots the results for the MOBOQM9 model.
    
    args:
        df: Results for the MOBOQM9 model.
        target_bool: Targets for the MOBOQM9 model.
    
    returns:
        matplotlib.pyplot.figure: Figure for the MOBOQM9 model.
    """
    fig, ax = plt.subplots(2, 2, figsize=(20,10))
    ax = ax.flatten()
    targets = [col for col in df.columns if not col.startswith("iteration")]
    global_values = torch.tensor(df[targets].values)
    global_pareto_idx = is_non_dominated(global_values)
    global_pareto_front = global_values[global_pareto_idx]
    for mask in target_bool:
        if not mask:
            global_values[:, mask] *= -1
    ref_point = global_values.min(0)[0]
    hv = DominatedPartitioning(ref_point=ref_point, Y=global_values)
    global_hv = hv.compute_hypervolume().item()
    
    hv_iterations = {"qEHVI": [], "qNEHVI": [], "random": []}
    
    for acq in ["qEHVI", "qNEHVI", "random"]:
        max_iterations = int(df[f'iteration_{acq}'].max())
        for iter_no in range(max_iterations + 1):
            current_data = df[df[f'iteration_{acq}'] <= iter_no]
            local_values = torch.tensor(current_data[targets].values)
            for mask in target_bool:
                if not mask:
                    local_values[:, mask] *= -1
            hv = DominatedPartitioning(ref_point=ref_point, Y=local_values)
            hv_iterations[acq].append(hv.compute_hypervolume().item())
    ax[0].plot(np.array(hv_iterations["qEHVI"]) / global_hv * 100, marker='o', label="qEHVI")
    ax[0].plot(np.array(hv_iterations["qNEHVI"]) / global_hv * 100, marker='*', label="qNEHVI")
    ax[0].plot(np.array(hv_iterations["random"]) / global_hv * 100, marker='s', label="random")
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Hypervolume')
    ax[0].set_title('Hypervolume per Iteration')
    ax[0].legend(loc="center right")

    # Plot qEHVI
    max_iterations = int(df['iteration_qEHVI'].max())
    current_data = df[df['iteration_qEHVI'] <= max_iterations]
    qEHVI_values = torch.tensor(current_data[targets].values)
    qEHVI_pareto_idx = is_non_dominated(qEHVI_values)
    qEHVI_pareto_front = qEHVI_values[qEHVI_pareto_idx]
    ax[1].scatter(qEHVI_pareto_front[:, 0], qEHVI_pareto_front[:, 1],
        color='red', label="qEHVI Pareto Front", marker="*", s=100)
    ax[1].scatter(df[targets[0]], df[targets[1]], alpha=0.3)
    ax[1].scatter(global_pareto_front[:, 0], global_pareto_front[:, 1],
        color='green', label="Global Pareto Front", marker="s")
    ax[1].set_xlabel(targets[0])
    ax[1].set_ylabel(targets[1])
    ax[1].set_title('Pareto Front for qEHVI')
    ax[1].legend(loc="upper right")
    
    # Plot qNEHVI
    max_iterations = int(df['iteration_qNEHVI'].max())
    current_data = df[df['iteration_qNEHVI'] <= max_iterations]
    qNEHVI_values = torch.tensor(current_data[targets].values)
    qNEHVI_pareto_idx = is_non_dominated(qNEHVI_values)
    qNEHVI_pareto_front = qNEHVI_values[qNEHVI_pareto_idx]
    ax[2].scatter(qNEHVI_pareto_front[:, 0], qNEHVI_pareto_front[:, 1],
        color='red', label="qNEHVI Pareto Front", marker="*", s=100)
    ax[2].scatter(df[targets[0]], df[targets[1]], alpha=0.3)
    ax[2].scatter(global_pareto_front[:, 0], global_pareto_front[:, 1],
        color='green', label="Global Pareto Front", marker="s")
    ax[2].set_xlabel(targets[0])
    ax[2].set_ylabel(targets[1])
    ax[2].set_title('Pareto Front for qNEHVI')
    ax[2].legend(loc="upper right")
    
    # Plot random
    max_iterations = int(df['iteration_random'].max())
    current_data = df[df['iteration_random'] <= max_iterations]
    random_values = torch.tensor(current_data[targets].values)
    random_pareto_idx = is_non_dominated(random_values)
    random_pareto_front = random_values[random_pareto_idx]
    ax[3].scatter(random_pareto_front[:, 0], random_pareto_front[:, 1],
        color='red', label="Random Pareto Front", marker="*", s=100)
    ax[3].scatter(df[targets[0]], df[targets[1]], alpha=0.3)
    ax[3].scatter(global_pareto_front[:, 0], global_pareto_front[:, 1],
        color='green', label="Global Pareto Front", marker="s")
    ax[3].set_xlabel(targets[0])
    ax[3].set_ylabel(targets[1])
    ax[3].set_title('Pareto Front for random')
    ax[3].legend(loc="upper right")
    
    return fig