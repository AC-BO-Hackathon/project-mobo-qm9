import pandas as pd
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.utils.multi_objective.hypervolume import Hypervolume

data = {
    "iteration": list(range(1, 11)),
    "target_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "target_2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "target_1_qeHVI": [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.01],
    "target_2_qeHVI": [10.1, 9.2, 8.3, 7.4, 6.5, 5.6, 4.7, 3.8, 2.9, 1.1],
    "target_1_qNEHVI": [0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.02, 1.11],
    "target_2_qNEHVI": [10.2, 9.3, 8.4, 7.5, 6.6, 5.7, 4.8, 3.9, 2.8, 1.2],
    "target_1_random": [0.09, 0.18, 0.29, 0.43, 0.54, 0.61, 0.71, 0.81, 0.92, 1.03],
    "target_2_random": [9.8, 8.7, 7.6, 6.9, 6.0, 5.2, 4.4, 3.3, 2.4, 1.5]
}

df=pd.DataFrame.from_dict(data)

# Create
def plot_results(df,max_iterations):
    """
    Plots the results for the MOBOQM9 model.
    
    args:
        df: Results for the MOBOQM9 model.
    
    returns:
        matplotlib.pyplot.figure: Figure for the MOBOQM9 model.
    """
    fig,ax=plt.subplots(1,2,figsize=(20,4))
    ref_point=[1.0,1.0]
    hv = Hypervolume(ref_point)
    global_hv=hv.compute(df.values[:,1:3])
    hv_iterations=[]

    for iter_no in range(1, max_iterations + 1):
        current_data=df.iloc[:i,1:3].values
        current_hv=hv.compute(current_data)
        hv_iterations.append(current_hv)
        ax[0].plot(range(1, max_iterations + 1), iteration_hvs, marker='o')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Hypervolume')
        ax[0].set_title('Hypervolume per Iteration')

    last_iteration_data = df.iloc[max_iterations - 1, 1:3].values
    last_iteration_hv = hv.compute(last_iteration_data)
    last_iteration_pareto_front = DominatedPartitioning(ref_point=ref_point, Y=last_iteration_data)

    all_data_pareto_front = DominatedPartitioning(ref_point=ref_point, Y=df.values[:, 1:3])

    ax[1].scatter(df['target_1'], df['target_2'], alpha=0.5, label='All Data')

    global_pareto_front = all_data_pareto_front[0]

    ax[1].plot(global_pareto_front[:, 0], global_pareto_front[:, 1], color='b', label='Global Pareto Front')

    ax[1].scatter(last_iteration_pareto_front[:, 0], last_iteration_pareto_front[:, 1], color='r', label='Last Iteration Pareto Front')

    ax[1].set_xlabel('Target 1')
    ax[1].set_ylabel('Target 2')
    ax[1].legend()

    # compute the global hv
    # compute the hv for each iteration up until that point
    # plot iteration vs hv
    # panel 2 - 4
    # find the pareto front of the last iteration 
    # find the pareto front of all the data
    # scatter plot all the data with alpha = 0.5
    # global pareto front with blue
    # last iteration pareto front with red *
    return fig
