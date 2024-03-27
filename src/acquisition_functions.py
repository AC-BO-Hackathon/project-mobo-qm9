from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import (
    optimize_acqf_discrete,
)
import torch

def optimize_qEHVI(model, reference, y_train, x_test, n_candidates):
    """
    Optimizes the qEHVI acquisition function for the MOBOQM9 model.
    
    args:
        model: Surrogate model for the MOBOQM9 model.
        reference: Reference points for the MOBOQM9 model.
        y_train: Targets for the MOBOQM9 model.
        x_test: Test points for the MOBOQM9 model.
        n_candidates: Number of candidates for the MOBOQM9 model.
    
    returns:
        candidates: Candidates for the MOBOQM9 model.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    partitioning = NondominatedPartitioning(
        ref_point=reference, Y=y_train
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        sampler=sampler,
        ref_point=reference,
        partitioning=partitioning,
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        choices=x_test, 
        q=n_candidates,
        unique=True
    )
    return torch.tensor(candidates)

def optimize_qNEHVI(model, reference, x_train, x_test, n_candidates):
    """
    Optimizes the qNEHVI acquisition function for the MOBOQM9 model.
    
    args:
        model: Surrogate model for the MOBOQM9 model.
        reference: Reference points for the MOBOQM9 model.
        x_train: Training points for the MOBOQM9 model.
        x_test: Test points for the MOBOQM9 model.
        n_candidates: Number of candidates for the MOBOQM9 model.
    
    returns:
        candidates: Candidates for the MOBOQM9 model.
    """
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference,
        X_baseline=x_train,
        prune_baseline=True,
        sampler=sampler,
    )
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        choices=x_test,
        q=n_candidates,
        unique=True
    )
    return torch.tensor(candidates)