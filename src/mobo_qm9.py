from typing import NamedTuple, Literal, List
import numpy as np
from loguru import logger
import torch
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.models import ModelListGP, SingleTaskGP
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import pandas as pd

from .data.cm_featurizer import get_coulomb_matrix
from .acquisition_functions import optimize_qEHVI, optimize_qNEHVI

N_TOTAL_POINTS = 138_728

class MOBOQM9Parameters(NamedTuple):
    """
    Parameters for the MOBOQM9 model.
    
    args:
        featurizer: Featurizer to use.
        kernel: Kernel to use.
        surrogate_model: Surrogate model to use.
        targets: List of targets to optimize.
        target_bools: List of booleans indicating wheather to minimize or 
            maximize each target.
        num_candidates: Number of candidates to optimize. Default is 1.
        num_total_points: Number of total points to use. Default is 2000.
        num_seed_points: Number of seed points to use. Default is 100.
        n_iters: Number of iterations. Default is 20.
    """
    featurizer: Literal["ECFP", "CM", "ACSF"]
    kernel: Literal["RBF", "Matern"]
    surrogate_model: Literal["GaussianProcess", "RandomForest"]
    targets: List[str]
    target_bools: List[bool]
    num_candidates: int = 1
    num_total_points: int = 2000
    num_seed_points: int = 100
    n_iters: int = 20

class MOBOQM9:
    """
    Class for the MOBOQM9 model.
    """
    def __init__(self, params: MOBOQM9Parameters):
        """
        Initializes the MOBOQM9 model.
        
        args:
            params: Parameters for the MOBOQM9 model.
        """
        self.params = params
        self.validate_params()
        self.total_indices = np.random.randint(0, N_TOTAL_POINTS, 
            self.params.num_total_points)
        self.features, self.targets = self.get_features_and_targets()
        self.train_indices = self.get_train_indices()
        self.dataframe = pd.DataFrame.from_dict(self.from_target_dict())
        self.acq_met = {"qEHVI": False, "qNEHVI": False, "random": False}

    def form_target_dict(self):
        """
        Forms the target dictionary for the MOBOQM9 model.
        
        returns:
            target_dict: Target dictionary for the MOBOQM9 model.
        """
        target_dict = {"iteration": None}
        for i, target in enumerate(self.params.targets):
            target_dict[target] = self.targets[:, i]
            target_dict["target_qEHVI"] = None
            target_dict["target_qNEHVI"] = None
            target_dict["target_random"] = None
        return target_dict
    
    def get_features_and_targets(self):
        """
        Gets the features and targets for the MOBOQM9 model.
        
        returns:
            features: Features for the MOBOQM9 model.
            targets: Targets for the MOBOQM9 model.
        """
        # ecfp and soap
        if self.params.featurizer == "CM":
            return get_coulomb_matrix(self.total_indices,
                self.params.targets)
        else:
            raise NotImplementedError
    
    def get_surrogate_model(self, acq):
        """
        Gets the surrogate model for the MOBOQM9 model.
            
        args:
            acq: Acquisition function to use.
        
        returns:
            model: Surrogate model for the MOBOQM9 model.
        """
        features = torch.tensor(self.features[self.train_indices["acq"]],
                                dtype=torch.double)
        targets = torch.tensor(self.correct_sign(self.targets[self.train_indices["acq"]]),
                                dtype=torch.double)
        var = torch.full_like(targets, 1e-6)

        if self.params.kernel == 'RBF':
            kernel = RBFKernel()
        elif self.params.kernel == 'Matern':
            kernel = MaternKernel()
        else:
            raise ValueError("Unsupported kernel type. Supported types are 'RBF', and 'Matern'.")

        models = [SingleTaskGP(features,
                               targets[:, i].unsqueeze(-1),
                               noise=var[:, i].unsqueeze(-1),
                               input_transform=Normalize(d=features.shape[-1]),
                               outcome_transform=Standardize(m=1),
                               likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                               mean_module=gpytorch.means.ConstantMean(),
                               covar_module=kernel) 
                  for i in range(targets.shape[1])]

        model = ModelListGP(*models)
        mll = gpytorch.mlls.SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def correct_sign(self, Y):
        y_copy = Y.copy()
        for idx, mask in enumerate(self.params.target_bools):
            if not mask:
                y_copy[:, idx] *= -1
        return y_copy
    
    def optimize_acquisition_function(self, model, acq):
        """
        Optimizes the acquisition function for the MOBOQM9 model.
        
        args:
            model: Surrogate model for the MOBOQM9 model.
            acq: Acquisition function to use.
        
        returns:
            candidates: Candidates for the MOBOQM9 model.
        """
        y_train = self.correct_sign(self.targets[self.train_indices["acq"]])
        y_train = torch.tensor(y_train, dtype=torch.double)
        x_train = torch.tensor(self.features[self.train_indices["acq"]], dtype=torch.double)
        x_test = torch.tensor(self.features[~self.train_indices["acq"]], dtype=torch.double)
        reference = y_train.mean(0)[0]
        if acq == "qEHVI":
            return optimize_qEHVI(model=model,
                                  reference=reference,
                                  y_train=y_train,
                                  x_test=x_test,
                                  n_candidates=self.params.num_candidates)
        elif acq == "qNEHVI":
            return optimize_qNEHVI(model=model,
                                   reference=reference,
                                   x_train=x_train,
                                   x_test=x_test,
                                   n_candidates=self.params.num_candidates)
        else:
            raise NotImplementedError
    
    def run_optimization(self):
        """
        Runs the MOBOQM9 optimization.
        """
        for iter in range(self.params.n_iters):
            logger.info(f"MOBOQM9 iteration {iter + 1} of {self.params.n_iters}.")
            for acq in ["qEHVI", "qNEHVI", "random"]:
                if self.acq_met[acq]:
                    continue
                model = self.get_surrogate_model(acq)
                if acq == "random":
                    for _ in range(self.params.num_candidates):
                        idx = np.random.choice(np.where(~self.train_indices)[0])
                        self.train_indices[acq][idx] = True
                else:
                    candidates = self.optimize_acquisition_function(model)
                    self.update_train_indices(candidates, acq)
                self.stopping_criteria_met(acq)
                
        logger.info("MOBOQM9 optimization finished.")
    
    def get_train_indices(self):
        """
        Gets the train indices for the MOBOQM9 model.
        
        returns:
            train_indices: Train indices for the MOBOQM9 model.
        """
        # add latin hypercube sampling if time permits
        temp_indices = np.random.randint(0, self.params.num_total_points,
            self.params.num_seed_points)
        mask = np.zeros(len(self.total_indices), dtype=bool)
        mask[temp_indices] = True
        return {"qEHVI": mask, "qNEHVI": mask, "random": mask}
    
    def stopping_criteria_met(self, acq):
        """
        Checks if the MOBOQM9 optimization has met the stopping criteria.
        
        returns:
            bool: True if the MOBOQM9 optimization has met the stopping criteria.
        """
        y_global = torch.tensor(self.targets)
        y_current = torch.tensor(self.targets[self.train_indices[acq]])
        ref_points = y_global.min(0)[0]
        bd_global = DominatedPartitioning(
            ref_point=ref_points,
            Y=y_global,
        )
        volume_global = bd_global.compute_hypervolume().item()
        bd_current = DominatedPartitioning(
            ref_point=ref_points,
            Y=y_current,
        )
        volume_current = bd_current.compute_hypervolume().item()
        self.acq_met[acq] = (volume_global == volume_current)
        
    
    def update_train_indices(self, candidates, acq):
        """
        Updates the train indices for the MOBOQM9 model.
        
        args:
            candidates: Candidates for the MOBOQM9 model.
            acq: Acquisition function to use.
        """
        for cand in candidates:
            for idx, feat in enumerate(self.features):
                if np.allclose(feat, cand):
                    self.train_indices[acq][idx] = True
    
    def validate_params(self):
        """
        Validates the parameters for the MOBOQM9 model.
        
        args:
            params: Parameters for the MOBOQM9 model.
        """
        assert self.params.featurizer in ["ECFP", "CM", "ACSF"], "Featurizer must be one of ECFP, CM, or ACSF."
        assert self.params.kernel in ["RBF", "Matern", "Tanimoto"], "Kernel must be one of RBF, Matern."
        assert self.params.surrogate_model in ["GaussianProcess", "RandomForest"], "Surrogate model must be one of GaussianProcess, or RandomForest."
        assert len(self.params.targets) == len(self.params.target_bools), "Number of targets must equal number of target booleans."
        assert self.params.num_total_points > 0, "Number of total points must be greater than zero."
        assert self.params.num_seed_points > 0, "Number of seed points must be greater than zero."
        assert self.params.n_iters > 0, "Number of iterations must be greater than zero."
