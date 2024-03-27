from typing import NamedTuple, Literal, List
import numpy as np
from loguru import logger

from .data.cm_featurizer import get_coulomb_matrix


N_TOTAL_POINTS = 138_728

class MOBOQM9Parameters(NamedTuple):
    """
    Parameters for the MOBOQM9 model.
    
    args:
        featurizer: Featurizer to use.
        kernel: Kernel to use.
        surrogate_model: Surrogate model to use.
        acq_func: Acquisition function to use.
        targets: List of targets to optimize.
        target_bools: List of booleans indicating wheather to minimize or 
            maximize each target.
        num_candidates: Number of candidates to optimize. Default is 1.
        num_total_points: Number of total points to use. Default is 2000.
        num_seed_points: Number of seed points to use. Default is 100.
        n_iters: Number of iterations. Default is 20.
    """
    featurizer: Literal["ECFP", "CM", "ACSF"]
    kernel: Literal["RBF", "Matern", "Tanimoto"]
    surrogate_model: Literal["GaussianProcess", "RandomForest"]
    acq_func: Literal["qEHVI", "qparEGO"]
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
    
    def get_features_and_targets(self):
        """
        Gets the features and targets for the MOBOQM9 model.
        
        returns:
            features: Features for the MOBOQM9 model.
            targets: Targets for the MOBOQM9 model.
        """
        # ecfp and soap
        if self.params.featurizer == "CM":
            features, targets = get_coulomb_matrix(self.total_indices,
                                                   self.params.targets)
        else:
            raise NotImplementedError
    
    def get_surrogate_model(self, X, y):
        """
        Gets the surrogate model for the MOBOQM9 model.
        
        args:
            X: Features for the MOBOQM9 model.
            y: Targets for the MOBOQM9 model.
            
        returns:
            model: Surrogate model for the MOBOQM9 model.
        """
        y_copy = y.copy()
        for idx, mask in enumerate(self.params.target_bools):
            if not mask:
                y_copy[:, idx] *= -1
        # build the model
        pass
    
    def optimize_acquisition_function(self, model):
        """
        Optimizes the acquisition function for the MOBOQM9 model.
        
        args:
            model: Surrogate model for the MOBOQM9 model.
        
        returns:
            candidates: Candidates for the MOBOQM9 model.
        """
        pass
    
    def run_optimization(self):
        """
        Runs the MOBOQM9 optimization.
        """
        for iter in range(self.params.n_iters):
            X = self.features[self.train_indices]
            y = self.targets[self.train_indices]
            model = self.get_surrogate_model(X, y)
            candidates = self.optimize_acquisition_function(model)
            self.update_train_indices(candidates)
            if self.stopping_criteria_met():
                break
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
        return self.total_indices[temp_indices]
    
    def stopping_criteria_met(self):
        """
        Checks if the MOBOQM9 optimization has met the stopping criteria.
        
        returns:
            bool: True if the MOBOQM9 optimization has met the stopping criteria.
        """
        # if current hv == best hv, then stop.
        pass
    
    def update_train_indices(self, candidates):
        """
        Updates the train indices for the MOBOQM9 model.
        
        args:
            candidates: Candidates for the MOBOQM9 model.
        """
        pass
    
    def validate_params(self):
        """
        Validates the parameters for the MOBOQM9 model.
        
        args:
            params: Parameters for the MOBOQM9 model.
        """
        assert self.params.featurizer in ["ECFP", "CM", "ACSF"], "Featurizer must be one of ECFP, CM, or ACSF."
        assert self.params.kernel in ["RBF", "Matern", "Tanimoto"], "Kernel must be one of RBF, Matern, or Tanimoto."
        assert self.params.surrogate_model in ["GaussianProcess", "RandomForest"], "Surrogate model must be one of GaussianProcess, or RandomForest."
        assert self.params.acq_func in ["qEHVI", "qparEGO"], "Acquisition function must be one of qEHVI, or qparEGO."
        assert len(self.params.targets) == len(self.params.target_bools), "Number of targets must equal number of target booleans."
        assert self.params.num_total_points > 0, "Number of total points must be greater than zero."
        assert self.params.num_seed_points > 0, "Number of seed points must be greater than zero."
        assert self.params.n_iters > 0, "Number of iterations must be greater than zero."