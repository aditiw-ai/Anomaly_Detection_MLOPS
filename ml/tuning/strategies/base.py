from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator

class TunerStrategy(ABC):
    """Abstract base class for hyperparameter tuning strategies."""
    
    @abstractmethod
    def tune(
        self,
        pipeline: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Dict[str, Any],
        tuning_config: Dict[str, Any]
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Execute the tuning strategy.
        
        Args:
            pipeline: Scikit-learn pipeline to tune
            X_train: Training features
            y_train: Training labels
            hyperparameters: Parameter grid or configuration
            tuning_config: Strategy-specific config (e.g., n_iter, cv)
            
        Returns:
            Tuple of (best_model, best_params)
        """
        pass
