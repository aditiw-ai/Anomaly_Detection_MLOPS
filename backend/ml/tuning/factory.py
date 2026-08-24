from typing import Dict, Type
from .strategies.base import TunerStrategy
from .strategies.manual import ManualTuner
from .strategies.grid import GridSearchTuner
from .strategies.random import RandomSearchTuner
from .strategies.bayesian import BayesianTuner

class TunerFactory:
    """Factory to create tuner instances."""
    
    _tuners: Dict[str, Type[TunerStrategy]] = {
        "manual": ManualTuner,
        "grid": GridSearchTuner,
        "random": RandomSearchTuner,
        "bayesian": BayesianTuner
    }
    
    @classmethod
    def get_tuner(cls, method: str) -> TunerStrategy:
        """Get tuner strategy by name."""
        tuner_cls = cls._tuners.get(method.lower())
        if not tuner_cls:
            raise ValueError(f"Unknown tuning method: {method}. Available: {list(cls._tuners.keys())}")
        return tuner_cls()
