from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
import logging
import time
from .base import TunerStrategy

logger = logging.getLogger(__name__)


class ManualTuner(TunerStrategy):
    """
    Executes a single training run with the fixed hyperparameters that were
    already baked into the pipeline by FraudDetectionTrainer._build_pipeline.

    In manual mode the trainer passes user hyperparameters directly to the
    model constructor (scalar values only — range dicts are pre-stripped by
    _create_model).  This tuner simply calls pipeline.fit() and returns.
    """

    def tune(
        self,
        pipeline: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Dict[str, Any],
        tuning_config: Dict[str, Any]
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:

        start_time = time.time()

        logger.info("[ManualTuner] Starting single training run (no hyperparameter search)")
        logger.info(f"[ManualTuner] Train shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"[ManualTuner] Label distribution: {y_train.value_counts().to_dict()}")

        # Log the model that was built
        try:
            model_step = pipeline.named_steps.get("model")
            logger.info(
                f"[ManualTuner] Model: {type(model_step).__name__} | "
                f"Params: {model_step.get_params()}"
            )
        except Exception:
            pass

        # Warn if any range-dict values sneak through — they won't be used
        # because the model was already constructed, but it helps debugging.
        range_keys = [k for k, v in hyperparameters.items() if isinstance(v, dict)]
        if range_keys:
            logger.warning(
                f"[ManualTuner] Received range-dict values for {range_keys}. "
                "These are IGNORED in manual mode (model was already built with defaults). "
                "Use grid/random/bayesian tuning to search over ranges."
            )

        # Log the effective scalar params that were actually baked into the model
        scalar_params = {
            k: v for k, v in hyperparameters.items()
            if not isinstance(v, dict)
        }
        logger.info(f"[ManualTuner] Effective hyperparameters used: {scalar_params}")

        # ── Fit ───────────────────────────────────────────────────────────────
        try:
            logger.info("[ManualTuner] Calling pipeline.fit() …")
            pipeline.fit(X_train, y_train)
        except Exception as e:
            logger.error(
                f"[ManualTuner] FAILED during pipeline.fit(): {e}",
                exc_info=True
            )
            raise

        elapsed = time.time() - start_time
        logger.info(f"[ManualTuner] ✅ Training complete in {elapsed:.1f}s")

        return pipeline, hyperparameters
