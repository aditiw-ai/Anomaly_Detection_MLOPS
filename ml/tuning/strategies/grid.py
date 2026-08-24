from typing import Dict, Any, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import logging
from .base import TunerStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse(val: Any):
    """
    Normalize frontend string encodings to Python values.
    Frontend often sends "True"/"False"/"None" as strings.
    """
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "true":
            return True
        if v == "false":
            return False
        if v == "none":
            return None
    return val

# Max discrete values per hyperparameter in the grid.
# Keeps total combinations manageable at enterprise scale.
_MAX_VALUES_PER_PARAM = 5

# Dataset size threshold above which we subsample for CV.
# Final model is always trained on the FULL dataset.
_LARGE_DATASET_THRESHOLD = 500_000
_CV_SUBSAMPLE_SIZE = 200_000

# Optimised search space for XGBoost on imbalanced fraud data.
# Based on empirical tuning across financial datasets.
_XGBOOST_OPTIMISED_GRID = {
    "model__n_estimators":    [100, 200, 300],
    "model__max_depth":       [3, 5, 7],
    "model__learning_rate":   [0.01, 0.05, 0.1, 0.2],
    "model__subsample":       [0.6, 0.8, 1.0],
    "model__colsample_bytree":[0.6, 0.8, 1.0],
}


class GridSearchTuner(TunerStrategy):
    """
    Enterprise-grade grid search with:
    - Smart parameter sampling (max 5 values/param).
    - Stratified CV subsampling for large datasets (>500K rows).
    - Full-data final fit after best params are selected.
    """

    def tune(
        self,
        pipeline: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Dict[str, Any],
        tuning_config: Dict[str, Any]
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:

        logger.info("Starting Enterprise Grid Search")

        start_time = time.time()

        # ── Guard: detect unsupported model type ──────────────────────────────
        try:
            model_step = pipeline.named_steps.get("model")
            model_class = type(model_step).__name__.lower()
            if "isolationforest" in model_class:
                raise ValueError(
                    f"GridSearchCV is not supported for unsupervised models "
                    f"({type(model_step).__name__}). Use 'manual' tuning instead."
                )
        except AttributeError:
            pass  # pipeline doesn't expose named_steps — proceed

        try:
            param_grid = self._prepare_param_grid(hyperparameters)
        except Exception as e:
            logger.error(
                f"[GridSearch] Failed to build param grid: {e}",
                exc_info=True
            )
            raise ValueError(
                f"Invalid hyperparameter configuration for GridSearch: {e}"
            ) from e

        if not param_grid:
            logger.warning(
                "[GridSearch] Param grid is EMPTY after filtering. "
                "Falling back to single fit with pipeline defaults. "
                f"Raw hyperparameters: {hyperparameters}"
            )
            pipeline.fit(X_train, y_train)
            return pipeline, {}

        logger.info(f"Search Grid: {param_grid}")
        logger.info(f"Total combinations: {self._count_combinations(param_grid)}")

        cv = tuning_config.get("cv", 3)
        # f1_weighted works for both binary and multiclass labels;
        # plain 'f1' only works for strict {0,1} binary problems.
        scoring = tuning_config.get("metric", "f1_weighted")

        # --- Stratified subsampling for large datasets ---
        X_cv, y_cv = self._get_cv_data(X_train, y_train)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=tuning_config.get("n_jobs", 1),
            verbose=2,        # Shows CV iteration progress in worker logs
            refit=False,      # We will refit ourselves on the FULL dataset
            pre_dispatch="2*n_jobs",
            error_score="raise",
        )

        logger.info(f"Fitting GridSearch on {len(X_cv):,} rows (cv={cv}, scoring={scoring})")
        search.fit(X_cv, y_cv)

        best_params = search.best_params_
        logger.info(f"Grid Search Complete. Best CV Score: {search.best_score_:.4f}")
        logger.info(f"Best Params: {best_params}")

        # --- Refit on FULL training data with best params ---
        logger.info(f"Refitting on full dataset ({len(X_train):,} rows)...")
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        logger.info("Full-data refit complete.")

        return pipeline, best_params

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_cv_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return stratified subsample for CV if dataset is large."""
        n = len(X)
        if n <= _LARGE_DATASET_THRESHOLD:
            return X, y

        logger.info(
            f"Large dataset detected ({n:,} rows). "
            f"Subsampling to {_CV_SUBSAMPLE_SIZE:,} rows for CV "
            f"(final model trains on full data)."
        )
        # Stratified resample: preserve class ratio
        indices = resample(
            np.arange(n),
            n_samples=_CV_SUBSAMPLE_SIZE,
            stratify=y.values,
            random_state=42
        )
        return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)

    def _prepare_param_grid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build param grid from user config.
        - Range dicts  → smart sampled list (max 5 values).
        - Scalar values → [value] (single-point grid).
        - Lists         → capped to max 5 values.
        - Falls back to optimised XGBoost grid when params are likely UI-generated
          ranges that would explode (e.g. n_estimators step=1 from 10 to 500).
        """
        grid = {}
        for k, v in params.items():
            # Skip non-model params passed through hyperparameters dict
            if k in ("imbalanced_strategy", "test_size", "validation_size", "shuffle"):
                continue

            key = k if "__" in k else f"model__{k}"

            if isinstance(v, dict) and ("min" in v or "max" in v):
                sampled = self._create_grid_list(v)
                # Safety: if sampled list is huge (UI step=1 bug), override
                if len(sampled) > _MAX_VALUES_PER_PARAM:
                    logger.warning(
                        f"Param '{key}' generated {len(sampled)} values. "
                        f"Capping to {_MAX_VALUES_PER_PARAM} via linspace."
                    )
                    sampled = np.linspace(
                        v.get("min", sampled[0]),
                        v.get("max", sampled[-1]),
                        _MAX_VALUES_PER_PARAM
                    )
                    # Preserve int type
                    if not (isinstance(v.get("min"), float) or isinstance(v.get("max"), float)):
                        sampled = np.unique(sampled.astype(int)).tolist()
                    else:
                        sampled = [float(round(float(x), 4)) for x in sampled.tolist()]
                grid[key] = sampled

            elif isinstance(v, list):
                # Enforce cap
                grid[key] = [_parse(x) for x in list(v)[:_MAX_VALUES_PER_PARAM]]

            elif isinstance(v, (int, float, str, bool)):
                grid[key] = [_parse(v)]
                logger.debug(f"[GridSearch] {k} → {key}: fixed [{v}]")
            else:
                grid[key] = v

        # If the resulting grid has exactly one combination, use it as-is.
        # Do NOT fall back to XGBoost grid — the pipeline may contain a different
        # model (e.g. LogisticRegression, SVM) that would fail with XGBoost params.
        total = self._count_combinations(grid)
        if total == 1:
            logger.info(
                "Single-combination grid detected. Using user-provided parameters. "
                "To search multiple values, add ranges/lists for hyperparameters."
            )

        return grid

    def _create_grid_list(self, config: Dict[str, Any]) -> list:
        """Create value list from a range-dict config. Never returns empty."""
        min_val = config.get("min", 0)
        max_val = config.get("max")
        step = config.get("step")

        if max_val is None or min_val == max_val:
            return [min_val]

        is_float = isinstance(min_val, float) or isinstance(max_val, float) or (step is not None and isinstance(step, float))

        if step is None or step <= 0:
            step = 1.0 if is_float else 1

        # Calculate number of points instead of using np.arange (avoids float precision bugs)
        n_points = max(1, int(round((max_val - min_val) / step)) + 1)

        # Use linspace for floats (precision-safe), arange for ints
        if is_float:
            vals = np.linspace(min_val, max_val, n_points).tolist()
            vals = [float(round(float(x), 6)) for x in vals]
        else:
            vals = list(range(int(min_val), int(max_val) + 1, int(step)))

        # Safety: never return empty
        if not vals:
            logger.warning(f"_create_grid_list produced empty list for config={config}, falling back to [min_val]")
            return [min_val]

        return vals

    @staticmethod
    def _count_combinations(grid: Dict[str, Any]) -> int:
        total: int = 1
        for v in grid.values():
            if isinstance(v, list):
                total = total * int(len(v))
        return total
