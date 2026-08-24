from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
import logging
import time
from .base import TunerStrategy

logger = logging.getLogger(__name__)


class BayesianTuner(TunerStrategy):
    """
    Executes Bayesian Optimization using scikit-optimize (BayesSearchCV).

    Uses a Gaussian Process surrogate model to intelligently sample the
    hyperparameter space, focusing evaluations on promising regions.
    More efficient than Random/Grid search for expensive-to-evaluate models.
    """

    # Keys that TrainingConfig injects into hyperparameters but are NOT
    # model hyper-parameters — they must be stripped before building the
    # search space.
    # NOTE: 'early_stopping_rounds' is excluded because LightGBM requires an
    # eval_set passed to fit() for early stopping, which cannot be forwarded
    # through a Scikit-Learn Pipeline/BayesSearchCV.
    _NON_MODEL_KEYS = {"imbalanced_strategy", "test_size", "validation_size", "shuffle", "_tuning_method", "multi_class", "early_stopping_rounds"}



    def tune(
        self,
        pipeline: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Dict[str, Any],
        tuning_config: Dict[str, Any]
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:

        start_time = time.time()

        # ── Dependency check ──────────────────────────────────────────────────
        try:
            from skopt import BayesSearchCV
        except ImportError:
            raise ImportError(
                "[BayesianTuner] scikit-optimize is not installed. "
                "Add it: pip install scikit-optimize"
            )

        # ── Guard: detect unsupported model type ──────────────────────────────
        try:
            model_step = pipeline.named_steps.get("model")
            if "isolationforest" in type(model_step).__name__.lower():
                raise ValueError(
                    f"BayesSearchCV is not supported for unsupervised models "
                    f"({type(model_step).__name__}). Use 'manual' tuning instead."
                )
        except AttributeError:
            pass

        # ── Build search space ────────────────────────────────────────────────
        try:
            search_space = self._prepare_search_space(hyperparameters)
        except Exception as e:
            logger.error(
                f"[BayesianTuner] Failed to build search space: {e}",
                exc_info=True
            )
            raise ValueError(
                f"Invalid hyperparameter configuration for BayesianTuner: {e}"
            ) from e

        if not search_space:
            logger.warning(
                "[BayesianTuner] Search space is EMPTY after filtering. "
                "Falling back to single fit with pipeline defaults. "
                f"Raw hyperparameters: {hyperparameters}"
            )
            pipeline.fit(X_train, y_train)
            return pipeline, {}

        # ── Search configuration ──────────────────────────────────────────────
        n_iter  = tuning_config.get("n_iter", 20)
        cv      = tuning_config.get("cv_folds", tuning_config.get("cv", 3))
        # f1_weighted works for both binary and multiclass labels;
        # plain 'f1' only works for strict {0,1} binary problems.
        scoring = tuning_config.get("metric", "f1_weighted")
        # NOTE: n_jobs=1 is used here for cross-platform compatibility — on
        # Windows, BayesSearchCV with n_jobs=-1 spawns child processes that
        # can be killed by the OS in script contexts without the
        # if __name__ == '__main__' guard. Workers are already running inside
        # Celery, so parallelism at the CV level is not needed.
        n_jobs  = tuning_config.get("n_jobs", 1)

        logger.info(
            f"[BayesianTuner] Starting | n_iter={n_iter} | cv={cv} | "
            f"scoring={scoring} | n_jobs={n_jobs}"
        )
        logger.info(f"[BayesianTuner] Train shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"[BayesianTuner] Label distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"[BayesianTuner] Search space ({len(search_space)} params):")
        for param, dim in search_space.items():
            logger.info(f"  {param}: {dim}")

        # ── Fit ───────────────────────────────────────────────────────────────
        try:
            search = BayesSearchCV(
                estimator=pipeline,
                search_spaces=search_space,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                random_state=42,
                error_score="raise"
            )

            logger.info("[BayesianTuner] Fitting … (Bayesian optimization in progress)")
            search.fit(X_train, y_train)

        except ValueError as e:
            logger.error(
                f"[BayesianTuner] FAILED — likely a scorer/label mismatch or bad param. "
                f"scoring='{scoring}', cv={cv}. Error: {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"[BayesianTuner] UNEXPECTED ERROR during fit: {e}",
                exc_info=True
            )
            raise

        # ── Results ───────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        logger.info(f"[BayesianTuner] ✅ Complete in {elapsed:.1f}s")
        logger.info(f"[BayesianTuner] Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"[BayesianTuner] Best params:")
        for k, v in search.best_params_.items():
            logger.info(f"  {k}: {v}")

        # Log top-5 candidates from CV results
        try:
            import pandas as _pd
            cv_df = _pd.DataFrame(search.cv_results_)
            top5 = (
                cv_df[["params", "mean_test_score", "std_test_score"]]
                .sort_values("mean_test_score", ascending=False)
                .head(5)
            )
            logger.info(f"[BayesianTuner] Top-5 candidates:\n{top5.to_string(index=False)}")
        except Exception:
            pass

        return search.best_estimator_, search.best_params_

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_search_space(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prefix parameters with 'model__' and convert every value into a
        valid skopt dimension object required by BayesSearchCV:

          - dict  {'min': X, 'max': Y}  →  Integer(X, Y) or Real(X, Y)
          - list  [a, b, c]             →  Categorical([a, b, c])
          - scalar int/float/str/bool   →  Categorical([v])   (fixed value)
          - already a skopt Dimension   →  returned as-is
        """
        from skopt.space import Integer, Real, Categorical

        space = {}
        skipped = []

        def _parse(val):
            if isinstance(val, str):
                v_low = val.strip().lower()
                if v_low == "true": return True
                if v_low == "false": return False
                if v_low == "none": return None
            return val

        for k, v in params.items():
            # Skip config-level keys that are not model hyper-parameters
            if k in self._NON_MODEL_KEYS:
                skipped.append(k)
                continue

            key = k if "__" in k else f"model__{k}"

            # Parse booleans and None from strings
            if isinstance(v, list):
                v = [_parse(x) for x in v]
            elif isinstance(v, (int, float, str, bool)):
                v = _parse(v)

            dimension = self._build_dimension(v, Integer, Real, Categorical)
            if dimension is not None:
                space[key] = dimension
                logger.debug(f"[BayesianTuner] {k} → {key}: {dimension}")
            else:
                logger.warning(
                    f"[BayesianTuner] Could not build dimension for '{k}' "
                    f"(type={type(v).__name__}, value={v}) — skipped."
                )

        if skipped:
            logger.info(f"[BayesianTuner] Skipped non-model keys: {skipped}")

        return space

    def _build_dimension(self, v, Integer, Real, Categorical):
        """Convert a raw hyperparameter value to a valid skopt dimension."""
        # Already a skopt Dimension — pass through
        try:
            from skopt.space.space import Dimension
            if isinstance(v, Dimension):
                return v
        except ImportError:
            pass

        # Frontend range dict: {'min': X, 'max': Y} or with optional 'step'
        if isinstance(v, dict):
            min_val = v.get("min")
            max_val = v.get("max")

            if min_val is None and max_val is None:
                logger.warning(f"[BayesianTuner] Skipping unsupported dict dimension (no min/max): {v}")
                return None

            # Sensible defaults when only one bound is given
            if min_val is None:
                min_val = 0
            if max_val is None:
                max_val = min_val

            # Normalize numeric strings (e.g. "-0.01") before comparisons
            if isinstance(min_val, str):
                try:
                    min_val = float(min_val) if "." in min_val else int(min_val)
                except ValueError:
                    pass
            if isinstance(max_val, str):
                try:
                    max_val = float(max_val) if "." in max_val else int(max_val)
                except ValueError:
                    pass

            is_float = isinstance(min_val, float) or isinstance(max_val, float)

            if min_val == max_val:
                return Categorical([min_val])
            if min_val > max_val:
                raise ValueError(
                    f"invalid range bounds: min ({min_val}) must be <= max ({max_val})"
                )

            if is_float:
                return Real(float(min_val), float(max_val))
            else:
                return Integer(int(min_val), int(max_val))

        # List → Categorical
        if isinstance(v, list):
            if not v:
                logger.warning("[BayesianTuner] Skipping empty list dimension")
                return None
            return Categorical(v)

        # Scalar → fixed Categorical with a single value
        if isinstance(v, (int, float, str, bool)):
            return Categorical([v])

        logger.warning(f"[BayesianTuner] Skipping unrecognized dimension type {type(v)}: {v}")
        return None
