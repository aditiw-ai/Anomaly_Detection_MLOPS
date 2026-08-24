from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import logging
import time
from .base import TunerStrategy

logger = logging.getLogger(__name__)


class RandomSearchTuner(TunerStrategy):
    """
    Executes a randomized search over specified parameter distributions.

    Randomly samples ``n_iter`` combinations from the specified distributions
    and selects the best-performing set via cross-validation.
    Much more efficient than GridSearch when there are many parameters or
    when parameters have continuous ranges.
    """

    # Keys that TrainingConfig/worker inject into hyperparameters but are NOT
    # valid pipeline search parameters.
    # NOTE: 'early_stopping_rounds' is excluded because LightGBM requires an
    # eval_set passed to fit() for early stopping, which cannot be forwarded
    # through a Scikit-Learn Pipeline/RandomizedSearch.
    _NON_MODEL_KEYS = {"imbalanced_strategy", "test_size", "validation_size", "shuffle", "_tuning_method", "multi_class", "early_stopping_rounds"}



    # Models that are unsupervised — CV with a classification scorer won't work
    _UNSUPPORTED_ALGORITHMS = set()

    def tune(
        self,
        pipeline: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Dict[str, Any],
        tuning_config: Dict[str, Any]
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:

        start_time = time.time()

        # ── Guard: detect unsupported model type ──────────────────────────────
        # Peek into the pipeline's final step to detect unsupervised models.
        try:
            model_step = pipeline.named_steps.get("model")
            model_class = type(model_step).__name__.lower()
            if any(tag in model_class for tag in ("isolationforest",)):
                raise ValueError(
                    f"RandomizedSearchCV is not supported for unsupervised models "
                    f"({type(model_step).__name__}). Use 'manual' tuning instead."
                )
        except AttributeError:
            pass  # pipeline doesn't expose named_steps — proceed and let sklearn report

        # ── Parameter distributions ───────────────────────────────────────────
        try:
            param_dist = self._prepare_param_dist(hyperparameters)
        except Exception as e:
            logger.error(
                f"[RandomSearch] Failed to build parameter distributions: {e}",
                exc_info=True
            )
            raise ValueError(
                f"Invalid hyperparameter configuration for RandomSearch: {e}"
            ) from e

        if not param_dist:
            logger.warning(
                "[RandomSearch] Parameter distribution is EMPTY after filtering. "
                "Falling back to single fit with pipeline defaults. "
                f"Raw hyperparameters: {hyperparameters}"
            )
            pipeline.fit(X_train, y_train)
            return pipeline, {}

        # ── Search configuration ──────────────────────────────────────────────
        n_iter   = tuning_config.get("n_iter", 20)
        cv       = tuning_config.get("cv_folds", tuning_config.get("cv", 3))
        # f1_weighted works for both binary and multiclass labels;
        # plain 'f1' only works for strict {0,1} binary problems.
        scoring  = tuning_config.get("metric", "f1_weighted")
        # n_jobs=1: RandomizedSearchCV runs inside a Celery worker — spawning
        # loky subprocesses from inside Celery can cause deadlocks / OOM.
        n_jobs   = tuning_config.get("n_jobs", 1)

        logger.info(
            f"[RandomSearch] Starting | n_iter={n_iter} | cv={cv} | "
            f"scoring={scoring} | n_jobs={n_jobs}"
        )
        logger.info(f"[RandomSearch] Train shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"[RandomSearch] Label distribution:\n{y_train.value_counts().to_dict()}")
        logger.info(f"[RandomSearch] Search space ({len(param_dist)} params):")
        for param, dist in param_dist.items():
            logger.info(f"  {param}: {dist}")

        # ── Fit ───────────────────────────────────────────────────────────────
        try:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                random_state=42,
                error_score="raise"   # surface per-fold errors immediately
            )

            logger.info("[RandomSearch] Fitting … (this may take a while)")
            search.fit(X_train, y_train)

        except ValueError as e:
            logger.error(
                f"[RandomSearch] FAILED — likely a scorer/label mismatch or bad param. "
                f"scoring='{scoring}', cv={cv}. Error: {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"[RandomSearch] UNEXPECTED ERROR during fit: {e}",
                exc_info=True
            )
            raise

        # ── Results ───────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        logger.info(f"[RandomSearch] ✅ Complete in {elapsed:.1f}s")
        logger.info(f"[RandomSearch] Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"[RandomSearch] Best params:")
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
            logger.info(f"[RandomSearch] Top-5 candidates:\n{top5.to_string(index=False)}")
        except Exception:
            pass  # non-critical, don't let logging failure break training

        return search.best_estimator_, search.best_params_

    def _prepare_param_dist(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefix parameters with 'model__', strip non-model keys, and ensure valid distributions."""
        dist = {}
        skipped = []

        def _parse(val):
            if isinstance(val, str):
                v_low = val.strip().lower()
                if v_low == "true": return True
                if v_low == "false": return False
                if v_low == "none": return None
            return val

        for k, v in params.items():
            # Skip config-level keys that are not pipeline parameters
            if k in self._NON_MODEL_KEYS:
                skipped.append(k)
                continue

            key = k if "__" in k else f"model__{k}"

            # Frontend range dict {'min': X, 'max': Y}
            if isinstance(v, dict) and ("min" in v or "max" in v):
                distribution = self._create_distribution(v)
                dist[key] = distribution
                logger.debug(f"[RandomSearch] {k} → {key}: {distribution}")
            # Scalar value → wrap in list (fixed value for RandomizedSearch)
            elif isinstance(v, (int, float, str, bool)):
                dist[key] = [_parse(v)]
                logger.debug(f"[RandomSearch] {k} → {key}: fixed [{v}]")
            # Already a list or scipy distribution
            elif isinstance(v, list):
                dist[key] = [_parse(x) for x in v]
                logger.debug(f"[RandomSearch] {k} → {key}: list {v}")
            else:
                # e.g. already a scipy distribution passed directly
                dist[key] = v
                logger.debug(f"[RandomSearch] {k} → {key}: passthrough {type(v).__name__}")

        if skipped:
            logger.info(f"[RandomSearch] Skipped non-model keys: {skipped}")

        return dist

    def _create_distribution(self, config: Dict[str, Any]):
        """Create scipy distribution from a frontend range dict."""
        min_val = config.get("min")
        max_val = config.get("max")
        step    = config.get("step")   # informational only for RandomSearch

        # Normalize numeric strings from frontend payloads
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

        if min_val is None:
            min_val = 0
        if max_val is None:
            return [min_val]
        if min_val > max_val:
            raise ValueError(
                f"invalid range bounds: min ({min_val}) must be <= max ({max_val})"
            )

        is_float = isinstance(min_val, float) or isinstance(max_val, float)

        if min_val == max_val:
            return [min_val]

        if step is not None and not is_float:
            # Honour explicit step for integers: enumerate the allowed values
            import numpy as np
            vals = np.arange(int(min_val), int(max_val) + 1, int(step)).tolist()
            logger.debug(
                f"[RandomSearch] Integer range [{min_val}, {max_val}] step={step} "
                f"→ {len(vals)}-element list: {vals[:5]}{'…' if len(vals) > 5 else ''}"
            )
            return vals

        if is_float:
            # uniform(loc=min, scale=max-min) → samples in [min, max]
            dist = uniform(loc=float(min_val), scale=float(max_val) - float(min_val))
            logger.debug(
                f"[RandomSearch] Float range [{min_val}, {max_val}] → uniform distribution"
            )
            return dist
        else:
            # randint(low, high) → samples in [low, high)  →  +1 to include max
            dist = randint(int(min_val), int(max_val) + 1)
            logger.debug(
                f"[RandomSearch] Int range [{min_val}, {max_val}] → randint distribution"
            )
            return dist
