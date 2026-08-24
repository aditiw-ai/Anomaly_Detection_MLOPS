"""
ML Algorithm Implementations
Fraud detection model training and inference.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ml.transformers.fraud_feature_engineer import FraudFeatureEngineer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class SafeRandomForestClassifier(RandomForestClassifier):
    """
    Wrapper around RandomForestClassifier that gracefully handles -1 values from 
    search-based tuners (which use -1 in their numeric bounds to represent 'None').
    """
    def set_params(self, **params):
        for k in ("max_depth", "max_leaf_nodes", "max_samples"):
            if k in params and params[k] in (-1, "-1", "-1.0", -1.0):
                params[k] = None
        return super().set_params(**params)
        
    def fit(self, X, y, sample_weight=None):
        for k in ("max_depth", "max_leaf_nodes", "max_samples"):
            if getattr(self, k, None) in (-1, "-1", "-1.0", -1.0):
                setattr(self, k, None)
        return super().fit(X, y, sample_weight=sample_weight)

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    algorithm: str = "xgboost"
    hyperparameters: Dict[str, Any] = None
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    imbalanced_strategy: str = "class_weight"  # class_weight, smote, undersample
    feature_config: Optional[Dict[str, Any]] = None  # Forwarded to FraudFeatureEngineer
    tuning_method: str = "manual"
    tuning_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.tuning_config is None:
            self.tuning_config = {}


@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    pipeline: Any
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    feature_names: list
    confusion_matrix: np.ndarray
    classification_report: str
    onnx_bytes: Optional[bytes] = None
    optimal_threshold: float = 0.5
    calibrator: Optional[Any] = None  # Platt scaler: LogisticRegression(y_prob -> y_true)


class FraudDetectionTrainer:
    """
    Trains fraud detection models using various algorithms.
    
    Supported algorithms:
    - xgboost: XGBoost Classifier (default)
    - lightgbm: LightGBM Classifier
    - random_forest: Random Forest
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model = None
        self.pipeline = None
        self.feature_names = []
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train model with automatic splitting.
        
        Use this for standalone training, notebooks, or when data is not pre-split.
        For production training jobs, use train_with_splits() instead.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            TrainingResult with model, metrics, and feature importance
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Delegate to train_with_splits
        return self.train_with_splits(X_train, y_train, X_test, y_test)
    
    def train_with_splits(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """
        Train model using pre-split data.
        
        Use this when data has already been split by DataProcessingService.
        This avoids redundant splitting and ensures consistency with stored splits.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            TrainingResult with model, metrics, and feature importance
        """
        logger.info(f"Training {self.config.algorithm} with pre-split data")
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Keep the supplied test split untouched. Carve validation out of the
        # training split before any imbalance handling or model fitting.
        validation_size = self.config.validation_size
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")

        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X_train,
            y_train,
            test_size=validation_size,
            random_state=self.config.random_state,
            stratify=y_train,
        )

        # Handle imbalance on fit data only. Validation and test retain their
        # original distributions for calibration and final metrics.
        X_fit, y_fit = self._handle_imbalanced(X_fit, y_fit)
        
        # BUILD SKLEARN PIPELINE
        self.pipeline = self._build_pipeline(y_fit)
        
        # HYPERPARAMETER TUNING
        from ml.tuning.factory import TunerFactory
        logger.info(f"Using tuning method: {self.config.tuning_method}")
        
        tuner = TunerFactory.get_tuner(self.config.tuning_method)
        
        self.pipeline, best_params = tuner.tune(
            pipeline=self.pipeline,
            X_train=X_fit,
            y_train=y_fit,
            hyperparameters=self.config.hyperparameters,
            tuning_config=self.config.tuning_config
        )
        
        # Extract the trained model from pipeline
        self.model = self.pipeline.named_steps['model']
        self.feature_names = self.pipeline.named_steps['fraud_features'].feature_names_out_

        # Detect binary vs multiclass from the training labels so prediction
        # generation below can branch accordingly.
        n_classes = y_fit.nunique()

        if n_classes <= 2:
            # --- Validation scores for calibration and threshold selection ---
            y_prob_validation_raw = self.predict_proba(X_validation)

            # --- Platt Scaling: calibrate biased XGBoost probabilities ---
            # scale_pos_weight inflates raw probabilities for ALL samples.
            # Platt scaling fits a sigmoid (LogisticRegression) on the raw scores
            # vs true labels so that output probabilities are true probabilities.
            # After calibration, threshold = 0.5 is operationally correct.
            calibrator = None
            y_prob_validation = y_prob_validation_raw  # fallback if calibration fails
            try:
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                calibrator.fit(y_prob_validation_raw.reshape(-1, 1), y_validation)
                y_prob_validation = calibrator.predict_proba(
                    y_prob_validation_raw.reshape(-1, 1)
                )[:, 1]
                logger.info(
                    f"Platt calibration applied on validation data. Raw prob mean: "
                    f"{y_prob_validation_raw.mean():.4f} → "
                    f"Calibrated mean: {y_prob_validation.mean():.4f}"
                )
            except Exception as cal_err:
                logger.warning(f"Platt calibration failed, using raw probabilities: {cal_err}")

            # --- Optimal threshold on calibrated probabilities (Max F1-Score) ---
            # Youden's J treats TPR and FPR equally, which on ~5% prevalence data
            # selects a very low threshold and produces a huge False Positive Rate.
            # We maximize the F1-Score instead to balance Precision and Recall for the minority class.
            optimal_threshold = 0.5
            try:
                from sklearn.metrics import precision_recall_curve
                precisions, recalls, thresholds = precision_recall_curve(
                    y_validation, y_prob_validation
                )
                
                # F1 = 2 * (P * R) / (P + R)
                f1_scores = np.divide(
                    2 * precisions * recalls,
                    precisions + recalls,
                    out=np.zeros_like(precisions),
                    where=(precisions + recalls) > 0
                )
                
                best_idx = int(np.argmax(f1_scores))
                # precision_recall_curve returns thresholds len 1 less than precisions/recalls
                if best_idx < len(thresholds):
                    optimal_threshold = float(np.clip(thresholds[best_idx], 0.01, 0.99))
                logger.info(f"Optimal threshold selected on validation data: {optimal_threshold:.4f}")
            except Exception as thresh_err:
                logger.warning(f"Could not compute F1-optimal threshold, using 0.5: {thresh_err}")

            # Apply locked calibration and threshold once to the untouched test set.
            y_prob_test_raw = self.predict_proba(X_test)
            y_prob = (
                calibrator.predict_proba(y_prob_test_raw.reshape(-1, 1))[:, 1]
                if calibrator is not None
                else y_prob_test_raw
            )
            y_pred = (y_prob >= optimal_threshold).astype(int)
        else:
            # Multiclass classification: Platt scaling (a sigmoid fit on a
            # single score) and a single decision threshold are binary-only
            # concepts and do not apply here. Use the full per-class
            # probability matrix from the pipeline and predict the class
            # with the highest predicted probability.
            calibrator = None
            optimal_threshold = 0.5
            y_prob = self.pipeline.predict_proba(X_test)
            class_labels = getattr(self.model, "classes_", np.arange(y_prob.shape[1]))
            y_pred = np.asarray(class_labels)[np.argmax(y_prob, axis=1)]
            logger.info(f"Multiclass predictions generated via argmax over {y_prob.shape[1]} classes")

        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        metrics["optimal_threshold"] = optimal_threshold
        
        # Get feature importance
        importance = self._get_feature_importance()
        
        # Build result
        result = TrainingResult(
            model=self.model,
            pipeline=self.pipeline,
            metrics=metrics,
            feature_importance=importance,
            feature_names=self.feature_names,
            confusion_matrix=confusion_matrix(y_test, y_pred),
            classification_report=classification_report(y_test, y_pred),
            optimal_threshold=optimal_threshold,
            calibrator=calibrator,
        )
        
        auc_log_str = f"{metrics['auc']:.4f}" if metrics.get("auc") is not None else "N/A"
        logger.info(
            f"Training complete. F1: {metrics['f1']:.4f}, AUC: {auc_log_str}, "
            f"Threshold: {optimal_threshold:.4f} (calibrated: {calibrator is not None})"
        )
        return result
    
    # Keys that TrainingConfig/worker inject into hyperparameters but are NOT
    # valid model constructor arguments.  All tuner strategies also strip these.
    _NON_MODEL_KEYS = {"imbalanced_strategy", "test_size", "validation_size", "shuffle"}

    def _create_model(self, y_train: pd.Series, use_defaults_only: bool = False):
        """
        Create the model based on algorithm choice.

        Args:
            y_train: Training labels (used for class-weight computation).
            use_defaults_only: When True, user hyperparameters are ignored and
                only algorithm defaults are applied.  Set this to True when a
                search-based tuner (grid / random / bayesian) will override
                parameters itself so that sklearn doesn't receive raw range
                dicts like {'min': 10, 'max': 50} as constructor arguments.
        """
        algorithm = self.config.algorithm.lower()
        params = self.config.hyperparameters.copy()

        # Always strip config-level keys that are not model hyperparameters
        for key in self._NON_MODEL_KEYS:
            params.pop(key, None)

        # Strip range-dict values (frontend format) — these are never valid
        # sklearn constructor arguments regardless of use_defaults_only.
        params = {
            k: v for k, v in params.items()
            if not isinstance(v, dict)
        }

        # When a search tuner will set the params, start from defaults only
        if use_defaults_only:
            params = {}
        
        # Helper to get class weights
        def get_class_weight_params(model_type):
            if self.config.imbalanced_strategy != "class_weight":
                return {}

            if model_type == "xgboost":
                unique_classes = pd.Series(y_train).dropna().unique()
                if len(unique_classes) <= 2:
                    neg_count = (y_train == 0).sum()
                    pos_count = (y_train == 1).sum()
                    return {"scale_pos_weight": neg_count / max(pos_count, 1)}
                return {}
            elif model_type in ["sklearn_balanced", "lightgbm"]:
                return {"class_weight": "balanced"}
            return {}

        # ---------------------------------------------------------------------
        # Model Factory Functions
        # ---------------------------------------------------------------------
        def create_xgboost():
            from xgboost import XGBClassifier
            n_classes = int(pd.Series(y_train).nunique()) if len(y_train) else 1
            multiclass = n_classes > 2

            defaults = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.3,       # XGBoost official default (eta=0.3)
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "colsample_bylevel": 1.0,
                "colsample_bynode": 1.0,
                "gamma": 0.0,
                "min_child_weight": 1.0,
                "max_delta_step": 0,
                "reg_alpha": 0.0,           # stored as 'alpha' by user, mapped below
                "reg_lambda": 1.0,          # stored as 'lambda' by user, mapped below
                "base_score": 0.5,
                "scale_pos_weight": 1.0,
                "max_bin": 256,
                "grow_policy": "depthwise",
                "booster": "gbtree",
                "tree_method": "hist",
                "objective": "multi:softprob" if multiclass else "binary:logistic",
                "eval_metric": "mlogloss" if multiclass else "logloss",
                "num_class": n_classes if multiclass else None,
                "seed": 0,
                "verbosity": 1,
            }
            defaults.update(get_class_weight_params("xgboost"))

            # Remap user-facing param names to XGBClassifier kwargs:
            # 'lambda' (reserved keyword) → 'reg_lambda'
            # 'alpha'                     → 'reg_alpha'
            if "lambda" in params:
                params["reg_lambda"] = params.pop("lambda")
            if "alpha" in params:
                params["reg_alpha"] = params.pop("alpha")

            # -------------------------------------------------------------------
            # Strip early_stopping_rounds — ALWAYS.
            # XGBoost requires an eval_set passed to fit() when this is set, but
            # sklearn Pipeline.fit() has no way to forward eval_set to the final
            # estimator.  Any non-zero value causes:
            #   ValueError: Must have at least 1 validation dataset for early stopping.
            # Users who need early stopping should use manual training outside a
            # Pipeline, or a custom sklearn wrapper that supports fit_params.
            # -------------------------------------------------------------------
            if params.pop("early_stopping_rounds", None) is not None:
                logger.warning(
                    "[XGBoost] 'early_stopping_rounds' was removed from hyperparameters. "
                    "Early stopping requires an eval_set which cannot be passed through a "
                    "sklearn Pipeline. The model will train for the full n_estimators rounds."
                )

            # -------------------------------------------------------------------
            # Sanitize objective: this is an XGBClassifier — regression / ranking
            # objectives are invalid here and will silently mispredict.
            # Force back to binary:logistic for binary or multi:softprob for multiclass.
            # -------------------------------------------------------------------
            user_obj = params.get("objective", defaults["objective"])
            _INVALID_OBJECTIVES = (
                "reg:", "rank:", "count:", "survival:"
            )
            if any(user_obj.startswith(pfx) for pfx in _INVALID_OBJECTIVES):
                logger.warning(
                    f"[XGBoost] objective='{user_obj}' is not valid for this classification task. "
                    f"Overriding with '{defaults['objective']}'."
                )
                params["objective"] = defaults["objective"]
                params.pop("tweedie_variance_power", None)

            if multiclass:
                if "binary:" in str(user_obj):
                    logger.warning(
                        f"[XGBoost] objective='{user_obj}' is binary-only. Overriding with 'multi:softprob'."
                    )
                    params["objective"] = "multi:softprob"
                if "multi:softmax" in str(user_obj):
                    params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
                params["eval_metric"] = "mlogloss"
            else:
                if any(str(user_obj).startswith(pfx) for pfx in ("multi:", "multi:softprob")):
                    logger.warning(
                        f"[XGBoost] objective='{user_obj}' is multiclass-only. Overriding with 'binary:logistic'."
                    )
                    params["objective"] = "binary:logistic"
                    params.pop("num_class", None)
                params.pop("num_class", None)

            # -------------------------------------------------------------------
            # Strip DART-only params when booster != dart.
            # These cause a XGBoost UserWarning and waste log space.
            # -------------------------------------------------------------------
            booster = params.get("booster", defaults["booster"])
            _DART_ONLY = {"rate_drop", "skip_drop", "normalize_type", "sample_type", "one_drop"}
            if booster != "dart":
                stripped_dart = {k for k in _DART_ONLY if k in params}
                if stripped_dart:
                    logger.warning(
                        f"[XGBoost] Stripping DART-only params {stripped_dart} "
                        f"because booster='{booster}' (not 'dart')."
                    )
                    for k in stripped_dart:
                        params.pop(k, None)

            # Update defaults with sanitized user params
            final_params = defaults.copy()
            final_params.update(params)
            if final_params.get("num_class") is None:
                final_params.pop("num_class", None)
            return XGBClassifier(**final_params)


        def create_lightgbm():
            from lightgbm import LGBMClassifier
            defaults = {
                "n_estimators":           100,
                "learning_rate":          0.1,
                "num_leaves":             64,
                "max_depth":              6,
                "min_data_in_leaf":       3,
                "max_delta_step":         0.0,
                "min_gain_to_split":      0.0,
                "max_bin":                255,
                "feature_fraction":       0.9,
                "feature_fraction_bynode":1.0,
                "bagging_fraction":       0.9,
                "bagging_freq":           1,
                "lambda_l1":              0.0,
                "lambda_l2":              0.0,
                "scale_pos_weight":       1.0,
                # NOTE: is_unbalance is NOT set in defaults — LightGBM raises an error
                # if both is_unbalance AND scale_pos_weight are present simultaneously.
                # We handle the mutual exclusion below after merging user params.
                "boosting_type":          "gbdt",   # LGBMClassifier uses boosting_type, not boosting
                "tree_learner":           "serial",
                "tweedie_variance_power": 1.5,
                "num_threads":            0,
                "verbosity":              -1,        # suppress LightGBM stdout by default
                "random_state":           self.config.random_state,
            }
            defaults.update(get_class_weight_params("lightgbm"))

            # ── Name corrections: UI uses 'boosting', LGBMClassifier uses 'boosting_type'
            if "boosting" in params:
                params["boosting_type"] = params.pop("boosting")

            # ── is_unbalance: UI sends "True"/"False" strings → convert to bool
            if "is_unbalance" in params:
                params["is_unbalance"] = str(params["is_unbalance"]).lower() == "true"

            # ── early_stopping_rounds: 0 means disabled
            early_stopping = params.pop("early_stopping_rounds", None)
            if early_stopping and early_stopping > 0:
                defaults["_early_stopping_rounds"] = early_stopping

            # ── metric: "auto" means let LightGBM decide — don't pass it explicitly
            if params.get("metric") == "auto":
                params.pop("metric")

            final_params = defaults.copy()
            final_params.update(params)
            final_params.pop("_early_stopping_rounds", None)

            # ── Coercions for LightGBM ────────────────────────────────────────
            for i_key in ("n_estimators", "num_leaves", "max_depth", "min_data_in_leaf", "max_bin", "bagging_freq", "num_threads"):
                val = final_params.get(i_key)
                if val is not None:
                    try:
                        final_params[i_key] = int(val)
                    except (TypeError, ValueError):
                        pass

            for f_key in ("learning_rate", "max_delta_step", "min_gain_to_split", "feature_fraction", "feature_fraction_bynode", "bagging_fraction", "lambda_l1", "lambda_l2", "scale_pos_weight", "tweedie_variance_power"):
                val = final_params.get(f_key)
                if val is not None:
                    try:
                        final_params[f_key] = float(val)
                    except (TypeError, ValueError):
                        pass

            # ── MUTUAL EXCLUSION: LightGBM raises an error if both
            # is_unbalance AND scale_pos_weight are present at the same time.
            is_unbalance_val = final_params.get("is_unbalance", False)
            if is_unbalance_val:
                # User chose is_unbalance=True → remove scale_pos_weight entirely
                final_params.pop("scale_pos_weight", None)
                logger.info("[LightGBM] is_unbalance=True → scale_pos_weight removed")
            else:
                # is_unbalance is False (the default) → remove it so LightGBM
                # doesn't see both keys and raise the conflict error
                final_params.pop("is_unbalance", None)

            return LGBMClassifier(**final_params)



        def create_random_forest():
            defaults = {
                "n_estimators":             100,
                "criterion":                "gini",
                "max_depth":                15,         # Cap depth for speed on 500k rows
                "min_samples_split":        10,
                "min_samples_leaf":         5,
                "min_weight_fraction_leaf": 0.0,
                "max_features":             "sqrt",
                "max_leaf_nodes":           None,       # None = unlimited
                "min_impurity_decrease":    0.0,
                "bootstrap":                True,
                "oob_score":                False,
                "n_jobs":                   4,          # capped — see n_jobs guard below
                "verbose":                  1,          # Add verbosity to see progress in logs
                "warm_start":               False,
                "ccp_alpha":                0.0,
                "max_samples":              0.8,        # Subsample rows for speed
                "random_state":             self.config.random_state,
            }
            defaults.update(get_class_weight_params("sklearn_balanced"))

            # ── bool coercion: UI sends "True"/"False" strings ────────────────
            for bool_key in ("bootstrap", "oob_score", "warm_start"):
                if bool_key in params:
                    params[bool_key] = str(params[bool_key]).lower() == "true"

            # ── -1 sentinel → None: UI sliders can't represent None natively ──
            # max_depth=-1 means "no limit" → pass None to sklearn
            # max_leaf_nodes=-1 same, max_samples=-1 same
            for none_key in ("max_depth", "max_leaf_nodes", "max_samples"):
                if params.get(none_key) in (-1, "-1"):
                    params[none_key] = None

            # ── class_weight: "None" string → actual None ─────────────────────
            if params.get("class_weight") == "None":
                params["class_weight"] = None

            # ── max_features: keep string ("sqrt","log2","None") or
            #    convert to None if user selected "None" ───────────────────────
            if params.get("max_features") == "None":
                params["max_features"] = None

            final_params = defaults.copy()
            final_params.update(params)

            # ── sklearn bootstrap constraints ─────────────────────────────────
            # RandomForestClassifier raises ValueError when:
            #   1. bootstrap=False AND max_samples is set (not None)
            #   2. bootstrap=False AND oob_score=True
            # Both features require sampling WITH replacement (bootstrap=True).
            if not final_params.get("bootstrap", True):
                if final_params.get("max_samples") is not None:
                    raise ValueError(
                        "max_samples cannot be set when bootstrap=False "
                        "(sklearn constraint). To use max_samples, set bootstrap=True."
                    )

                if final_params.get("oob_score", False):
                    raise ValueError(
                        "oob_score=True requires bootstrap=True "
                        "(sklearn constraint). To use out-of-bag scoring, set bootstrap=True."
                    )

            # ── n_jobs memory guard ───────────────────────────────────────────────
            # n_jobs=-1 tells sklearn to use ALL CPU cores. With RandomForest on a
            # large dataset inside Docker (4GB limit), each parallel thread needs its
            # own copy of the data being processed — 12 cores × ~300MB = OOM kill.
            # Cap to 6 (slightly higher now that max_depth is capped and max_samples is 0.8).
            MAX_RF_JOBS = 6
            n_jobs_val = final_params.get("n_jobs", MAX_RF_JOBS)
            try:
                n_jobs_int = int(n_jobs_val)
            except (TypeError, ValueError):
                n_jobs_int = MAX_RF_JOBS
            if n_jobs_int < 1 or n_jobs_int > MAX_RF_JOBS:
                logger.warning(
                    "[RandomForest] n_jobs=%s capped to %d to prevent OOM "
                    "(Docker memory limit 4GB, 500K-row dataset).",
                    n_jobs_val, MAX_RF_JOBS
                )
                final_params["n_jobs"] = MAX_RF_JOBS

            return SafeRandomForestClassifier(**final_params)



        def create_logistic_regression():
            from sklearn.linear_model import LogisticRegression
            defaults = {
                "penalty":            "l2",
                "dual":               False,
                "tol":                1e-4,
                "C":                  1.0,
                "fit_intercept":      True,
                "intercept_scaling":  1.0,
                "random_state":       self.config.random_state,
                "solver":             "lbfgs",
                "max_iter":           100,
                "verbose":            0,
                "warm_start":         False,
                # n_jobs=-1 only during manual runs; search tuners already
                # parallelise at the CV level, so model-level parallelism
                # inside a Celery worker causes deadlocks / OOM on Linux.
                "n_jobs":             1 if use_defaults_only else -1,
            }
            # Respect the imbalanced_strategy setting — if user chose SMOTE or
            # undersample, class_weight must be None to avoid double-weighting.
            defaults.update(get_class_weight_params("sklearn_balanced"))

            final_params = defaults.copy()
            final_params.update(params)

            # ── bool coercion: UI sends "True"/"False" strings ────────────────
            for bool_key in ("dual", "fit_intercept", "warm_start"):
                if bool_key in final_params:
                    final_params[bool_key] = str(final_params[bool_key]).lower() == "true"

            # ── penalty "None" string → actual None ───────────────────────────
            if str(final_params.get("penalty", "")).lower() == "none":
                final_params["penalty"] = None

            # ── class_weight "None" string → actual None ──────────────────────
            if str(final_params.get("class_weight", "")).lower() == "none":
                final_params["class_weight"] = None

            # ── float coercions ───────────────────────────────────────────────
            for f_key in ("C", "tol", "intercept_scaling"):
                val = final_params.get(f_key)
                if val is not None:
                    try:
                        final_params[f_key] = float(val)
                    except (TypeError, ValueError):
                        pass

            # ── int coercions ─────────────────────────────────────────────────
            val_mi = final_params.get("max_iter")
            if val_mi is not None:
                try:
                    final_params["max_iter"] = int(val_mi)
                except (TypeError, ValueError):
                    pass

            # ── l1_ratio: only pass to sklearn when penalty='elasticnet' ──────
            # sklearn raises ValueError if l1_ratio is provided for any other
            # penalty, so strip it unless we actually need it.
            active_penalty = final_params.get("penalty")
            if active_penalty == "elasticnet":
                # Default to 0.5 if the user left it as None
                if final_params.get("l1_ratio") is None:
                    final_params["l1_ratio"] = 0.5
                else:
                    try:
                        final_params["l1_ratio"] = float(final_params["l1_ratio"])
                    except (TypeError, ValueError):
                        final_params["l1_ratio"] = 0.5
            else:
                # Remove l1_ratio entirely for non-elasticnet penalties
                final_params.pop("l1_ratio", None)

            # ── multi_class deprecated in sklearn ≥ 1.5 — always remove ──────
            final_params.pop("multi_class", None)

            return LogisticRegression(**final_params)

        def create_decision_tree():
            from sklearn.tree import DecisionTreeClassifier
            defaults = {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "max_features": None,
                "random_state": self.config.random_state,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
            }
            defaults.update(get_class_weight_params("sklearn_balanced"))
            final_params = defaults.copy()
            final_params.update(params)

            # String "None" -> None
            for none_key in ("max_depth", "max_features", "max_leaf_nodes", "class_weight"):
                if str(final_params.get(none_key, "")).lower() == "none":
                    final_params[none_key] = None

            # int coercion for max_depth / max_leaf_nodes
            for i_key in ("max_depth", "max_leaf_nodes"):
                val = final_params.get(i_key)
                if val is not None:
                    try:
                        final_params[i_key] = int(val)
                    except (TypeError, ValueError):
                        pass

            # Coerce int vs float for min_samples
            for sz_key in ("min_samples_split", "min_samples_leaf"):
                val = final_params.get(sz_key)
                if val is not None:
                    try:
                        f_val = float(val)
                        if f_val >= 1.0:
                            final_params[sz_key] = int(f_val)
                        else:
                            final_params[sz_key] = f_val
                    except (TypeError, ValueError):
                        pass

            for float_key in ("min_weight_fraction_leaf", "min_impurity_decrease"):
                val = final_params.get(float_key)
                if val is not None:
                    try:
                        final_params[float_key] = float(val)
                    except (TypeError, ValueError):
                        pass

            return DecisionTreeClassifier(**final_params)

        def create_svm():
            from sklearn.svm import SVC
            defaults = {
                "C": 1.0, 
                "kernel": "rbf", 
                "degree": 3,
                "gamma": "scale",
                "coef0": 0.0,
                "shrinking": True,
                "probability": True,  # Required for predict_proba
                "tol": 1e-3,
                "cache_size": 200,
                "verbose": False,
                "max_iter": -1,
                "decision_function_shape": "ovr",
                "break_ties": False,
                "random_state": self.config.random_state
            }
            defaults.update(get_class_weight_params("sklearn_balanced"))
            final_params = defaults.copy()
            final_params.update(params)

            # String "None" -> None
            if str(final_params.get("class_weight", "")).lower() == "none":
                final_params["class_weight"] = None

            # Coerce booleans
            for b_key in ("shrinking", "probability", "verbose", "break_ties"):
                val = final_params.get(b_key)
                if val is not None:
                    if isinstance(val, bool):
                        final_params[b_key] = val
                    else:
                        final_params[b_key] = str(val).lower() == "true"

            # float coercions
            for f_key in ("C", "coef0", "tol", "cache_size"):
                val = final_params.get(f_key)
                if val is not None:
                    try:
                        final_params[f_key] = float(val)
                    except (TypeError, ValueError):
                        pass
                    
            # gamma coercion
            g_val = final_params.get("gamma")
            if g_val is not None and str(g_val).lower() not in ("scale", "auto"):
                try:
                    final_params["gamma"] = float(g_val)
                except (TypeError, ValueError):
                    pass
                    
            # int coercions
            for i_key in ("degree", "max_iter"):
                val = final_params.get(i_key)
                if val is not None:
                    try:
                        final_params[i_key] = int(val)
                    except (TypeError, ValueError):
                        pass
                    
            return SVC(**final_params)

        def create_knn():
            from sklearn.neighbors import KNeighborsClassifier
            defaults = {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto",
                "leaf_size": 30,
                "p": 2,
                "metric": "minkowski",
                "metric_params": None,
                "n_jobs": 1 if use_defaults_only else -1
            }
            # KNN doesn't support class_weight
            final_params = defaults.copy()
            final_params.update(params)

            if str(final_params.get("metric_params", "")).lower() == "none":
                final_params["metric_params"] = None

            for i_key in ("n_neighbors", "leaf_size", "p", "n_jobs"):
                val = final_params.get(i_key)
                if val is not None:
                    try:
                        final_params[i_key] = int(val)
                    except (TypeError, ValueError):
                        pass

            return KNeighborsClassifier(**final_params)

        def create_naive_bayes():
            from sklearn.naive_bayes import GaussianNB
            defaults = {"var_smoothing": 1e-9}
            # GaussianNB does not support class_weight (use priors if needed, or SMOTE)
            final_params = defaults.copy()
            final_params.update(params)
            
            v_val = final_params.get("var_smoothing")
            if v_val is not None:
                try:
                    final_params["var_smoothing"] = float(v_val)
                except (TypeError, ValueError):
                    pass
                    
            return GaussianNB(**final_params)

        def create_neural_network():
            from sklearn.neural_network import MLPClassifier
            defaults = {
                "hidden_layer_sizes": (100,), 
                "activation": "relu",
                "solver": "adam", 
                "alpha": 0.0001, 
                "batch_size": "auto",
                "learning_rate": "constant", 
                "learning_rate_init": 0.001,
                "power_t": 0.5,
                "max_iter": 200, 
                "shuffle": True,
                "momentum": 0.9,
                "early_stopping": False,
                "validation_fraction": 0.1,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-8,
                "random_state": self.config.random_state
            }
            final_params = defaults.copy()
            final_params.update(params)
            
            # Coerce booleans coming as strings
            for b_key in ("shuffle", "early_stopping"):
                val = final_params.get(b_key)
                if val is not None:
                    if isinstance(val, bool):
                        final_params[b_key] = val
                    else:
                        final_params[b_key] = str(val).lower() == "true"

            # Coerce floats
            for f_key in ("alpha", "learning_rate_init", "power_t", "momentum", "validation_fraction", "beta_1", "beta_2", "epsilon"):
                val = final_params.get(f_key)
                if val is not None:
                    try:
                        final_params[f_key] = float(val)
                    except (TypeError, ValueError):
                        pass
                    
            # Coerce ints
            val_mi = final_params.get("max_iter")
            if val_mi is not None:
                try:
                    final_params["max_iter"] = int(val_mi)
                except (TypeError, ValueError):
                    pass

            # batch_size can be "auto" or int
            val_bs = final_params.get("batch_size")
            if str(val_bs).lower() != "auto" and val_bs is not None:
                try:
                    final_params["batch_size"] = int(val_bs)
                except (TypeError, ValueError):
                    pass

            # hidden_layer_sizes expects a tuple. The UI sends a scalar (e.g. 100).
            hls = final_params.get("hidden_layer_sizes")
            if isinstance(hls, tuple):
                final_params["hidden_layer_sizes"] = hls
            elif isinstance(hls, (int, float)):
                final_params["hidden_layer_sizes"] = (int(hls),)
            elif isinstance(hls, str):
                try:
                    final_params["hidden_layer_sizes"] = (int(hls),)
                except ValueError:
                    # Fallback if unparseable
                    final_params["hidden_layer_sizes"] = (100,)
                    
            return MLPClassifier(**final_params)

        # ---------------------------------------------------------------------
        # Registry
        # ---------------------------------------------------------------------
        model_registry = {
            "xgboost": create_xgboost,
            "lightgbm": create_lightgbm,
            "random_forest": create_random_forest,
            "logistic_regression": create_logistic_regression,
            "decision_tree": create_decision_tree,
            "svm": create_svm,
            "knn": create_knn,
            "naive_bayes": create_naive_bayes,
            "neural_network": create_neural_network,
        }

        if algorithm not in model_registry:
            # Fallback or error
            available = ", ".join(model_registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: {available}")

        # Instantiate
        return model_registry[algorithm]()
    
    def _handle_imbalanced(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle imbalanced data if needed."""
        if self.config.imbalanced_strategy == "smote":
            if len(X) > 500_000:
                logger.warning(
                    f"Dataset too large for SMOTE ({len(X)} rows). "
                    f"Falling back to class_weight strategy."
                )
                return X, y
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.config.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                logger.info(f"Applied SMOTE: {len(X)} -> {len(X_resampled)}")
                return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
            except ImportError:
                logger.warning("imblearn not installed, skipping SMOTE")
        
        elif self.config.imbalanced_strategy == "undersample":
            from sklearn.utils import resample
            
            X_combined = X.copy()
            X_combined["_target"] = y
            
            majority = X_combined[X_combined["_target"] == 0]
            minority = X_combined[X_combined["_target"] == 1]
            
            majority_downsampled = resample(
                majority,
                n_samples=len(minority),
                random_state=self.config.random_state
            )
            
            combined = pd.concat([majority_downsampled, minority])
            logger.info(f"Applied undersampling: {len(X)} -> {len(combined)}")
            
            return combined.drop("_target", axis=1), combined["_target"]
        
        return X, y
    
    def _get_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities internally."""
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.pipeline, "decision_function"):
            return self.pipeline.decision_function(X)
        else:
            return self.pipeline.predict(X).astype(float)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained")
            
        y_pred = self.pipeline.predict(X)
            
        return y_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not trained")
        
        # Use internal helper which handles IF logic
        return self._get_probabilities(X)
    
    def _compute_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        n_classes = y_true.nunique()

        # precision_score / recall_score / f1_score default to average="binary",
        # which raises ValueError when y_true has more than 2 classes. Use
        # "weighted" for multiclass targets while preserving the existing
        # binary behavior untouched.
        metric_average = "weighted" if n_classes > 2 else "binary"

        auc_val = None
        try:
            if n_classes < 2:
                logger.warning("Only one class in y_true — AUC undefined, setting to None")
            elif n_classes == 2:
                # Binary classification: existing behavior, y_prob is the
                # positive-class probability (1D array).
                auc_val = float(roc_auc_score(y_true, y_prob))
            else:
                # Multiclass classification: sklearn requires multi_class to
                # be explicitly set to 'ovr' or 'ovo'. We use 'ovr' with
                # weighted averaging to account for class imbalance.
                auc_val = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                )
        except Exception as auc_err:
            logger.warning(f"Could not compute ROC-AUC, setting to None: {auc_err}")
            auc_val = None

        return {
            "precision": float(precision_score(y_true, y_pred, average=metric_average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=metric_average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=metric_average, zero_division=0)),
            "auc": auc_val,
            "accuracy": float((y_true == y_pred).mean()),
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model."""
        # For pipeline, we need to access the final step
        model = self.pipeline.named_steps.get('model', self.model)
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            return dict(zip(self.feature_names, importance.tolist()))
        return {}

    def _build_pipeline(self, y_train: pd.Series) -> Pipeline:
        """
        Build sklearn Pipeline with preprocessing + model.

        For tree-based models (XGBoost, LightGBM, RandomForest):
        - No scaling needed (tree models are scale-invariant)
        - FraudFeatureEngineer already handles all preprocessing
        - XGBoost/LGBM handle NaNs natively

        For other models:
        - Add SimpleImputer (median) to handle NaNs from feature engineering
        - Add StandardScaler for better convergence

        When a search-based tuner (grid / random / bayesian) is selected the
        model is built with **default parameters only** so that the tuner can
        safely call pipeline.set_params() without encountering raw range dicts
        in the already-instantiated estimator.
        """
        fraud_engineer = FraudFeatureEngineer(config=self.config.feature_config)
        # Use defaults-only for search-based tuning; manual uses user params
        use_defaults = self.config.tuning_method.lower() not in ("manual", "")
        model = self._create_model(y_train, use_defaults_only=use_defaults)
        
        # Tree models 
        if self.config.algorithm in ['xgboost', 'lightgbm', 'random_forest']:
            pipeline = Pipeline([
                ('fraud_features', fraud_engineer),
                ('model', model)
            ])
            logger.info(f"Built pipeline for {self.config.algorithm} (no scaling)")
        
        # Other models (Linear, SVM, NN, KNN)
        else:
            pipeline = Pipeline([
                ('fraud_features', fraud_engineer),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            logger.info(f"Built pipeline for {self.config.algorithm} (imputer + scaler)")
        
        return pipeline