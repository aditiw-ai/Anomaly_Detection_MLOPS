"""
Hyperparameter Conflict Validator
Checks per-algorithm hyperparameter constraints and returns plain-English
warnings about invalid combinations — without modifying any values or
blocking job creation.

Usage::

    from app.schemas.hyperparameter_validator import check_hyperparameters

    warnings = check_hyperparameters("random_forest", hyperparameters)
    # warnings: list[str] — empty means no issues found
"""
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_hyperparameters(algorithm: str, params: Dict[str, Any]) -> List[str]:
    """
    Validate ``params`` against the rules for ``algorithm``.

    Returns a list of human-readable warning strings.
    An empty list means no problems were found.
    No values are modified.  No exceptions are raised.
    """
    checker = _CHECKERS.get(algorithm.lower())
    if checker is None:
        return []
    return checker(params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_val(v: Any) -> bool:
    """Interpret common truthy representations."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def _is_set(v: Any) -> bool:
    """True when v is not None / not the string 'None'."""
    return v is not None and str(v).strip().lower() != "none"


def _get_vals(v: Any) -> List[Any]:
    """
    Extract scalar values from potential grid/random search structures.
    Handles:
      - raw values: 10 -> [10]
      - lists: [10, 20] -> [10, 20]
      - dicts: {'min': 5, 'max': 50} -> [5, 50]
      - nested: [{'min': 5, 'max': 50}, 100] -> [5, 50, 100]
    """
    if v is None: return []
    if isinstance(v, dict):
        return [val for k, val in v.items() if k in ("min", "max") and val is not None]
    if isinstance(v, list):
        res = []
        for item in v:
            if isinstance(item, dict):
                res.extend(_get_vals(item))
            else:
                res.append(item)
        return res
    return [v]



# ---------------------------------------------------------------------------
# Per-algorithm checkers
# ---------------------------------------------------------------------------

def _check_xgboost(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # ── early_stopping_rounds ─────────────────────────────────────────────────────
    esr = params.get("early_stopping_rounds")
    # 0 / None / unset = disabled (safe). Only flag when explicitly > 0.
    try:
        esr_int = int(esr) if esr is not None else 0
    except (TypeError, ValueError):
        esr_int = 0
    if esr_int > 0:
        warnings.append(
            "❌ early_stopping_rounds={val} is not supported when training inside "
            "a sklearn Pipeline.  XGBoost requires an eval_set passed to fit(), "
            "but Pipeline.fit() cannot forward it.  "
            "Remove 'early_stopping_rounds' or set it to 0 to disable early stopping."
            .format(val=esr)
        )

    # ── objective ─────────────────────────────────────────────────────────
    obj = params.get("objective", "")
    _INVALID_OBJECTIVES = ("reg:", "rank:", "count:", "survival:")
    if obj and any(obj.startswith(p) for p in _INVALID_OBJECTIVES):
        warnings.append(
            "❌ objective='{obj}' is a regression/ranking objective but this is "
            "an XGBClassifier (binary fraud detection).  "
            "Valid classification objectives: binary:logistic (recommended), "
            "binary:logitraw, binary:hinge, multi:softprob."
            .format(obj=obj)
        )

    # ── DART-only params with gbtree/gblinear booster ───────────────────────────────────────
    booster = params.get("booster", "gbtree")
    # These params are often present in default XGBoost dicts with zero / default values.
    # Only flag when they are explicitly set to non-zero/non-default values AND
    # the booster is NOT dart (where they would actually be used).
    _DART_DEFAULTS = {
        "rate_drop":      0.0,   # DART: fraction of trees to drop (0 = inactive)
        "skip_drop":      0.0,   # DART: probability of skipping dropout (0 = inactive)
        "normalize_type": "tree", # DART: default value, harmless to pass
        "sample_type":    "uniform",  # DART: default value
        "one_drop":       False,  # DART: default value
    }
    activated_dart = [
        k for k, default_val in _DART_DEFAULTS.items()
        if k in params and params[k] != default_val and params[k] not in (None, "")
    ]
    if activated_dart and booster != "dart":
        warnings.append(
            "⚠️ {params} are DART booster parameters with non-default values, "
            "but booster='{booster}'.  These parameters will be ignored by XGBoost.  "
            "Either set booster='dart' or remove: {params}."
            .format(params=activated_dart, booster=booster)
        )

    # ── learning_rate range ───────────────────────────────────────────────
    for lr in _get_vals(params.get("learning_rate")):
        try:
            lr_f = float(lr)
            if not (0 < lr_f <= 1):
                warnings.append(
                    "⚠️ learning_rate={val} is outside the valid range (0, 1].  "
                    "Typical values: 0.01 – 0.3."
                    .format(val=lr)
                )
        except (TypeError, ValueError):
            warnings.append(
                "❌ learning_rate='{val}' is not a valid number.".format(val=lr)
            )

    # ── subsample / column-sampling bounds ───────────────────────────────
    for key in ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"):
        for val in _get_vals(params.get(key)):
            try:
                v_f = float(val)
                if not (0 < v_f <= 1):
                    warnings.append(
                        "❌ {key}={val} is out of bounds. Valid range is (0, 1].".format(
                            key=key, val=val
                        )
                    )
            except (TypeError, ValueError):
                warnings.append(
                    "❌ {key}='{val}' is not a valid number.".format(key=key, val=val)
                )

    # ── n_estimators ─────────────────────────────────────────────────────
    for n in _get_vals(params.get("n_estimators")):
        try:
            if int(n) <= 0:
                warnings.append(
                    "❌ n_estimators={val} must be > 0.".format(val=n)
                )
        except (TypeError, ValueError):
            warnings.append(
                "❌ n_estimators='{val}' is not a valid integer.".format(val=n)
            )

    # ── max_depth ─────────────────────────────────────────────────────────
    for md in _get_vals(params.get("max_depth")):
        try:
            if int(md) <= 0:
                warnings.append(
                    "❌ max_depth={val} must be > 0.".format(val=md)
                )
        except (TypeError, ValueError):
            warnings.append(
                "❌ max_depth='{val}' is not a valid integer.".format(val=md)
            )


    return warnings


def _check_random_forest(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    def _get_list(v: Any) -> List[Any]:
        if v is None: return [None]
        if isinstance(v, list): return v
        return [v]

    bootstraps = _get_list(params.get("bootstrap", True))
    max_samples_list = _get_list(params.get("max_samples"))
    oob_scores = _get_list(params.get("oob_score", False))

    has_false_bootstrap = any(not _bool_val(b) for b in bootstraps if b is not None)
    has_max_samples = any(_is_set(ms) for ms in max_samples_list)
    has_true_oob = any(_bool_val(oob) for oob in oob_scores if oob is not None)

    # ── bootstrap=False + max_samples ────────────────────────────────────
    if has_false_bootstrap and has_max_samples:
        warnings.append(
            "❌ bootstrap=False and max_samples cannot be used together.  "
            "max_samples controls how many samples to draw per tree, which only "
            "applies when bootstrap=True (sampling WITH replacement).  "
            "Fix: either set bootstrap=True, or remove max_samples."
        )

    # ── bootstrap=False + oob_score=True ─────────────────────────────────
    if has_false_bootstrap and has_true_oob:
        warnings.append(
            "❌ oob_score=True requires bootstrap=True.  "
            "Out-of-bag scoring is computed from samples NOT drawn during "
            "bootstrap — it cannot exist when bootstrap=False.  "
            "Fix: either set bootstrap=True, or set oob_score=False."
        )

    # ── n_estimators ─────────────────────────────────────────────────────
    for n in _get_vals(params.get("n_estimators")):
        try:
            if int(n) <= 0:
                warnings.append(f"❌ n_estimators={n} must be > 0.")
        except (TypeError, ValueError):
            warnings.append(f"❌ n_estimators='{n}' is not a valid integer.")


    # ── Soft dependencies (Performance / tuning warnings) ────────────────
    def _safe_int(v):
        if v in (None, -1, "-1", "None", ""): return None
        if isinstance(v, dict):  # range dict {'min': X, 'max': Y} from tuning UI
            raw = v.get("min") or v.get("max")
            if raw is None: return None
            try: return int(raw)
            except (TypeError, ValueError): return None
        if isinstance(v, list):
            if not v: return None
            v = v[0]  # Just check the first element of the grid for soft warnings
            if isinstance(v, dict): return None
        try: return int(v)
        except (TypeError, ValueError): return None

    n_est_int = _safe_int(params.get("n_estimators"))
    md_int = _safe_int(params.get("max_depth"))
    msl_int = _safe_int(params.get("min_samples_leaf"))
    mss_int = _safe_int(params.get("min_samples_split"))

    # 4 & 5: max_depth vs min_samples (Overfitting / Underfitting)
    if md_int is not None and md_int > 30:
        if msl_int is not None and msl_int <= 2:
            warnings.append("⚠️ max_depth is very high (>30) and min_samples_leaf is very low (<=2). This combination may lead to severe overfitting. Consider increasing min_samples_leaf or decreasing max_depth.")
            
    if md_int is not None and md_int < 5:
        if msl_int is not None and msl_int > 20:
            warnings.append("⚠️ max_depth is very low (<5) and min_samples_leaf is very high (>20). This combination may lead to underfitting. Consider decreasing min_samples_leaf or increasing max_depth.")
        elif mss_int is not None and mss_int > 50:
            warnings.append("⚠️ max_depth is very low (<5) and min_samples_split is very high (>50). This combination may lead to underfitting. Consider decreasing min_samples_split or increasing max_depth.")

    # 6: max_features vs n_estimators
    mf_list = _get_list(params.get("max_features"))
    for mf in mf_list:
        mf_is_low = False
        if mf == "log2":
            mf_is_low = True
        elif mf not in (None, "sqrt", "auto", "None"):
            try:
                if float(mf) < 0.3:
                    mf_is_low = True
            except (ValueError, TypeError):
                pass
        if mf_is_low and n_est_int is not None and n_est_int < 50:
            warnings.append("⚠️ max_features is low, which increases tree randomness. A low n_estimators (<50) might not be enough to reach a stable ensemble. Consider increasing n_estimators.")
            break

    # 7: max_samples vs n_estimators
    if not has_false_bootstrap and has_max_samples:
        for ms in max_samples_list:
            if ms is None: continue
            try:
                ms_float = float(ms)
                if ms_float < 0.5 and n_est_int is not None and n_est_int < 50:
                    warnings.append("⚠️ max_samples is low (<0.5), meaning each tree sees a small fraction of data. A low n_estimators (<50) might not be sufficient. Consider increasing n_estimators.")
                    break
            except (TypeError, ValueError):
                pass

    # 8: ccp_alpha vs tree complexity
    ccp_list = _get_list(params.get("ccp_alpha"))
    for ccp in ccp_list:
        if ccp is None: continue
        try:
            ccp_float = float(ccp)
            if ccp_float < 0:
                warnings.append("❌ ccp_alpha={val} must be >= 0.".format(val=ccp))
            elif ccp_float > 0.1 and md_int is not None and md_int < 10:
                warnings.append("⚠️ ccp_alpha is high (>0.1), which causes aggressive pruning. Combined with a low max_depth, this might prune the tree down to a stump. Consider lowering ccp_alpha.")
        except (TypeError, ValueError):
            pass

    return warnings



def _check_lightgbm(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    is_unbalance = _bool_val(params.get("is_unbalance", False))
    scale_pos_weight = params.get("scale_pos_weight")
    class_weight = params.get("class_weight")

    # ── is_unbalance + scale_pos_weight mutual exclusion ─────────────────
    if is_unbalance and _is_set(scale_pos_weight) and str(scale_pos_weight) != "1.0":
        warnings.append(
            "❌ is_unbalance=True and scale_pos_weight={val} cannot be set "
            "simultaneously.  LightGBM will raise an error.  "
            "Use one or the other: is_unbalance=True OR scale_pos_weight={val}."
            .format(val=scale_pos_weight)
        )

    # ── is_unbalance + class_weight='balanced' ────────────────────────────
    if is_unbalance and class_weight == "balanced":
        warnings.append(
            "❌ is_unbalance=True and class_weight='balanced' are redundant and "
            "LightGBM may raise a conflict error.  "
            "Use only one: either is_unbalance=True or class_weight='balanced'."
        )

    # ── learning_rate ─────────────────────────────────────────────────────
    for lr in _get_vals(params.get("learning_rate")):
        try:
            if not (0 < float(lr) <= 1):
                warnings.append(
                    "⚠️ learning_rate={val} is outside the valid range (0, 1].".format(val=lr)
                )
        except (TypeError, ValueError):
            warnings.append(
                "❌ learning_rate='{val}' is not a valid number.".format(val=lr)
            )


    return warnings


def _check_svm(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # ── C must be > 0 ─────────────────────────────────────────────────────
    for c in _get_vals(params.get("C")):
        try:
            if float(c) <= 0:
                warnings.append(
                    "❌ C={val} must be > 0.  "
                    "C is the regularization parameter: smaller = more regularization."
                    .format(val=c)
                )
        except (TypeError, ValueError):
            warnings.append("❌ C='{val}' is not a valid number.".format(val=c))

    # ── degree must be >= 0 ───────────────────────────────────────────────
    for deg in _get_vals(params.get("degree")):
        try:
            if int(deg) < 0:
                warnings.append("❌ degree={val} must be >= 0. Usually tuning range is 2-5.".format(val=deg))
        except (TypeError, ValueError):
            warnings.append("❌ degree='{val}' is not a valid integer.".format(val=deg))

    # ── gamma check ───────────────────────────────────────────────────────
    for g in _get_vals(params.get("gamma")):
        if str(g).lower() not in ("scale", "auto", ""):
            try:
                g_f = float(g)
                if g_f < 0:
                    warnings.append("❌ gamma={val} must be >= 0 if float.".format(val=g))
            except (TypeError, ValueError):
                warnings.append(
                    "❌ gamma='{val}' is invalid. Use 'scale', 'auto', or a float >= 0.".format(val=g)
                )

    # ── tol must be > 0 ───────────────────────────────────────────────────
    for tol in _get_vals(params.get("tol")):
        try:
            if float(tol) <= 0:
                warnings.append("❌ tol={val} must be > 0. Default is 1e-3.".format(val=tol))
        except (TypeError, ValueError):
            warnings.append("❌ tol='{val}' is not a valid float.".format(val=tol))

    # ── cache_size must be > 0 ────────────────────────────────────────────
    for cache in _get_vals(params.get("cache_size")):
        try:
            if int(cache) <= 0:
                warnings.append("❌ cache_size={val} must be > 0 MB.".format(val=cache))
        except (TypeError, ValueError):
            warnings.append("❌ cache_size='{val}' is not a valid integer.".format(val=cache))

    # ── max_iter check ────────────────────────────────────────────────────
    for mi in _get_vals(params.get("max_iter")):
        try:
            mi_val = int(mi)
            if mi_val < -1 or mi_val == 0:
                warnings.append("❌ max_iter={val} must be > 0, or -1 for no limit.".format(val=mi))
        except (TypeError, ValueError):
            warnings.append("❌ max_iter='{val}' is not a valid integer.".format(val=mi))
    return warnings


def _check_logistic_regression(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    solvers = set(_get_vals(params.get("solver", "lbfgs")))
    penalties = set(_get_vals(params.get("penalty", "l2")))

    # Normalize "None" string → None
    penalties = {None if str(p).lower() == "none" else p for p in penalties}

    # ── solver + penalty compatibility ────────────────────────────────────
    _SOLVER_PENALTIES: Dict[str, set] = {
        "lbfgs":      {"l2", None},
        "liblinear":  {"l1", "l2"},
        "saga":       {"elasticnet", "l1", "l2", None},
        "sag":        {"l2", None},
        "newton-cg":  {"l2", None},
    }

    for s in solvers:
        allowed = _SOLVER_PENALTIES.get(str(s), set())
        if allowed:
            # Check if ANY selected penalty is incompatible with this solver
            for p in penalties:
                if p not in allowed:
                    warnings.append(
                        "❌ solver='{solver}' does not support penalty='{penalty}'. "
                        "Supported penalties: {allowed}."
                        .format(
                            solver=s,
                            penalty=p,
                            allowed=sorted(str(a) for a in allowed)
                        )
                    )

        # ── elasticnet requires saga ───────────────────────────────────────────
        if "elasticnet" in penalties and str(s) != "saga":
            warnings.append(
                "❌ penalty='elasticnet' requires solver='saga'. "
                "Current solver='{solver}'."
                .format(solver=s)
            )

    # ── l1_ratio only valid with elasticnet ───────────────────────────────
    for l1_ratio in _get_vals(params.get("l1_ratio")):
        if "elasticnet" not in penalties:
            warnings.append(
                "⚠️ l1_ratio={val} is only used when penalty='elasticnet' and solver='saga'. "
                "It will be ignored."
                .format(val=l1_ratio)
            )
        else:
            try:
                r = float(l1_ratio)
                if not (0.0 <= r <= 1.0):
                    warnings.append(
                        "❌ l1_ratio={val} must be in [0, 1]. "
                        "0 = L2 only, 1 = L1 only, 0.5 = equal mix.".format(val=l1_ratio)
                    )
            except (TypeError, ValueError):
                warnings.append(
                    "❌ l1_ratio='{val}' is not a valid float.".format(val=l1_ratio)
                )


    # ── dual: only valid for liblinear + l2 ───────────────────────────────
    dual = params.get("dual")
    if dual is not None and _bool_val(dual):
        # Only valid if all tested combinations are liblinear + l2
        if any(s != "liblinear" for s in solvers) or any(p != "l2" for p in penalties):
            warnings.append(
                "⚠️ dual=True is only supported when solver='liblinear' and penalty='l2'. "
                "Set dual=False (default) or restrict your tuning to solver='liblinear' + penalty='l2'."
            )

    # ── multinomial not supported by liblinear ────────────────────────────
    multi_class = params.get("multi_class", "auto")
    if multi_class == "multinomial" and "liblinear" in solvers:
        warnings.append(
            "❌ multi_class='multinomial' is not supported by solver='liblinear'. "
            "Use solver='lbfgs', 'sag', or 'saga' for multinomial."
        )

    # ── C must be > 0 ─────────────────────────────────────────────────────
    for c in _get_vals(params.get("C")):
        try:
            if float(c) <= 0:
                warnings.append(
                    "❌ C={val} must be > 0. "
                    "Smaller C = stronger regularization; larger C = weaker.".format(val=c)
                )
        except (TypeError, ValueError):
            warnings.append("❌ C='{val}' is not a valid number.".format(val=c))


    # ── tol must be > 0 ───────────────────────────────────────────────────
    for tol in _get_vals(params.get("tol")):
        try:
            if float(tol) <= 0:
                warnings.append(
                    "❌ tol={val} must be > 0. Default is 1e-4.".format(val=tol)
                )
        except (TypeError, ValueError):
            warnings.append("❌ tol='{val}' is not a valid float.".format(val=tol))


    # ── max_iter must be > 0 ──────────────────────────────────────────────
    for max_iter in _get_vals(params.get("max_iter")):
        try:
            if int(max_iter) <= 0:
                warnings.append(
                    "❌ max_iter={val} must be > 0. "
                    "Increase to 200–1000 if solver convergence warnings appear.".format(val=max_iter)
                )
        except (TypeError, ValueError):
            warnings.append("❌ max_iter='{val}' is not a valid integer.".format(val=max_iter))


    # ── intercept_scaling: only meaningful with liblinear + fit_intercept ──
    iscale = params.get("intercept_scaling")
    if iscale is not None:
        try:
            if float(iscale) != 1.0:   # 1.0 is the default — silently ignore unchanged default
                # If any chosen solver is not liblinear, it will ignore intercept_scaling.
                if any(str(s) != "liblinear" for s in solvers):
                    warnings.append(
                        "⚠️ intercept_scaling={val} only has effect when solver='liblinear'. "
                        "Non-liblinear solver(s) will ignore this value.".format(val=iscale)
                    )
        except (TypeError, ValueError):
            pass

    return warnings



def _check_neural_network(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # ── hidden_layer_sizes ────────────────────────────────────────────────
    for hls in _get_vals(params.get("hidden_layer_sizes")):
        try:
            val = int(hls)
            if val < 1:
                warnings.append("❌ hidden_layer_sizes={val} must be >= 1.".format(val=val))
        except (TypeError, ValueError):
            # Tuples come through here as un-int-able, which is fine, we ignore them
            pass

    # ── alpha ─────────────────────────────────────────────────────────────
    for alpha in _get_vals(params.get("alpha")):
        try:
            if float(alpha) < 0:
                warnings.append("❌ alpha={val} must be >= 0.".format(val=alpha))
        except (TypeError, ValueError):
            pass

    # ── max_iter ──────────────────────────────────────────────────────────
    for max_iter in _get_vals(params.get("max_iter")):
        try:
            if int(max_iter) <= 0:
                warnings.append("❌ max_iter={val} must be > 0. Default is 200.".format(val=max_iter))
        except (TypeError, ValueError):
            warnings.append("❌ max_iter='{val}' is not a valid integer.".format(val=max_iter))

    # ── batch_size ────────────────────────────────────────────────────────
    for bs in _get_vals(params.get("batch_size")):
        if str(bs).lower() not in ("auto", ""):
            try:
                if int(bs) <= 0:
                    warnings.append("❌ batch_size={val} must be > 0.".format(val=bs))
            except (TypeError, ValueError):
                warnings.append("❌ batch_size='{val}' is not valid. Use 'auto' or an integer > 0.".format(val=bs))

    # ── learning_rate_init ────────────────────────────────────────────────
    for lr in _get_vals(params.get("learning_rate_init")):
        try:
            if float(lr) <= 0:
                warnings.append("❌ learning_rate_init={val} must be > 0.".format(val=lr))
        except (TypeError, ValueError):
            warnings.append("❌ learning_rate_init='{val}' is not a valid float.".format(val=lr))

    # ── momentum ──────────────────────────────────────────────────────────
    for m in _get_vals(params.get("momentum")):
        try:
            mf = float(m)
            if mf < 0 or mf > 1:
                warnings.append("❌ momentum={val} must be between 0 and 1.".format(val=m))
        except (TypeError, ValueError):
            warnings.append("❌ momentum='{val}' is not a valid float.".format(val=m))

    # ── validation_fraction ───────────────────────────────────────────────
    for vf in _get_vals(params.get("validation_fraction")):
        try:
            v_float = float(vf)
            if v_float <= 0 or v_float >= 1:
                warnings.append("❌ validation_fraction={val} must be between 0 and 1 (exclusive).".format(val=vf))
        except (TypeError, ValueError):
            warnings.append("❌ validation_fraction='{val}' is not a valid float.".format(val=vf))

    # ── Adam Betas and Epsilon ────────────────────────────────────────────
    for b1 in _get_vals(params.get("beta_1")):
        try:
            b1_f = float(b1)
            if b1_f < 0 or b1_f >= 1:
                warnings.append("❌ beta_1={val} must be in [0, 1).".format(val=b1))
        except (TypeError, ValueError):
            warnings.append("❌ beta_1='{val}' is not a valid float.".format(val=b1))

    for b2 in _get_vals(params.get("beta_2")):
        try:
            b2_f = float(b2)
            if b2_f < 0 or b2_f >= 1:
                warnings.append("❌ beta_2={val} must be in [0, 1).".format(val=b2))
        except (TypeError, ValueError):
            warnings.append("❌ beta_2='{val}' is not a valid float.".format(val=b2))

    for eps in _get_vals(params.get("epsilon")):
        try:
            if float(eps) <= 0:
                warnings.append("❌ epsilon={val} must be > 0.".format(val=eps))
        except (TypeError, ValueError):
            warnings.append("❌ epsilon='{val}' is not a valid float.".format(val=eps))

    return warnings



# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

    return warnings


def _get_vals(v: Any) -> List[Any]:
    """
    Extract scalar values from potential grid/random search structures.
    Handles:
      - raw values: 10 -> [10]
      - lists: [10, 20] -> [10, 20]
      - dicts: {'min': 5, 'max': 50} -> [5, 50]
      - nested: [{'min': 5, 'max': 50}, 100] -> [5, 50, 100]
    """
    if v is None: return []
    if isinstance(v, dict):
        return [val for k, val in v.items() if k in ("min", "max") and val is not None]
    if isinstance(v, list):
        res = []
        for item in v:
            if isinstance(item, dict):
                res.extend(_get_vals(item))
            else:
                res.append(item)
        return res
    return [v]


def _check_decision_tree(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # ── max_depth ─────────────────────────────────────────────────────────
    for md in _get_vals(params.get("max_depth")):
        if str(md).lower() not in ("none", ""):
            try:
                if int(md) <= 0:
                    warnings.append(f"❌ max_depth={md} must be > 0 (or leave unset/None for unlimited depth).")
            except (TypeError, ValueError):
                warnings.append(f"❌ max_depth='{md}' is not a valid integer.")

    # ── min_samples_split / min_samples_leaf ──────────────────────────────
    for m_key in ("min_samples_split", "min_samples_leaf"):
        for val in _get_vals(params.get(m_key)):
            try:
                fval = float(val)
                if fval < 0.0 or (m_key == "min_samples_split" and fval == 0.0):
                    warnings.append(f"❌ {m_key}={val} must be > 0.")
            except (TypeError, ValueError):
                pass
                
    # ── max_leaf_nodes ────────────────────────────────────────────────────
    for mln in _get_vals(params.get("max_leaf_nodes")):
        if str(mln).lower() not in ("none", ""):
            try:
                if int(mln) < 2:
                    warnings.append(f"❌ max_leaf_nodes={mln} must be >= 2 (or leave unset/None for unlimited nodes).")
            except (TypeError, ValueError):
                warnings.append(f"❌ max_leaf_nodes='{mln}' is not a valid integer.")

    # ── min_weight_fraction_leaf / min_impurity_decrease / ccp_alpha ──────
    for f_key in ("min_weight_fraction_leaf", "min_impurity_decrease", "ccp_alpha"):
        for val in _get_vals(params.get(f_key)):
            try:
                fval = float(val)
                if fval < 0:
                    warnings.append(f"❌ {f_key}={val} must be >= 0.")
                if f_key == "min_weight_fraction_leaf" and fval > 0.5:
                    warnings.append(f"❌ min_weight_fraction_leaf={val} must be <= 0.5.")
            except (TypeError, ValueError):
                pass

    return warnings


def _check_knn(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # ── n_neighbors ───────────────────────────────────────────────────────
    for k in _get_vals(params.get("n_neighbors")):
        try:
            if int(k) < 1:
                warnings.append(
                    "❌ n_neighbors={val} must be >= 1.".format(val=k)
                )
        except (TypeError, ValueError):
            warnings.append("❌ n_neighbors='{val}' is not a valid integer.".format(val=k))

    # ── leaf_size ─────────────────────────────────────────────────────────
    for ls in _get_vals(params.get("leaf_size")):
        try:
            if int(ls) < 1:
                warnings.append("❌ leaf_size={val} must be >= 1.".format(val=ls))
        except (TypeError, ValueError):
            warnings.append("❌ leaf_size='{val}' is not a valid integer.".format(val=ls))

    # ── p (Minkowski power) ───────────────────────────────────────────────
    for p in _get_vals(params.get("p")):
        try:
            if float(p) < 1:
                warnings.append(
                    "❌ p={val} (Minkowski power) must be >= 1.  "
                    "Common values: 1=Manhattan, 2=Euclidean.".format(val=p)
                )
        except (TypeError, ValueError):
            pass

    # ── n_jobs ────────────────────────────────────────────────────────────
    for nj in _get_vals(params.get("n_jobs")):
        try:
            nj_val = int(nj)
            if nj_val < -1 or nj_val == 0:
                warnings.append("❌ n_jobs={val} must be >= 1, or -1 for all cores.".format(val=nj))
        except (TypeError, ValueError):
            warnings.append("❌ n_jobs='{val}' is not a valid integer.".format(val=nj))

    return warnings


def _check_naive_bayes(params: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    # ── var_smoothing ─────────────────────────────────────────────────────
    for vs in _get_vals(params.get("var_smoothing")):
        try:
            if float(vs) <= 0:
                warnings.append(
                    "❌ var_smoothing={val} must be > 0.  "
                    "It is added to variances to prevent division by zero."
                    .format(val=vs)
                )
        except (TypeError, ValueError):
            pass


    return warnings


_CHECKERS = {
    # Tree / ensemble
    "xgboost":             _check_xgboost,
    "lightgbm":            _check_lightgbm,
    "random_forest":       _check_random_forest,
    "decision_tree":       _check_decision_tree,
    # Linear
    "logistic_regression": _check_logistic_regression,
    "svm":                 _check_svm,
    # Instance-based
    "knn":                 _check_knn,
    # Probabilistic
    "naive_bayes":         _check_naive_bayes,
    # Neural
    "neural_network":      _check_neural_network,
}
