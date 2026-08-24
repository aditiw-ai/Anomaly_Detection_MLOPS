"""
Model Explainability
SHAP-based explanations for fraud predictions.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    feature_contributions: Dict[str, float]
    base_value: float
    prediction_value: float
    top_positive_features: List[Dict[str, Any]]
    top_negative_features: List[Dict[str, Any]]
    explanation_text: str


class FraudExplainer:
    """
    Generates human-readable explanations for fraud predictions.
    
    Uses SHAP (SHapley Additive exPlanations) for feature attribution.
    """
    
    def __init__(self, model: Any = None, feature_names: List[str] = None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (XGBoost, LightGBM, or sklearn)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        
        if model is not None:
            self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            import shap
            
            model_type = type(self.model).__name__
            
            if "XGB" in model_type or "LGBM" in model_type:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info(f"Initialized TreeExplainer for {model_type}")
            else:
                # Use KernelExplainer for other models (slower)
                logger.info(f"TreeExplainer not available for {model_type}, using summary")
                self.explainer = None
                
        except ImportError:
            logger.warning("SHAP not installed, explanations will be limited")
            self.explainer = None
    
    def explain(
        self,
        features: np.ndarray,
        feature_names: List[str] = None,
    ) -> ExplanationResult:
        """
        Generate explanation for a single prediction.
        
        Args:
            features: Feature array (1, n_features)
            feature_names: Optional feature names override
        
        Returns:
            ExplanationResult with feature attributions
        """
        names = feature_names or self.feature_names
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self.explainer is not None:
            return self._explain_with_shap(features, names)
        else:
            return self._explain_fallback(features, names)
    
    def _explain_with_shap(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> ExplanationResult:
        """Generate SHAP-based explanation."""
        import shap
        
        shap_values = self.explainer.shap_values(features)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        values = shap_values[0]
        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
        
        # Build feature contributions
        contributions = {}
        for i, name in enumerate(feature_names[:len(values)]):
            contributions[name] = float(values[i])
        
        # Sort by absolute contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top positive and negative features
        top_positive = [
            {"feature": k, "contribution": v, "value": float(features[0, i])}
            for i, (k, v) in enumerate(sorted_contributions) if v > 0
        ][:5]
        
        top_negative = [
            {"feature": k, "contribution": v, "value": float(features[0, i])}
            for i, (k, v) in enumerate(sorted_contributions) if v < 0
        ][:5]
        
        # Generate explanation text
        prediction_value = base_value + sum(values)
        explanation_text = self._generate_explanation_text(
            top_positive, top_negative, prediction_value
        )
        
        return ExplanationResult(
            feature_contributions=contributions,
            base_value=base_value,
            prediction_value=float(prediction_value),
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_text=explanation_text,
        )
    
    def _explain_fallback(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> ExplanationResult:
        """
        Fallback explanation when SHAP is not available.
        Uses feature importance from model if available.
        """
        contributions = {}
        
        # Try to get feature importance from model
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            for i, name in enumerate(feature_names[:len(importance)]):
                # Approximate contribution as importance * feature_value
                contributions[name] = float(importance[i] * features[0, i])
        else:
            # Just use feature values as placeholder
            for i, name in enumerate(feature_names[:features.shape[1]]):
                contributions[name] = float(features[0, i])
        
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        top_positive = [
            {"feature": k, "contribution": v}
            for k, v in sorted_contributions if v > 0
        ][:5]
        
        top_negative = [
            {"feature": k, "contribution": v}
            for k, v in sorted_contributions if v < 0
        ][:5]
        
        return ExplanationResult(
            feature_contributions=contributions,
            base_value=0.5,
            prediction_value=0.0,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_text="Feature importance-based explanation (SHAP not available)",
        )
    
    def _generate_explanation_text(
        self,
        top_positive: List[Dict],
        top_negative: List[Dict],
        prediction_value: float,
    ) -> str:
        """Generate human-readable explanation text."""
        lines = []
        
        risk_level = "High" if prediction_value > 0.7 else "Medium" if prediction_value > 0.4 else "Low"
        lines.append(f"Risk Level: {risk_level} (Score: {prediction_value:.2f})")
        lines.append("")
        
        if top_positive:
            lines.append("Factors increasing fraud risk:")
            for f in top_positive[:3]:
                lines.append(f"  • {f['feature']}: +{f['contribution']:.3f}")
        
        if top_negative:
            lines.append("")
            lines.append("Factors decreasing fraud risk:")
            for f in top_negative[:3]:
                lines.append(f"  • {f['feature']}: {f['contribution']:.3f}")
        
        return "\n".join(lines)
