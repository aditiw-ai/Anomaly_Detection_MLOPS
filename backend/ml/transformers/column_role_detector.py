"""
Column Role Detector
Auto-detects semantic roles of DataFrame columns for feature engineering.
Supports user-provided overrides via column_mapping.
"""
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns (case-insensitive)
# ---------------------------------------------------------------------------
AMOUNT_PATTERNS = re.compile(
    r"(amount|amt|value|price|cost|total|sum|balance|charge|fee|spend|revenue)",
    re.IGNORECASE,
)
TIMESTAMP_PATTERNS = re.compile(
    r"(date|time|timestamp|datetime|created|updated|occurred|when|ts\b)",
    re.IGNORECASE,
)
USER_PATTERNS = re.compile(
    r"(user|customer|client|account|member|person|buyer|sender|payer|cardholder)",
    re.IGNORECASE,
)
ID_PATTERNS = re.compile(
    r"(^id$|_id$|^id_|transaction.?id|order.?id|record.?id|row.?id|index|^pk$|^sk$)",
    re.IGNORECASE,
)
TARGET_NAMES = [
    "is_fraud", "fraud_label", "class", "target", "label",
    "is_fraudulent", "is fraudulent", "is fraud", "fraud", "fraudulent",
    "is_fraud?", "fraud?", # Handle question marks if present
    "_fraud_label",
]
# Merchant / category-style column names (hint, not decisive)
MERCHANT_PATTERNS = re.compile(
    r"(merchant|vendor|seller|store|shop|retailer|category|product.?cat|channel|type|method|device)",
    re.IGNORECASE,
)

# Thresholds
CATEGORY_CARDINALITY_MAX = 50   # max unique values to treat as categorical
ID_CARDINALITY_RATIO = 0.80     # if nunique / len > this, probably an ID
POSITIVE_VALUE_RATIO = 0.90     # fraction of values > 0 to confirm "amount"


@dataclass
class ColumnRoles:
    """Detected semantic roles for DataFrame columns."""
    amount_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    user_col: Optional[str] = None
    target_col: Optional[str] = None
    category_cols: List[str] = field(default_factory=list)
    numeric_cols: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=list)
    # reverse lookup: original col name -> semantic role used by the mapping
    applied_mapping: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "amount_col": self.amount_col,
            "timestamp_col": self.timestamp_col,
            "user_col": self.user_col,
            "target_col": self.target_col,
            "category_cols": self.category_cols,
            "numeric_cols": self.numeric_cols,
            "id_cols": self.id_cols,
        }


class ColumnRoleDetector:
    """
    Detects semantic roles of DataFrame columns.

    Priority order:
        1. User-provided column_mapping overrides everything.
        2. Heuristic detection fills remaining roles.

    Usage::

        detector = ColumnRoleDetector()
        roles = detector.detect(df, column_mapping={"Transaction Amount": "amount"})
    """

    # Reverse role map: semantic role -> attribute name on ColumnRoles
    _ROLE_TO_ATTR = {
        "amount": "amount_col",
        "timestamp": "timestamp_col",
        "user_id": "user_col",
        "target": "target_col",
    }

    def detect(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> ColumnRoles:
        """
        Detect column roles.

        Parameters
        ----------
        df : pd.DataFrame
            The raw DataFrame to inspect.
        column_mapping : dict, optional
            User-provided mapping  ``{original_col_name: semantic_role}``.
            Recognised roles: ``amount``, ``timestamp``, ``user_id``, ``target``.
            Any column mapped to an unrecognised role is treated as a rename
            hint and the column is classified by heuristic as usual.

        Returns
        -------
        ColumnRoles
            Dataclass with all detected roles populated.
        """
        roles = ColumnRoles()
        assigned: set = set()  # columns already assigned a role

        # ------------------------------------------------------------------
        # Phase 1: Apply user-provided mapping
        # ------------------------------------------------------------------
        if column_mapping:
            for orig_col, role in column_mapping.items():
                if orig_col not in df.columns:
                    logger.warning(
                        f"column_mapping key '{orig_col}' not found in DataFrame "
                        f"(available: {list(df.columns)[:10]}...)"
                    )
                    continue
                role_lower = role.lower().strip()
                attr = self._ROLE_TO_ATTR.get(role_lower)
                if attr:
                    setattr(roles, attr, orig_col)
                    assigned.add(orig_col)
                    roles.applied_mapping[orig_col] = role_lower
                    logger.debug(f"User mapping: '{orig_col}' -> {role_lower}")
                else:
                    # Unrecognised role — might be a category hint
                    logger.debug(
                        f"User mapping: '{orig_col}' -> '{role_lower}' (unrecognised role, will auto-classify)"
                    )

        # ------------------------------------------------------------------
        # Phase 2: Auto-detect target column (if not mapped)
        # ------------------------------------------------------------------
        if roles.target_col is None:
            roles.target_col = self._detect_target(df, assigned)
            if roles.target_col:
                assigned.add(roles.target_col)

        # ------------------------------------------------------------------
        # Phase 3: Auto-detect timestamp (if not mapped)
        # ------------------------------------------------------------------
        if roles.timestamp_col is None:
            roles.timestamp_col = self._detect_timestamp(df, assigned)
            if roles.timestamp_col:
                assigned.add(roles.timestamp_col)

        # ------------------------------------------------------------------
        # Phase 4: Auto-detect amount (if not mapped)
        # ------------------------------------------------------------------
        if roles.amount_col is None:
            roles.amount_col = self._detect_amount(df, assigned)
            if roles.amount_col:
                assigned.add(roles.amount_col)

        # ------------------------------------------------------------------
        # Phase 5: Auto-detect user / account column (if not mapped)
        # ------------------------------------------------------------------
        if roles.user_col is None:
            roles.user_col = self._detect_user(df, assigned)
            if roles.user_col:
                assigned.add(roles.user_col)

        # ------------------------------------------------------------------
        # Phase 6: Classify remaining columns as ID, category, or numeric
        # ------------------------------------------------------------------
        for col in df.columns:
            if col in assigned:
                continue
            dtype = df[col].dtype

            # Check for ID columns first
            if self._is_id_column(df, col):
                roles.id_cols.append(col)
                assigned.add(col)
                continue

            # Categorical: object/string dtype OR low-cardinality numeric
            if dtype == object or str(dtype) == "category":
                if df[col].nunique() <= CATEGORY_CARDINALITY_MAX:
                    roles.category_cols.append(col)
                else:
                    roles.id_cols.append(col)  # high-card string -> treat as ID
                assigned.add(col)
                continue

            # Boolean
            if dtype == bool or dtype == "bool":
                roles.numeric_cols.append(col)
                assigned.add(col)
                continue

            # Numeric
            if pd.api.types.is_numeric_dtype(dtype):
                roles.numeric_cols.append(col)
                assigned.add(col)
                continue

            # Datetime that wasn't caught as timestamp
            if pd.api.types.is_datetime64_any_dtype(dtype):
                # Extra datetime col — treat as numeric (epoch)
                roles.numeric_cols.append(col)
                assigned.add(col)
                continue

            # Fallback: try to see if it's a parseable date string
            if dtype == object:
                roles.category_cols.append(col)
                assigned.add(col)

        logger.info(f"Column roles detected: {roles.summary()}")
        return roles

    # ------------------------------------------------------------------
    # Private detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_target(df: pd.DataFrame, assigned: set) -> Optional[str]:
        """Detect target / label column."""
        # Exact match (case-insensitive, ignoring punctuation/spaces)
        import re
        def _normalize(s):
            return re.sub(r'[^a-z0-9]', '', s.lower())

        # Map: normalized_name -> original_name
        norm_map = {_normalize(c): c for c in df.columns if c not in assigned}
        
        for name in TARGET_NAMES:
            norm_target = _normalize(name)
            if norm_target in norm_map:
                original_col = norm_map[norm_target]
                logger.debug(f"Auto-detected target column (fuzzy match): '{original_col}'")
                return original_col

        # Check for binary columns with suggestive names
        for col in df.columns:
            if col in assigned:
                continue
            if df[col].nunique() == 2 and re.search(r"fraud|label|class|target", col, re.IGNORECASE):
                logger.debug(f"Auto-detected target column (binary heuristic): '{col}'")
                return col

        return None

    @staticmethod
    def _detect_timestamp(df: pd.DataFrame, assigned: set) -> Optional[str]:
        """Detect timestamp column."""
        # First: explicit datetime dtype
        for col in df.columns:
            if col in assigned:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.debug(f"Auto-detected timestamp column (datetime dtype): '{col}'")
                return col

        # Second: name heuristic + parseable
        for col in df.columns:
            if col in assigned:
                continue
            if TIMESTAMP_PATTERNS.search(col):
                # Try to parse a sample
                try:
                    sample = df[col].dropna().head(20)
                    pd.to_datetime(sample)
                    logger.debug(f"Auto-detected timestamp column (name + parseable): '{col}'")
                    return col
                except (ValueError, TypeError):
                    continue

        return None

    @staticmethod
    def _detect_amount(df: pd.DataFrame, assigned: set) -> Optional[str]:
        """Detect amount / monetary value column."""
        candidates = []
        for col in df.columns:
            if col in assigned:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if AMOUNT_PATTERNS.search(col):
                # Confirm mostly positive
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    continue
                positive_ratio = (non_null > 0).mean()
                candidates.append((col, positive_ratio))

        if candidates:
            # Pick the candidate with highest positive ratio
            best = max(candidates, key=lambda x: x[1])
            logger.debug(f"Auto-detected amount column: '{best[0]}' (positive ratio: {best[1]:.2f})")
            return best[0]

        return None

    @staticmethod
    def _detect_user(df: pd.DataFrame, assigned: set) -> Optional[str]:
        """Detect user / customer identifier column."""
        for col in df.columns:
            if col in assigned:
                continue
            if USER_PATTERNS.search(col):
                logger.debug(f"Auto-detected user column: '{col}'")
                return col
        return None

    @staticmethod
    def _is_id_column(df: pd.DataFrame, col: str) -> bool:
        """Heuristic: is this column an identifier (should be dropped)?"""
        # Name heuristic
        if ID_PATTERNS.search(col):
            return True

        # High-cardinality integer with sequential-looking values
        if pd.api.types.is_integer_dtype(df[col]):
            nunique_ratio = df[col].nunique() / max(len(df), 1)
            if nunique_ratio > ID_CARDINALITY_RATIO:
                return True

        return False
