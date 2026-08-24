"""
Feature Service
Business logic for feature engineering operations.
"""
from app.core.time import now_ist
from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID
import logging
import asyncio

import json
import hashlib
import io
import copy
import pandas as pd
from sqlalchemy import select, func, update, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feature_set import FeatureSet
from app.models.dataset import Dataset
from app.core.storage import storage_service

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for feature engineering operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_feature_sets(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[FeatureSet], int]:
        """List feature sets with pagination and filtering."""
        query = select(FeatureSet).order_by(FeatureSet.created_at.desc())
        count_query = select(func.count(FeatureSet.id))
        
        # Filter non-deleted
        conditions = [FeatureSet.is_deleted == False]
        
        if dataset_id:
            conditions.append(FeatureSet.dataset_id == UUID(dataset_id))
        
        if status:
            conditions.append(FeatureSet.status == status)
            
        # Apply conditions
        query = query.where(and_(*conditions))
        count_query = count_query.where(and_(*conditions))
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        result = await self.db.execute(query)
        feature_sets = result.scalars().all()
        
        return list(feature_sets), total
    
    async def get_feature_set(self, feature_set_id: str) -> Optional[FeatureSet]:
        """Get a single feature set by ID."""
        try:
            uuid_id = UUID(feature_set_id)
        except ValueError:
            return None
        
        result = await self.db.execute(
            select(FeatureSet).where(FeatureSet.id == uuid_id)
        )
        return result.scalar_one_or_none()

    async def get_preprocessing_plan(self, feature_set_id: str) -> Optional[Dict[str, Any]]:
        """Return preprocessing profile + recommendations + decisions for a feature set."""
        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set:
            return None

        if not feature_set.preprocessing_profile or not feature_set.preprocessing_recommendations:
            profile, recommendations = await self._generate_preprocessing_plan(feature_set)
            feature_set.preprocessing_profile = profile
            feature_set.preprocessing_recommendations = recommendations
            if not feature_set.preprocessing_decisions:
                feature_set.preprocessing_decisions = self._default_decisions(recommendations)
            await self.db.commit()
            await self.db.refresh(feature_set)

        decisions = feature_set.preprocessing_decisions or self._default_decisions(
            feature_set.preprocessing_recommendations or {}
        )
        source_df = await self._load_feature_set_source_df(feature_set)
        from ml.transformers.column_role_detector import ColumnRoleDetector
        detector = ColumnRoleDetector()
        roles = detector.detect(source_df, column_mapping=(feature_set.config or {}).get("column_mapping"))
        engineered_features = self._build_engineered_feature_explanations(
            feature_set=feature_set,
            source_df=source_df,
            roles=roles,
        )
        return {
            "feature_set_id": str(feature_set.id),
            "plan_version": int(feature_set.preprocessing_plan_version or 0),
            "committed_at": feature_set.preprocessing_committed_at.isoformat() + "Z"
            if feature_set.preprocessing_committed_at
            else None,
            "committed_by": feature_set.preprocessing_committed_by,
            "profile": feature_set.preprocessing_profile or [],
            "recommendations": feature_set.preprocessing_recommendations or {},
            "decisions": decisions,
            "source_column_count": len(source_df.columns),
            "engineered_feature_count": len(engineered_features),
            "engineered_features": engineered_features,
        }

    async def update_preprocessing_decisions(
        self,
        feature_set_id: str,
        updates: List[Dict[str, Any]],
        actor_id: str,
        expected_plan_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set:
            raise ValueError("Feature set not found")

        current_version = int(feature_set.preprocessing_plan_version or 0)
        if expected_plan_version is not None and expected_plan_version != current_version:
            raise RuntimeError("Plan version mismatch. Refresh and retry your changes.")

        if not feature_set.preprocessing_recommendations:
            await self.get_preprocessing_plan(feature_set_id)
            await self.db.refresh(feature_set)

        recommendations = feature_set.preprocessing_recommendations or {}
        # IMPORTANT: JSON columns are plain dict/list; mutate on copies so SQLAlchemy
        # reliably detects changes and persists updates.
        decisions = copy.deepcopy(
            feature_set.preprocessing_decisions or self._default_decisions(recommendations)
        )
        audit = copy.deepcopy(feature_set.preprocessing_audit or [])

        for item in updates:
            feature_key = str(item.get("feature_key", "")).strip()
            if not feature_key or feature_key not in recommendations:
                raise ValueError(f"Unknown feature '{feature_key}' in decision update")

            prev = decisions.get(feature_key) or {}
            if prev.get("status") == "locked" and item.get("status") not in {"locked"}:
                raise ValueError(f"Feature '{feature_key}' is locked and cannot be modified")

            selected_method = item.get("selected_method", prev.get("selected_method"))
            if selected_method:
                allowed_methods = {recommendations[feature_key].get("method")}
                for alt in recommendations[feature_key].get("alternatives", []):
                    allowed_methods.add(alt.get("method"))
                allowed_methods.update(
                    {
                        "no_transform",
                        "mean_imputation",
                        "median_imputation",
                        "most_frequent_imputation",
                        "standard_scaling",
                        "minmax_scaling",
                        "robust_scaling",
                        "normalization",
                        "one_hot_encoding",
                        "label_encoding",
                        "ordinal_encoding",
                        "target_encoding",
                        "outlier_capping",
                        "log_transformation",
                        "date_feature_extraction",
                        "text_vectorization",
                        "feature_selection",
                        "duplicate_removal",
                        "data_type_correction",
                        "do_not_transform",
                    }
                )
                if selected_method not in allowed_methods:
                    raise ValueError(
                        f"Method '{selected_method}' is not allowed for feature '{feature_key}'"
                    )

            status = item.get("status", prev.get("status", "pending"))
            if status not in {"pending", "accepted", "rejected", "overridden", "locked", "do_not_transform"}:
                raise ValueError(f"Invalid status '{status}' for feature '{feature_key}'")

            confidence = float(recommendations[feature_key].get("confidence", 0.0))
            reason = str(item.get("change_reason") or "").strip()
            if status == "overridden" and confidence >= 0.8 and not reason:
                raise ValueError(
                    f"Feature '{feature_key}' requires change_reason when overriding a high-confidence recommendation"
                )

            new_decision = {
                "status": status,
                "selected_method": selected_method or recommendations[feature_key].get("method"),
                "selected_params": item.get("selected_params", prev.get("selected_params", {})),
                "change_reason": reason or prev.get("change_reason"),
                "updated_by": actor_id,
                "updated_at": now_ist().isoformat(),
                "mode": item.get("mode", "manual"),
            }
            decisions[feature_key] = new_decision

            audit.append(
                {
                    "action": "decision_update",
                    "feature_key": feature_key,
                    "old_value": prev,
                    "new_value": new_decision,
                    "changed_by": actor_id,
                    "changed_at": now_ist().isoformat(),
                    "plan_version": current_version,
                    "change_reason": reason or None,
                }
            )

        feature_set.preprocessing_decisions = copy.deepcopy(decisions)
        feature_set.preprocessing_audit = copy.deepcopy(audit)
        await self.db.commit()
        await self.db.refresh(feature_set)
        return {
            "plan_version": int(feature_set.preprocessing_plan_version or 0),
            "decisions": feature_set.preprocessing_decisions or {},
        }

    async def commit_preprocessing_plan(self, feature_set_id: str, actor_id: str) -> Dict[str, Any]:
        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set:
            raise ValueError("Feature set not found")

        if not feature_set.preprocessing_recommendations:
            await self.get_preprocessing_plan(feature_set_id)
            await self.db.refresh(feature_set)

        feature_set.preprocessing_plan_version = int(feature_set.preprocessing_plan_version or 0) + 1
        feature_set.preprocessing_committed_at = now_ist()
        feature_set.preprocessing_committed_by = actor_id

        snapshot = {
            "feature_set_id": str(feature_set.id),
            "plan_version": int(feature_set.preprocessing_plan_version),
            "profile": feature_set.preprocessing_profile or [],
            "recommendations": feature_set.preprocessing_recommendations or {},
            "decisions": feature_set.preprocessing_decisions
            or self._default_decisions(feature_set.preprocessing_recommendations or {}),
            "committed_at": feature_set.preprocessing_committed_at.isoformat(),
            "committed_by": actor_id,
        }
        checksum = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()
        config = copy.deepcopy(feature_set.config or {})
        config["preprocessing_plan_snapshot"] = snapshot
        config["preprocessing_plan_checksum"] = checksum
        feature_set.config = copy.deepcopy(config)

        audit = copy.deepcopy(feature_set.preprocessing_audit or [])
        audit.append(
            {
                "action": "commit",
                "feature_key": None,
                "old_value": None,
                "new_value": {"plan_version": int(feature_set.preprocessing_plan_version), "checksum": checksum},
                "changed_by": actor_id,
                "changed_at": now_ist().isoformat(),
                "plan_version": int(feature_set.preprocessing_plan_version),
                "change_reason": "Plan committed",
            }
        )
        feature_set.preprocessing_audit = copy.deepcopy(audit)

        await self.db.commit()
        await self.db.refresh(feature_set)
        return {
            "feature_set_id": str(feature_set.id),
            "plan_version": int(feature_set.preprocessing_plan_version),
            "committed_at": feature_set.preprocessing_committed_at.isoformat() + "Z",
            "committed_by": actor_id,
            "snapshot_checksum": checksum,
        }

    async def get_preprocessing_audit(self, feature_set_id: str) -> Optional[List[Dict[str, Any]]]:
        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set:
            return None
        audit = feature_set.preprocessing_audit or []
        return sorted(
            audit,
            key=lambda item: item.get("changed_at") or "",
            reverse=True,
        )
    
    async def create_feature_set(
        self,
        dataset_id: str,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
    ) -> FeatureSet:
        """Create a new feature set and trigger computation."""
        # Verify dataset exists
        dataset = await self.db.execute(
            select(Dataset).where(Dataset.id == UUID(dataset_id))
        )
        if not dataset.scalar_one_or_none():
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Compute config hash for deduplication
        config_hash = self._compute_config_hash(dataset_id, config)
        
        # Check for existing COMPLETED feature set with same hash (and not deleted)
        existing_result = await self.db.execute(
            select(FeatureSet).where(
                FeatureSet.config_hash == config_hash,
                FeatureSet.status == "COMPLETED",
                FeatureSet.is_deleted == False
            ).order_by(FeatureSet.created_at.desc())
        )
        existing_set = existing_result.scalars().first()
        
        if existing_set:
            logger.info(f"Reusing existing feature set {existing_set.id} for identical config")
            existing_set.reused = True  # strict typing might complain, but runtime is fine
            return existing_set

        # Create feature set record
        feature_set = FeatureSet(
            dataset_id=UUID(dataset_id),
            name=name,
            description=description,
            config=config,
            config_hash=config_hash,
            status="QUEUED",
        )
        
        self.db.add(feature_set)
        await self.db.commit()
        await self.db.refresh(feature_set)
        
        # Trigger async computation
        from app.workers.feature_worker import compute_features
        compute_features.delay(str(feature_set.id))
        
        logger.info(f"Created feature set {feature_set.id}, computation queued")
        
        feature_set.reused = False
        return feature_set

    def _compute_config_hash(self, dataset_id: str, config: Dict[str, Any]) -> str:
        """Generate a deterministic hash for deduplication."""
        # Sort keys to ensure consistent JSON string
        # Filter out random/volatile fields if any (none in current config spec)
        config_str = json.dumps(config, sort_keys=True)
        raw_str = f"{dataset_id}:{config_str}"
        return hashlib.sha256(raw_str.encode('utf-8')).hexdigest()
    
    async def update_feature_set_status(
        self,
        feature_set_id: str,
        status: str,
        progress: float = None,
        error_message: str = None,
        selected_features: List[str] = None,
        selection_report: Dict = None,
    ) -> bool:
        """Update feature set status and results."""
        try:
            uuid_id = UUID(feature_set_id)
        except ValueError:
            return False
        
        update_data = {"status": status}
        if selected_features:
            update_data["selected_features"] = selected_features
            update_data["selected_feature_count"] = len(selected_features)
        if selection_report:
            update_data["selection_report"] = selection_report
        if error_message:
            update_data["error_message"] = error_message
        if status == "COMPLETED":
            from datetime import datetime
            update_data["completed_at"] = now_ist()
        
        await self.db.execute(
            update(FeatureSet)
            .where(FeatureSet.id == uuid_id)
            .values(**update_data)
        )
        await self.db.commit()
        return True
    
    async def delete_feature_set(self, feature_set_id: str) -> bool:
        """Delete a feature set and its blob artifacts from Azure Storage."""
        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set:
            return False
        
        # Soft delete
        feature_set.is_deleted = True
        feature_set.deleted_at = now_ist()
        
        # We KEEP blob artifacts for now to allow potential restore or audit.
        # If hard cleanup is needed, a separate 'prune_deleted' task can be made.
        
        await self.db.commit()
        logger.info(f"Soft deleted feature set {feature_set_id}")
        return True
    
    async def analyze_feature_set(self, feature_set_id: str) -> bool:
        """Trigger async analysis of a feature set."""
        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set:
            return False
            
        from app.workers.feature_worker import analyze_features
        analyze_features.delay(feature_set_id)
        return True

    async def preview_features(self, feature_set_id: str, limit: int = 10) -> Dict[str, Any]:
        """Preview feature data from parquet file."""
        import pandas as pd
        import io
        feature_set, blob_data, resolved_storage_path = await self._download_feature_set_blob(feature_set_id)

        try:
            df = pd.read_parquet(io.BytesIO(blob_data))
        except Exception as exc:
            logger.exception("Feature preview download/parse failed for %s", resolved_storage_path)
            raise ValueError(f"Feature preview load failed: {exc}")
        
        # Return first N rows
        preview_df = df.head(limit)
        
        # Replace NaN with null for JSON serialization
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        
        return {
            "columns": list(preview_df.columns),
            "rows": preview_df.to_dict(orient="records"),
            "total_rows": len(df),
            "shape": df.shape,
            "dataset_version": feature_set.version
        }

    async def download_feature_dataset(self, feature_set_id: str, output_format: str = "parquet") -> Tuple[bytes, str, str]:
        """
        Download full engineered dataset file for a feature set.

        Returns:
            tuple(bytes_content, filename, content_type)
        """
        import pandas as pd
        feature_set, blob_data, resolved_storage_path = await self._download_feature_set_blob(feature_set_id)

        normalized_format = (output_format or "parquet").strip().lower()
        if normalized_format not in {"parquet", "csv", "xlsx"}:
            raise ValueError("Unsupported download format. Allowed values: parquet, csv, xlsx")

        blob_name = resolved_storage_path.split("/", 1)[1]
        source_extension = "parquet"
        if "." in blob_name:
            source_extension = blob_name.rsplit(".", 1)[1].lower()

        output_bytes = blob_data
        if normalized_format != source_extension:
            if source_extension != "parquet":
                raise ValueError(
                    f"Conversion from '{source_extension}' is not supported for this feature set; download as '{source_extension}'"
                )

            import io
            try:
                df = pd.read_parquet(io.BytesIO(blob_data))
            except Exception as exc:
                raise ValueError(f"Failed to load feature dataset for conversion: {exc}")

            def _convert_bytes() -> bytes:
                if normalized_format == "csv":
                    return df.to_csv(index=False).encode("utf-8")
                if normalized_format == "xlsx":
                    xlsx_buffer = io.BytesIO()
                    try:
                        # xlsxwriter is faster than openpyxl for large write-only exports.
                        with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
                            df.to_excel(writer, index=False, sheet_name="training_dataset")
                    except Exception:
                        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                            df.to_excel(writer, index=False, sheet_name="training_dataset")
                    return xlsx_buffer.getvalue()
                return blob_data

            try:
                output_bytes = await asyncio.to_thread(_convert_bytes)
            except Exception as exc:
                raise ValueError(f"{normalized_format.upper()} conversion failed: {exc}")

        filename = f"training_dataset_{feature_set.id}.{normalized_format}"
        content_type = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "parquet": "application/octet-stream",
        }[normalized_format]
        return output_bytes, filename, content_type

    async def _download_feature_set_blob(self, feature_set_id: str) -> Tuple[FeatureSet, bytes, str]:
        """Resolve storage path variants and download feature-set blob."""
        from app.core.config import get_settings
        from app.core.storage import storage_service

        feature_set = await self.get_feature_set(feature_set_id)
        if not feature_set or not feature_set.storage_path:
            raise ValueError(f"Feature set {feature_set_id} not found or missing storage path")

        settings = get_settings()
        raw_path = str(feature_set.storage_path).strip()
        features_container = settings.AZURE_STORAGE_CONTAINER_FEATURES
        datasets_container = settings.AZURE_STORAGE_CONTAINER_DATASETS

        # Resolve historical storage path variants:
        # - "features/train/<id>/features.parquet" (blob-only path under features container)
        # - "<container>/<blob_path>" (full storage path)
        candidates = [f"{features_container}/{raw_path}"]
        if "/" in raw_path:
            prefix, remainder = raw_path.split("/", 1)
            if prefix in (features_container, datasets_container):
                candidates.append(raw_path)
                candidates.append(f"{prefix}/{remainder}")
                candidates.append(f"{features_container}/{raw_path}")
                candidates.append(f"{features_container}/{remainder}")

        seen = set()
        candidate_paths = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                candidate_paths.append(c)

        blob_data = None
        resolved_storage_path = None
        local_storage_root = Path(__file__).resolve().parents[2]
        normalized_raw_path = raw_path.replace("\\", "/")
        local_candidates = []
        for candidate in (
            local_storage_root / normalized_raw_path,
            Path.cwd() / normalized_raw_path,
            local_storage_root / "storage" / normalized_raw_path.replace("storage/", "", 1),
        ):
            if candidate.exists() and candidate.is_file():
                local_candidates.append(str(candidate))

        for local_candidate in local_candidates:
            try:
                blob_data = Path(local_candidate).read_bytes()
                resolved_storage_path = local_candidate
                break
            except OSError:
                continue

        if blob_data is None:
            for storage_path in candidate_paths:
                try:
                    blob_data = await storage_service.download_dataset(storage_path)
                    resolved_storage_path = storage_path
                    break
                except FileNotFoundError:
                    continue

        if blob_data is None or resolved_storage_path is None:
            raise ValueError(
                "Feature dataset source not found in storage: "
                f"{raw_path}. Tried: {', '.join(candidate_paths)}"
            )

        return feature_set, blob_data, resolved_storage_path

    async def _generate_preprocessing_plan(
        self, feature_set: FeatureSet
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        df = await self._load_feature_set_source_df(feature_set)

        from ml.transformers.column_role_detector import ColumnRoleDetector

        detector = ColumnRoleDetector()
        roles = detector.detect(df, column_mapping=(feature_set.config or {}).get("column_mapping"))

        profile: List[Dict[str, Any]] = []
        for col in df.columns:
            series = df[col]
            missing_pct = float(series.isna().mean() * 100.0)
            unique_count = int(series.nunique(dropna=True))
            dtype_name = str(series.dtype)
            detected_type = self._detect_feature_type(col, series, roles)

            issues: List[str] = []
            if missing_pct > 0:
                issues.append("missing_values")
            if unique_count <= 1:
                issues.append("constant_feature")

            outlier_pct = 0.0
            skewness = 0.0
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                skewness = float(numeric.skew(skipna=True) or 0.0)
                q1 = float(numeric.quantile(0.25))
                q3 = float(numeric.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - (1.5 * iqr)
                    upper = q3 + (1.5 * iqr)
                    outlier_pct = float(((numeric < lower) | (numeric > upper)).mean() * 100.0)
                    if outlier_pct >= 5.0:
                        issues.append("outliers")
                if abs(skewness) >= 1.0:
                    issues.append("skewed_distribution")
            else:
                if detected_type in {"categorical", "text"} and unique_count > 50:
                    issues.append("high_cardinality")
                if detected_type == "text":
                    issues.append("unstructured_text")

            if dtype_name == "object":
                numeric_like = pd.to_numeric(series, errors="coerce")
                numeric_like_ratio = float(numeric_like.notna().mean())
                if numeric_like_ratio >= 0.8:
                    issues.append("data_type_mismatch")

            profile.append(
                {
                    "feature_key": col,
                    "detected_type": detected_type,
                    "dtype": dtype_name,
                    "missing_pct": round(missing_pct, 2),
                    "unique_count": unique_count,
                    "outlier_pct": round(outlier_pct, 2),
                    "skewness": round(skewness, 4),
                    "issues": sorted(set(issues)),
                }
            )

        recommendations = self._build_recommendations(profile)
        return profile, recommendations

    async def _load_feature_set_source_df(self, feature_set: FeatureSet) -> pd.DataFrame:
        dataset_res = await self.db.execute(select(Dataset).where(Dataset.id == feature_set.dataset_id))
        dataset = dataset_res.scalar_one_or_none()
        if not dataset:
            raise ValueError(f"Dataset {feature_set.dataset_id} not found for feature profiling")

        blob = await storage_service.download_dataset(dataset.storage_path)
        if dataset.file_format == "parquet":
            return pd.read_parquet(io.BytesIO(blob))
        if dataset.file_format == "json":
            return pd.read_json(io.BytesIO(blob))
        try:
            return pd.read_csv(io.BytesIO(blob))
        except UnicodeDecodeError:
            try:
                return pd.read_csv(io.BytesIO(blob), encoding="cp1252")
            except UnicodeDecodeError:
                return pd.read_csv(io.BytesIO(blob), encoding="ISO-8859-1")

    @staticmethod
    def _detect_feature_type(
        col: str, series: pd.Series, roles
    ) -> str:
        if col == roles.target_col:
            return "target"
        if col in roles.id_cols:
            return "id"
        if col == roles.timestamp_col:
            return "datetime"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        avg_len = float(series.dropna().astype(str).str.len().mean() or 0.0)
        if avg_len > 25:
            return "text"
        return "categorical"

    def _build_recommendations(self, profile: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        recs: Dict[str, Dict[str, Any]] = {}
        for item in profile:
            feature = item["feature_key"]
            detected_type = item["detected_type"]
            issues = set(item.get("issues", []))
            missing_pct = float(item.get("missing_pct", 0.0))
            outlier_pct = float(item.get("outlier_pct", 0.0))
            skewness = abs(float(item.get("skewness", 0.0)))
            unique_count = int(item.get("unique_count", 0))

            method = "no_transform"
            reason = "No strong preprocessing signal detected; keeping feature as-is."
            assumptions: List[str] = []
            expected_impact: List[str] = ["maintains current feature semantics"]
            alternatives: List[Dict[str, Any]] = []
            confidence = 0.62

            if "constant_feature" in issues:
                method = "feature_selection"
                reason = "Feature has near-zero variance and is unlikely to improve model learning."
                confidence = 0.94
                expected_impact = ["reduces noise", "improves training stability"]
                alternatives = [{"method": "do_not_transform", "tradeoff": "keeps feature for manual review"}]
            elif detected_type == "numeric":
                if "missing_values" in issues:
                    if outlier_pct >= 5:
                        method = "median_imputation"
                        reason = "Missing values detected with outliers present; median is robust to extreme values."
                        confidence = 0.9
                        alternatives = [
                            {"method": "mean_imputation", "tradeoff": "faster but sensitive to outliers"},
                            {"method": "most_frequent_imputation", "tradeoff": "can distort numeric distributions"},
                        ]
                    else:
                        method = "mean_imputation"
                        reason = "Missing values detected with low outlier influence; mean preserves central tendency."
                        confidence = 0.81
                        alternatives = [
                            {"method": "median_imputation", "tradeoff": "more robust but may underrepresent mean behavior"}
                        ]
                    expected_impact = ["prevents row drops", "improves model readiness"]
                elif "outliers" in issues:
                    method = "robust_scaling"
                    reason = "Outliers detected; robust scaling reduces the influence of extreme values."
                    confidence = 0.86
                    alternatives = [
                        {"method": "outlier_capping", "tradeoff": "caps extremes but changes value distribution"},
                        {"method": "standard_scaling", "tradeoff": "sensitive to outliers"},
                    ]
                    expected_impact = ["improves convergence for scale-sensitive models", "stabilizes gradients"]
                elif "skewed_distribution" in issues and skewness >= 1.0:
                    method = "log_transformation"
                    reason = "Strong skew detected; log transform can normalize distribution shape."
                    confidence = 0.8
                    alternatives = [
                        {"method": "robust_scaling", "tradeoff": "reduces outlier effect but may not fix skew"}
                    ]
                    expected_impact = ["improves linear model behavior", "reduces variance dominance"]
                else:
                    method = "standard_scaling"
                    reason = "Numeric feature is suitable for scale normalization for model-stable training."
                    confidence = 0.72
                    alternatives = [
                        {"method": "minmax_scaling", "tradeoff": "bounded range but sensitive to outliers"},
                        {"method": "do_not_transform", "tradeoff": "no scaling for tree-only workflows"},
                    ]
                    expected_impact = ["faster optimization for linear/svm/nn models"]
            elif detected_type == "categorical":
                if "high_cardinality" in issues or unique_count > 50:
                    method = "target_encoding"
                    reason = "High-cardinality categorical feature benefits from compact supervised encoding."
                    confidence = 0.82
                    alternatives = [
                        {"method": "label_encoding", "tradeoff": "compact but introduces arbitrary order"},
                        {"method": "one_hot_encoding", "tradeoff": "can cause high dimensionality"},
                    ]
                    expected_impact = ["controls feature explosion", "retains predictive signal"]
                else:
                    method = "one_hot_encoding"
                    reason = "Low-cardinality categorical feature is best represented via one-hot encoding."
                    confidence = 0.89
                    alternatives = [
                        {"method": "ordinal_encoding", "tradeoff": "only appropriate for ordered categories"},
                        {"method": "label_encoding", "tradeoff": "may imply false ordinal relation"},
                    ]
                    expected_impact = ["improves model interpretability", "avoids ordinal bias"]
            elif detected_type == "datetime":
                method = "date_feature_extraction"
                reason = "Datetime feature likely contains periodic signals such as hour/day/week patterns."
                confidence = 0.93
                alternatives = [
                    {"method": "do_not_transform", "tradeoff": "raw datetime often unusable for most estimators"}
                ]
                expected_impact = ["adds temporal signal", "improves seasonality capture"]
            elif detected_type == "text":
                method = "text_vectorization"
                reason = "Unstructured text must be transformed into numeric vectors for model consumption."
                confidence = 0.9
                alternatives = [
                    {"method": "do_not_transform", "tradeoff": "text cannot be used directly by tabular estimators"}
                ]
                expected_impact = ["enables text signal in training"]
            elif detected_type in {"id", "target"}:
                method = "do_not_transform"
                reason = "Identifier/target fields should not be transformed by default preprocessing."
                confidence = 0.95
                expected_impact = ["prevents leakage risk"]

            if missing_pct > 30:
                assumptions.append("high missingness may require business-level review before production use")

            recs[feature] = {
                "feature_key": feature,
                "method": method,
                "params": {},
                "reason": reason,
                "assumptions": assumptions,
                "confidence": round(min(max(confidence, 0.01), 0.99), 2),
                "alternatives": alternatives,
                "expected_impact": expected_impact,
                "mode": "auto",
            }
        return recs

    def _default_decisions(self, recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        now = now_ist().isoformat()
        return {
            feature: {
                "status": "pending",
                "selected_method": rec.get("method"),
                "selected_params": rec.get("params", {}),
                "change_reason": None,
                "updated_by": "system",
                "updated_at": now,
                "mode": "auto",
            }
            for feature, rec in recommendations.items()
        }

    def _build_engineered_feature_explanations(
        self,
        feature_set: FeatureSet,
        source_df: pd.DataFrame,
        roles,
    ) -> List[Dict[str, Any]]:
        feature_names = list(feature_set.all_features or feature_set.selected_features or [])
        if not feature_names:
            report = feature_set.selection_report or {}
            feature_names = list(report.get("feature_names") or [])

        source_columns = list(source_df.columns)
        selection_scores = ((feature_set.selection_report or {}).get("scores") or {})
        removed = ((feature_set.selection_report or {}).get("removed") or {})
        removed_variance = set(removed.get("variance_filter") or [])
        removed_correlation = set(removed.get("correlation_filter") or [])
        target_col = getattr(roles, "target_col", None)

        explanations: List[Dict[str, Any]] = []
        for feature_name in feature_names:
            if feature_name == target_col:
                explanations.append(
                    {
                        "feature_name": feature_name,
                        "source_columns": [feature_name],
                        "category": "target",
                        "transformation": "target_passthrough",
                        "why": "Target column is attached for supervised training and is not an engineered predictor.",
                        "training_status": "target",
                    }
                )
                continue

            explanation = self._describe_engineered_feature(
                feature_name=feature_name,
                source_columns=source_columns,
                roles=roles,
            )

            if feature_name in removed_variance:
                training_status = "removed_variance"
            elif feature_name in removed_correlation:
                training_status = "removed_correlation"
            elif feature_name in selection_scores:
                training_status = str(selection_scores[feature_name].get("recommendation", "available"))
            else:
                training_status = "available"

            explanation["training_status"] = training_status
            explanations.append(explanation)

        return explanations

    def _describe_engineered_feature(
        self,
        feature_name: str,
        source_columns: List[str],
        roles,
    ) -> Dict[str, Any]:
        amount_col = getattr(roles, "amount_col", None)
        user_col = getattr(roles, "user_col", None)
        timestamp_col = getattr(roles, "timestamp_col", None)

        def numeric_source_prefix(name: str) -> Optional[str]:
            matches = sorted(
                [col for col in source_columns if name.startswith(f"{col}_")],
                key=len,
                reverse=True,
            )
            return matches[0] if matches else None

        source_cols: List[str] = []
        category = "general"
        transformation = "derived_feature"
        why = "Derived by the feature engineering pipeline to make the raw data more useful for model training."

        if feature_name == "amount_log":
            source_cols = [amount_col] if amount_col else []
            category = "transaction"
            transformation = "log_transform"
            why = "Compresses large transaction amounts so extreme values do not dominate the model."
        elif feature_name == "amount_sqrt":
            source_cols = [amount_col] if amount_col else []
            category = "transaction"
            transformation = "sqrt_transform"
            why = "Softens the effect of high transaction amounts while preserving order."
        elif feature_name == "amount_zscore":
            source_cols = [amount_col] if amount_col else []
            category = "transaction"
            transformation = "zscore_standardization"
            why = "Shows how far an amount is from the typical transaction amount in standard deviation units."
        elif feature_name == "is_round_amount":
            source_cols = [amount_col] if amount_col else []
            category = "transaction"
            transformation = "rule_based_flag"
            why = "Flags rounded amounts that can sometimes indicate synthetic or suspicious payment behavior."
        elif feature_name == "is_high_value":
            source_cols = [amount_col] if amount_col else []
            category = "transaction"
            transformation = "quantile_threshold_flag"
            why = "Marks transactions above the learned high-value threshold to highlight unusual spending."
        elif feature_name == "amount_cents":
            source_cols = [amount_col] if amount_col else []
            category = "transaction"
            transformation = "fractional_component_extraction"
            why = "Captures the decimal pattern of an amount, which can sometimes separate human and synthetic behavior."
        elif feature_name in {
            "user_avg_amount", "user_std_amount", "user_txn_count", "user_total_amount",
            "user_max_amount", "amount_vs_user_avg", "amount_vs_user_std", "is_user_max"
        }:
            source_cols = [col for col in [user_col, amount_col] if col]
            category = "behavioral"
            transformation = "user_history_aggregation"
            why = "Uses the user’s historical payment behavior to compare the current transaction with past patterns."
        elif feature_name in {
            "hour_of_day", "day_of_week", "day_of_month", "is_weekend",
            "is_night", "is_business_hours", "hour_sin", "hour_cos",
            "time_since_last_txn", "is_rapid_transaction"
        }:
            source_cols = [col for col in [timestamp_col, user_col] if col]
            category = "temporal"
            transformation = "datetime_feature_extraction"
            why = "Extracts timing and sequence signals that help the model detect unusual transaction rhythms."
        elif feature_name.startswith("velocity_") or feature_name.startswith("amount_sum_"):
            source_cols = [col for col in [user_col, timestamp_col, amount_col] if col]
            category = "aggregation"
            transformation = "rolling_window_aggregation"
            why = "Summarizes recent user activity over time windows to capture velocity and accumulation patterns."
        elif feature_name.endswith("_encoded"):
            base = numeric_source_prefix(feature_name[:-8] + "_encoded") or feature_name[:-8]
            source_cols = [base] if base else []
            category = "categorical"
            transformation = "ordinal_encoding"
            why = "Converts a categorical value into a stable numeric code so the model can consume it."
        elif feature_name.endswith("_fraud_rate"):
            base = numeric_source_prefix(feature_name[:-11] + "_fraud_rate") or feature_name[:-11]
            source_cols = [base] if base else []
            category = "categorical"
            transformation = "target_rate_encoding"
            why = "Replaces categories with learned fraud-rate signal to preserve predictive information compactly."
        elif feature_name.endswith("_log"):
            base = numeric_source_prefix(feature_name)
            source_cols = [base] if base else []
            category = "numeric"
            transformation = "log_transform"
            why = "Reduces skew and the impact of extreme numeric values."
        elif feature_name.endswith("_zscore"):
            base = numeric_source_prefix(feature_name)
            source_cols = [base] if base else []
            category = "numeric"
            transformation = "zscore_standardization"
            why = "Normalizes a numeric column relative to its training mean and standard deviation."
        else:
            matched_source = numeric_source_prefix(feature_name)
            source_cols = [matched_source] if matched_source else []

        return {
            "feature_name": feature_name,
            "source_columns": source_cols,
            "category": category,
            "transformation": transformation,
            "why": why,
        }
    
    
    async def get_default_config(self) -> dict:
        """Get default feature engineering configuration."""
        return {
        "column_mapping": None,  # Auto-detect column roles if not provided
        "transaction_features": True,
        "behavioral_features": True,
        "temporal_features": True,
        "aggregation_features": True,
        "aggregation_windows": ["1h", "24h", "7d"],
        "enable_feature_selection": True,
        "max_features": 30,
        "variance_threshold": 0.01,
        "correlation_threshold": 0.95,
    }
    
    # ===== Synchronous methods for Celery workers =====
    
    def get_feature_set_sync(self, feature_set_id: str) -> Optional[FeatureSet]:
        """Get a single feature set by ID (sync version for Celery)."""
        try:
            uuid_id = UUID(feature_set_id)
        except ValueError:
            return None
        
        result = self.db.execute(
            select(FeatureSet).where(FeatureSet.id == uuid_id)
        )
        return result.scalar_one_or_none()
    
    def update_feature_set_status_sync(
        self,
        feature_set_id: str,
        status: str,
        progress: float = None,
        error_message: str = None,
        selected_features: List[str] = None,
        selection_report: Dict = None,
        storage_path: str = None,
        feature_count: int = None,
        all_features: List[str] = None,
        input_rows: int = None,
        processing_time_seconds: int = None,
    ) -> bool:
        """Update feature set status and results (sync version for Celery)."""
        try:
            uuid_id = UUID(feature_set_id)
        except ValueError:
            return False
        
        update_data = {"status": status}
        if selected_features:
            update_data["selected_features"] = selected_features
            update_data["selected_feature_count"] = len(selected_features)
        if selection_report:
            update_data["selection_report"] = selection_report
        if error_message:
            update_data["error_message"] = error_message
        if storage_path:
            update_data["storage_path"] = storage_path
        if feature_count is not None:
            update_data["feature_count"] = feature_count
        if all_features is not None:
            update_data["all_features"] = all_features
        if input_rows is not None:
            update_data["input_rows"] = input_rows
        if processing_time_seconds is not None:
            update_data["processing_time_seconds"] = processing_time_seconds
        if status == "COMPLETED":
            from datetime import datetime
            update_data["completed_at"] = now_ist()
        
        self.db.execute(
            update(FeatureSet)
            .where(FeatureSet.id == uuid_id)
            .values(**update_data)
        )
        return True

