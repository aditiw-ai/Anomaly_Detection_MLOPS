"""
Enterprise dataset artifact naming framework.

Generates traceable, sortable, cloud-safe artifact IDs and display names for:
- Raw uploads
- Merged datasets
- Split (train/test/validation)
- Feature engineering outputs

Format: {stage}_{timestamp_iso}_{short_id}[_{suffix}]
- Timestamp: YYYYMMDDTHHMMSSZ (UTC, sortable)
- short_id: 8-char alphanumeric for collision avoidance
- Character set: [a-z0-9_] only (Azure Blob / S3 / GCP safe)
"""
from app.core.time import IST, now_ist
import re
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


# Stage tokens for pipeline identification
class ArtifactStage:
    RAW = "raw"
    MERGE = "merge"
    SPLIT = "split"
    FEAT = "feat"


# Split type suffix
class SplitSuffix:
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


def _utc_timestamp() -> str:
    """Return current UTC time as YYYYMMDDTHHMMSSZ (sortable, unambiguous)."""
    return datetime.now(IST).strftime("%Y%m%dT%H%M%SZ")


def _short_id(length: int = 8) -> str:
    """Return a short alphanumeric id (lowercase hex from uuid4)."""
    return uuid4().hex[:length].lower()


def _sanitize_slug(value: str, max_length: int = 64) -> str:
    """Convert a string to a safe path segment: [a-z0-9_] only."""
    if not value:
        return ""
    # Lowercase, replace spaces and invalid chars with underscore
    s = value.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_length] if s else _short_id(8)


def generate_artifact_id(
    stage: str,
    suffix: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Generate an enterprise artifact ID.

    Args:
        stage: One of ArtifactStage.RAW, MERGE, SPLIT, FEAT
        suffix: Optional suffix (e.g. 'train', 'test', 'validation' for split)
        timestamp: Optional fixed timestamp (for determinism); default now UTC

    Returns:
        Artifact ID like raw_20260210T143022Z_a1b2c3d4 or
        split_20260210T144022Z_j9k0l1m2_train
    """
    ts = timestamp or _utc_timestamp()
    sid = _short_id()
    parts = [stage, ts, sid]
    if suffix:
        parts.append(_sanitize_slug(suffix, max_length=32))
    return "_".join(parts)


def generate_raw_artifact_id(timestamp: Optional[str] = None) -> str:
    """Artifact ID for a raw upload."""
    return generate_artifact_id(ArtifactStage.RAW, timestamp=timestamp)


def generate_merge_artifact_id(timestamp: Optional[str] = None) -> str:
    """Artifact ID for a merged dataset."""
    return generate_artifact_id(ArtifactStage.MERGE, timestamp=timestamp)


def generate_split_job_artifact_id(timestamp: Optional[str] = None) -> str:
    """Artifact ID for a split job (shared by train/test/validation)."""
    return generate_artifact_id(ArtifactStage.SPLIT, timestamp=timestamp)


def generate_split_artifact_id(
    split_job_id: str,
    split_type: str,
) -> str:
    """
    Full artifact ID for a single split (e.g. train or test).
    Uses the same base as the job so train/test are grouped.

    Args:
        split_job_id: From generate_split_job_artifact_id()
        split_type: 'train', 'test', or 'validation'
    """
    safe = _sanitize_slug(split_type, max_length=32)
    return f"{split_job_id}_{safe}" if safe else split_job_id


def generate_feat_artifact_id(timestamp: Optional[str] = None) -> str:
    """Artifact ID for a feature engineering output."""
    return generate_artifact_id(ArtifactStage.FEAT, timestamp=timestamp)


def display_name_merge(sources: Optional[list] = None) -> str:
    """Human-readable display name for a merged dataset (UTC date-time)."""
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    if sources:
        return f"Merge of {', '.join(sources[:3])}{'…' if len(sources) > 3 else ''}"
    return f"Merge {ts}"


def display_name_split(prefix: Optional[str] = None, split_type: str = "Split") -> str:
    """Human-readable display name for a split job or split dataset."""
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    if prefix:
        return f"{prefix} — {split_type} {ts}"
    return f"Split {ts}"


def display_name_feat(dataset_name: Optional[str] = None, version: str = "1.0") -> str:
    """Human-readable display name for a feature set."""
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    if dataset_name:
        return f"{dataset_name} — Features v{version}"
    return f"Features {ts}"


def is_legacy_storage_path(storage_path: str) -> bool:
    """
    Return True if this path uses the pre-framework naming (no stage_timestamp_ prefix).
    Used to keep backward compatibility when reading.
    """
    if not storage_path:
        return True
    # Path is container/blob_path; we care about blob_path
    parts = storage_path.split("/", 1)
    blob_path = parts[1] if len(parts) == 2 else storage_path
    # New format: raw/raw_... or merged/merge_... or processed/split_...
    for stage in (ArtifactStage.RAW, ArtifactStage.MERGE, ArtifactStage.SPLIT, ArtifactStage.FEAT):
        if f"{stage}/" in blob_path or blob_path.startswith(f"{stage}_"):
            # Could be new format (raw/raw_...) or legacy (raw/user_name/...)
            segments = blob_path.split("/")
            if len(segments) >= 2:
                first = segments[1] if segments[0] in ("raw", "merged", "processed", "features") else segments[0]
                # New: first segment after type looks like stage_YYYYMMDDTHHMMSSZ_hex
                if re.match(rf"^{stage}_\d{{8}}T\d{{6}}Z_[a-f0-9]{{8}}", first):
                    return False
    return True
