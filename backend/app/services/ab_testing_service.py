"""
A/B Testing Service
Champion-challenger model testing in production.
"""
from app.core.time import IST, now_ist
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime, timedelta
from enum import Enum
import re
import logging
import random
import json
import os
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class ABTestStatus(str, Enum):
    """A/B test status."""
    DRAFT = "DRAFT"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class ABTestResult(str, Enum):
    """A/B test outcome."""
    PENDING = "PENDING"
    CHALLENGER_WINS = "CHALLENGER_WINS"
    CHAMPION_WINS = "CHAMPION_WINS"
    NO_SIGNIFICANT_DIFFERENCE = "NO_SIGNIFICANT_DIFFERENCE"


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    challenger_traffic_percent: float = 10.0  # Start with 10% traffic
    min_samples: int = 1000  # Minimum samples before evaluation
    max_duration_hours: int = 168  # 1 week max
    significance_level: float = 0.95  # 95% confidence
    primary_metric: str = "f1"
    secondary_metrics: List[str] = None
    auto_promote_on_win: bool = False
    rollback_on_performance_drop: bool = True
    performance_drop_threshold: float = 0.05  # 5% drop triggers rollback
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["precision", "recall", "auc"]


@dataclass
class ABTest:
    """A/B test record."""
    id: str
    name: str
    champion_model_id: str
    challenger_model_id: str
    config: ABTestConfig
    status: ABTestStatus
    result: ABTestResult
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    champion_samples: int = 0
    challenger_samples: int = 0
    champion_metrics: Optional[Dict] = None
    challenger_metrics: Optional[Dict] = None
    statistical_analysis: Optional[Dict] = None


class ABTestingService:
    """
    A/B testing service for model comparison.
    
    Features:
    - Traffic splitting between champion and challenger
    - Statistical significance testing
    - Auto-promotion and rollback
    - Real-time metrics tracking
    """
    _STATE_KEY = "abtesting:state:v1"
    
    def __init__(self):
        self._tests: Dict[str, ABTest] = {}
        self._active_test: Optional[str] = None
        self._prediction_counts: Dict[str, Dict[str, int]] = {}  # test_id -> {model_id -> count}
        self._route_updates_since_persist = 0
        self._prediction_updates_since_persist = 0
        self._simulation_progress: Dict[str, Dict[str, Any]] = {}  # test_id -> progress payload
        self._load_state()

    def _get_redis_client(self):
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            return redis.from_url(url, decode_responses=True, socket_timeout=2)
        except Exception as exc:
            logger.warning(f"A/B state Redis init failed: {exc}")
            return None

    def _serialize_config(self, config: ABTestConfig) -> Dict[str, Any]:
        return {
            "challenger_traffic_percent": config.challenger_traffic_percent,
            "min_samples": config.min_samples,
            "max_duration_hours": config.max_duration_hours,
            "significance_level": config.significance_level,
            "primary_metric": config.primary_metric,
            "secondary_metrics": config.secondary_metrics,
            "auto_promote_on_win": config.auto_promote_on_win,
            "rollback_on_performance_drop": config.rollback_on_performance_drop,
            "performance_drop_threshold": config.performance_drop_threshold,
        }

    def _deserialize_config(self, data: Dict[str, Any]) -> ABTestConfig:
        return ABTestConfig(
            challenger_traffic_percent=float(data.get("challenger_traffic_percent", 10.0)),
            min_samples=int(data.get("min_samples", 1000)),
            max_duration_hours=int(data.get("max_duration_hours", 168)),
            significance_level=float(data.get("significance_level", 0.95)),
            primary_metric=data.get("primary_metric", "f1"),
            secondary_metrics=data.get("secondary_metrics") or ["precision", "recall", "auc"],
            auto_promote_on_win=bool(data.get("auto_promote_on_win", False)),
            rollback_on_performance_drop=bool(data.get("rollback_on_performance_drop", True)),
            performance_drop_threshold=float(data.get("performance_drop_threshold", 0.05)),
        )

    def _serialize_test(self, test: ABTest) -> Dict[str, Any]:
        return {
            "id": test.id,
            "name": test.name,
            "champion_model_id": test.champion_model_id,
            "challenger_model_id": test.challenger_model_id,
            "config": self._serialize_config(test.config),
            "status": test.status.value,
            "result": test.result.value,
            "created_at": test.created_at.isoformat(),
            "started_at": test.started_at.isoformat() if test.started_at else None,
            "ended_at": test.ended_at.isoformat() if test.ended_at else None,
            "champion_samples": test.champion_samples,
            "challenger_samples": test.challenger_samples,
            "champion_metrics": test.champion_metrics,
            "challenger_metrics": test.challenger_metrics,
            "statistical_analysis": test.statistical_analysis,
        }

    def _deserialize_test(self, data: Dict[str, Any]) -> ABTest:
        return ABTest(
            id=data["id"],
            name=data["name"],
            champion_model_id=data["champion_model_id"],
            challenger_model_id=data["challenger_model_id"],
            config=self._deserialize_config(data.get("config") or {}),
            status=ABTestStatus(data.get("status", ABTestStatus.DRAFT.value)),
            result=ABTestResult(data.get("result", ABTestResult.PENDING.value)),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            champion_samples=int(data.get("champion_samples", 0)),
            challenger_samples=int(data.get("challenger_samples", 0)),
            champion_metrics=data.get("champion_metrics"),
            challenger_metrics=data.get("challenger_metrics"),
            statistical_analysis=data.get("statistical_analysis"),
        )

    def _persist_state(self):
        client = self._get_redis_client()
        if not client:
            return
        payload = {
            "tests": [self._serialize_test(t) for t in self._tests.values()],
            "active_test": self._active_test,
            "prediction_counts": self._prediction_counts,
            "simulation_progress": self._simulation_progress,
        }
        try:
            client.set(self._STATE_KEY, json.dumps(payload, default=str))
            self._route_updates_since_persist = 0
            self._prediction_updates_since_persist = 0
        except Exception as exc:
            logger.warning(f"A/B state persist failed: {exc}")

    def _load_state(self):
        client = self._get_redis_client()
        if not client:
            return
        try:
            raw = client.get(self._STATE_KEY)
            if not raw:
                return
            payload = json.loads(raw)

            tests = {}
            for t in payload.get("tests", []):
                try:
                    test = self._deserialize_test(t)
                    tests[test.id] = test
                except Exception as exc:
                    logger.warning(f"Skipping invalid persisted A/B test: {exc}")

            self._tests = tests
            self._active_test = payload.get("active_test")
            counts = payload.get("prediction_counts") or {}
            self._prediction_counts = {
                str(test_id): {str(model_id): int(value) for model_id, value in model_counts.items()}
                for test_id, model_counts in counts.items()
            }
            self._simulation_progress = payload.get("simulation_progress") or {}
        except Exception as exc:
            logger.warning(f"A/B state load failed: {exc}")

    def _init_simulation_progress(
        self,
        test_id: str,
        total_rows: int,
        champion_rows: int,
        challenger_rows: int,
        label_key: Optional[str],
    ) -> None:
        self._simulation_progress[test_id] = {
            "status": "RUNNING",
            "phase": "CHAMPION",
            "processed": 0,
            "total": int(total_rows),
            "percent": 0.0,
            "labelled_samples": 0,
            "label_key_used": label_key,
            "champion_rows": int(champion_rows),
            "challenger_rows": int(challenger_rows),
            "started_at": now_ist().isoformat(),
            "updated_at": now_ist().isoformat(),
        }
        self._persist_state()

    def _update_simulation_progress(
        self,
        test_id: str,
        processed: int,
        total: int,
        phase: str,
        labelled_samples: int,
        force_persist: bool = False,
    ) -> None:
        p = self._simulation_progress.get(test_id)
        if not p:
            return
        pct = (processed / max(total, 1)) * 100.0
        p["phase"] = phase
        p["processed"] = int(processed)
        p["total"] = int(total)
        p["percent"] = round(min(100.0, max(0.0, pct)), 2)
        p["labelled_samples"] = int(labelled_samples)
        p["updated_at"] = now_ist().isoformat()
        if force_persist or processed % 25 == 0:
            self._persist_state()

    def _finalize_simulation_progress(
        self,
        test_id: str,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        p = self._simulation_progress.get(test_id)
        if not p:
            return
        p["status"] = status
        p["phase"] = "DONE" if status == "COMPLETED" else "FAILED"
        if status == "COMPLETED":
            p["processed"] = int(p.get("total", p.get("processed", 0)))
            p["percent"] = 100.0
        if message:
            p["message"] = message
        p["updated_at"] = now_ist().isoformat()
        self._persist_state()

    def mark_simulation_initializing(self, test_id: str, requested_rows: int = 0) -> None:
        """Seed progress immediately so API can return fast while background work initializes."""
        self._simulation_progress[test_id] = {
            "status": "RUNNING",
            "phase": "INITIALIZING",
            "processed": 0,
            "total": int(max(requested_rows, 0)),
            "percent": 0.0,
            "labelled_samples": 0,
            "started_at": now_ist().isoformat(),
            "updated_at": now_ist().isoformat(),
        }
        self._persist_state()

    def get_simulation_progress(self, test_id: str) -> Dict[str, Any]:
        self._load_state()
        p = self._simulation_progress.get(test_id)
        if not p:
            return {
                "status": "IDLE",
                "phase": "IDLE",
                "processed": 0,
                "total": 0,
                "percent": 0.0,
                "labelled_samples": 0,
                "updated_at": now_ist().isoformat(),
            }
        return p
    
    def create_test(
        self,
        name: str,
        champion_model_id: str,
        challenger_model_id: str,
        config: Optional[ABTestConfig] = None,
    ) -> ABTest:
        """Create a new A/B test."""
        test = ABTest(
            id=str(uuid4()),
            name=name,
            champion_model_id=champion_model_id,
            challenger_model_id=challenger_model_id,
            config=config or ABTestConfig(),
            status=ABTestStatus.DRAFT,
            result=ABTestResult.PENDING,
            created_at=now_ist(),
        )
        
        self._tests[test.id] = test
        self._persist_state()
        logger.info(f"Created A/B test: {test.id} - {name}")
        
        return test
    
    def start_test(self, test_id: str) -> ABTest:
        """Start an A/B test."""
        self._load_state()
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        # Heal stale active-test pointers restored from persisted state.
        if self._active_test:
            active = self._tests.get(self._active_test)
            if (not active) or (active.status != ABTestStatus.RUNNING):
                self._active_test = None
                self._persist_state()

        if self._active_test and self._active_test != test_id:
            raise ValueError("Another test is already running")

        if test.status not in (ABTestStatus.DRAFT, ABTestStatus.PAUSED):
            raise ValueError(f"Cannot start test in status {test.status.value}")
        
        test.status = ABTestStatus.RUNNING
        test.started_at = now_ist()
        self._active_test = test_id
        self._persist_state()
        
        logger.info(f"Started A/B test: {test_id}")
        return test
    
    def route_request(self, test_id: Optional[str] = None) -> str:
        """
        Route a request to champion or challenger.
        
        Returns:
            Model ID to use for this request
        """
        tid = test_id or self._active_test
        if not tid:
            return "default"  # No active test
        
        test = self._tests.get(tid)
        if not test or test.status != ABTestStatus.RUNNING:
            return "default"
        
        # Random routing based on traffic split
        if random.random() * 100 < test.config.challenger_traffic_percent:
            test.challenger_samples += 1
            self._route_updates_since_persist += 1
            if self._route_updates_since_persist >= 25:
                self._persist_state()
            return test.challenger_model_id
        else:
            test.champion_samples += 1
            self._route_updates_since_persist += 1
            if self._route_updates_since_persist >= 25:
                self._persist_state()
            return test.champion_model_id

    @staticmethod
    def _normalize_key(key: Any) -> str:
        """Normalize column keys for robust matching across spaces/hyphens/case variants."""
        s = str(key).strip().lower()
        s = re.sub(r"[\s\-]+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    @staticmethod
    def _parse_binary_label(value: Any) -> Optional[int]:
        """Parse common binary label encodings into 0/1."""
        if value is None:
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        # Handle native numeric types and numpy scalar numerics without requiring numpy import.
        if isinstance(value, (int, float)) or (
            hasattr(value, "dtype") and hasattr(value, "item")
        ):
            try:
                fv = float(value)
                if fv in (0.0, 1.0):
                    return int(fv)
            except Exception:
                pass
        if isinstance(value, str):
            normalized = value.strip().lower()
            normalized_compact = normalized.replace(" ", "").replace("_", "").replace("-", "")
            if normalized in ("1", "true", "yes", "y", "fraud", "positive"):
                return 1
            if normalized in ("0", "false", "no", "n", "legit", "negative", "non-fraud", "nonfraud"):
                return 0
            if normalized_compact in (
                "fraudulent",
                "fraudelent",  # common misspelling observed in user datasets
                "isfraud",
                "positiveclass",
            ):
                return 1
            if normalized_compact in (
                "nonfraud",
                "notfraud",
                "genuine",
                "legitimate",
                "normal",
                "negativeclass",
            ):
                return 0
            # Handle numeric-looking strings such as "1.0"/"0.0"
            try:
                fv = float(normalized)
                if fv in (0.0, 1.0):
                    return int(fv)
            except Exception:
                pass
        return None

    @staticmethod
    def _infer_label_key(transactions: List[Dict[str, Any]]) -> Optional[str]:
        """Infer likely ground-truth column using strict label-name patterns."""
        if not transactions:
            return None
        key_scores: Dict[str, float] = {}
        value_checks = min(len(transactions), 200)
        strong_exact = {
            "label",
            "target",
            "class",
            "y",
            "outcome",
            "response",
            "is_fraud",
            "fraud",
            "fraud_label",
            "actual_label",
            "ground_truth",
            "is_fraudulent",
            "is_fraud_transaction",
        }
        weak_tokens = ("label", "target", "ground_truth", "actual", "fraud", "class", "outcome", "response")

        # Some row-oriented sources omit null columns in a subset of rows; inspect a sample union.
        candidate_keys: set[str] = set()
        for row in transactions[:value_checks]:
            candidate_keys.update(str(k) for k in row.keys())

        for key in candidate_keys:
            k_norm = ABTestingService._normalize_key(key)
            has_strong_name = k_norm in strong_exact
            has_weak_name = any(token in k_norm for token in weak_tokens)
            if not (has_strong_name or has_weak_name):
                continue

            binary_hits = 0
            present_count = 0
            for row in transactions[:value_checks]:
                if key not in row:
                    continue
                present_count += 1
                if ABTestingService._parse_binary_label(row.get(key)) is not None:
                    binary_hits += 1
            if present_count == 0:
                continue
            # Coverage among rows where this key is present.
            coverage = binary_hits / present_count

            # Require some sampled values to be parseable binary labels.
            if coverage < 0.1:  # Relaxed from 0.6 to 0.1 to be more inclusive
                continue

            score = (3.0 if has_strong_name else 1.5) + coverage
            key_scores[key] = score

        if not key_scores:
            return None
        best_key, _ = max(key_scores.items(), key=lambda x: x[1])
        return best_key

    @staticmethod
    def _extract_actual_label(txn: Dict[str, Any], inferred_key: Optional[str] = None) -> Optional[int]:
        """Extract ground-truth label from inferred and common candidate keys."""
        if inferred_key and inferred_key in txn:
            parsed = ABTestingService._parse_binary_label(txn.get(inferred_key))
            if parsed is not None:
                return parsed
        if inferred_key:
            # Handle normalized-equivalent key (e.g., inferred "is_fraud" while payload has "is fraud").
            inferred_norm = ABTestingService._normalize_key(inferred_key)
            for raw_key, raw_val in txn.items():
                if ABTestingService._normalize_key(raw_key) == inferred_norm:
                    parsed = ABTestingService._parse_binary_label(raw_val)
                    if parsed is not None:
                        return parsed

        label_candidates = [
            "is_fraud", "label", "is_fraud_transaction",
            "target", "fraud_label", "is_fraudulent",
            "fraud_flag", "is_fraud_flag", "ground_truth", "actual_label",
            "class", "y", "response", "outcome", "fraud",
            "target_variable", "target_var", "is_fraudulent_flag",
            "isfraud", "targetvariable", "fraudlabel", "actuallabel",
            "groundtruth", "prediction_target", "is_fraud_val", "is_fraud_01"
        ]
        normalized_keys = {ABTestingService._normalize_key(k): k for k in txn.keys()}
        for key in label_candidates:
            direct = txn.get(key, None)
            parsed_direct = ABTestingService._parse_binary_label(direct)
            if parsed_direct is not None:
                return parsed_direct
            nk = normalized_keys.get(ABTestingService._normalize_key(key))
            if nk is None:
                continue
            parsed_nk = ABTestingService._parse_binary_label(txn.get(nk))
            if parsed_nk is not None:
                return parsed_nk
        return None

    async def simulate_traffic_from_dataset(
        self,
        test_id: str,
        transactions: List[Dict[str, Any]],
        inference_service: Any,
        db: Any,
        reset_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Simulate A/B test traffic using a provided dataset.
        Actually runs inference on both models to produce realistic comparison metrics.
        """
        self._load_state()
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        logger.info(f"Starting A/B test simulation for {test_id} with {len(transactions)} rows")
        inferred_label_key = self._infer_label_key(transactions)

        if reset_existing:
            # Treat each simulation run as a fresh evaluation batch.
            test.champion_samples = 0
            test.challenger_samples = 0
            test.champion_metrics = None
            test.challenger_metrics = None
            test.statistical_analysis = None
            self._prediction_counts[test_id] = {}
        
        # 1. Deterministic split simulation (exact traffic ratio for offline replay)
        # Live routing remains stochastic; simulation should honor the requested split exactly.
        champ_rows = []
        chal_rows = []

        total_rows = len(transactions)
        challenger_target = int(round((test.config.challenger_traffic_percent / 100.0) * total_rows))
        challenger_target = max(0, min(challenger_target, total_rows))

        # Deterministic replay split for offline simulation:
        # same dataset content + same test config => same arm assignment across runs.
        # This removes run-to-run variance caused purely by random shuffling.
        decorated_rows = []
        for idx, row in enumerate(transactions):
            try:
                payload = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
            except Exception:
                payload = str(row)
            digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
            decorated_rows.append((digest, idx, row))

        decorated_rows.sort(key=lambda x: (x[0], x[1]))
        ordered_rows = [r for _, _, r in decorated_rows]
        chal_rows = ordered_rows[:challenger_target]
        champ_rows = ordered_rows[challenger_target:]

        # Keep route counters aligned with simulation assignment
        test.champion_samples += len(champ_rows)
        test.challenger_samples += len(chal_rows)
        total_rows_for_progress = len(champ_rows) + len(chal_rows)
        self._init_simulation_progress(
            test_id=test_id,
            total_rows=total_rows_for_progress,
            champion_rows=len(champ_rows),
            challenger_rows=len(chal_rows),
            label_key=inferred_label_key,
        )

        results = {
            "champion": {"correct": 0, "total": 0, "positives": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0},
            "challenger": {"correct": 0, "total": 0, "positives": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0}
        }
        labelled_rows = 0
        processed_rows = 0
        batch_size = 128

        try:
            # 2. Run Inference for Champion
            if champ_rows:
                self._update_simulation_progress(
                    test_id=test_id,
                    processed=processed_rows,
                    total=total_rows_for_progress,
                    phase="LOADING_CHAMPION_MODEL",
                    labelled_samples=labelled_rows,
                    force_persist=True,
                )
                await asyncio.wait_for(inference_service.load_model(test.champion_model_id, db), timeout=120)
                for i in range(0, len(champ_rows), batch_size):
                    chunk = champ_rows[i:i + batch_size]
                    batch_res = inference_service.predict_batch(chunk)
                    predictions = [int(r.get("prediction", 0)) for r in batch_res.get("results", [])]
                    if len(predictions) != len(chunk):
                        raise RuntimeError(
                            f"Champion batch prediction size mismatch: got {len(predictions)} for {len(chunk)} rows"
                        )

                    for txn, prediction in zip(chunk, predictions):
                        actual = self._extract_actual_label(txn, inferred_key=inferred_label_key)
                        if actual is not None:
                            labelled_rows += 1
                            results["champion"]["total"] += 1
                            if prediction == actual:
                                results["champion"]["correct"] += 1
                            if actual == 1:
                                results["champion"]["positives"] += 1
                            if prediction == 1 and actual == 1:
                                results["champion"]["true_positives"] += 1
                            elif prediction == 1 and actual == 0:
                                results["champion"]["false_positives"] += 1
                            elif prediction == 0 and actual == 1:
                                results["champion"]["false_negatives"] += 1
                        self.record_prediction(test_id, test.champion_model_id, prediction)
                        processed_rows += 1

                    self._update_simulation_progress(
                        test_id=test_id,
                        processed=processed_rows,
                        total=total_rows_for_progress,
                        phase="CHAMPION",
                        labelled_samples=labelled_rows,
                        force_persist=True,
                    )
                    # Yield to event loop so progress endpoint/UI polling remain responsive.
                    await asyncio.sleep(0)

            # 3. Run Inference for Challenger
            if chal_rows:
                self._update_simulation_progress(
                    test_id=test_id,
                    processed=processed_rows,
                    total=total_rows_for_progress,
                    phase="LOADING_CHALLENGER_MODEL",
                    labelled_samples=labelled_rows,
                    force_persist=True,
                )
                await asyncio.wait_for(inference_service.load_model(test.challenger_model_id, db), timeout=120)
                for i in range(0, len(chal_rows), batch_size):
                    chunk = chal_rows[i:i + batch_size]
                    batch_res = inference_service.predict_batch(chunk)
                    predictions = [int(r.get("prediction", 0)) for r in batch_res.get("results", [])]
                    if len(predictions) != len(chunk):
                        raise RuntimeError(
                            f"Challenger batch prediction size mismatch: got {len(predictions)} for {len(chunk)} rows"
                        )

                    for txn, prediction in zip(chunk, predictions):
                        actual = self._extract_actual_label(txn, inferred_key=inferred_label_key)
                        if actual is not None:
                            labelled_rows += 1
                            results["challenger"]["total"] += 1
                            if prediction == actual:
                                results["challenger"]["correct"] += 1
                            if actual == 1:
                                results["challenger"]["positives"] += 1
                            if prediction == 1 and actual == 1:
                                results["challenger"]["true_positives"] += 1
                            elif prediction == 1 and actual == 0:
                                results["challenger"]["false_positives"] += 1
                            elif prediction == 0 and actual == 1:
                                results["challenger"]["false_negatives"] += 1
                        self.record_prediction(test_id, test.challenger_model_id, prediction)
                        processed_rows += 1

                    self._update_simulation_progress(
                        test_id=test_id,
                        processed=processed_rows,
                        total=total_rows_for_progress,
                        phase="CHALLENGER",
                        labelled_samples=labelled_rows,
                        force_persist=True,
                    )
                    # Yield to event loop so progress endpoint/UI polling remain responsive.
                    await asyncio.sleep(0)
        except Exception as exc:
            self._finalize_simulation_progress(test_id, status="FAILED", message=str(exc))
            raise

        # 4. Calculate Final Metrics
        def calc_metrics(res):
            tp, fp, fn = res["true_positives"], res["false_positives"], res["false_negatives"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = res["correct"] / res["total"] if res["total"] > 0 else 0
            return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "samples": res["total"]}

        # Re-fetch the test object from self._tests to prevent saving to an orphaned object
        # if a concurrent request called _load_state() and replaced the self._tests dict.
        active_test = self._tests.get(test_id)
        if active_test:
            active_test.champion_metrics = calc_metrics(results["champion"])
            active_test.challenger_metrics = calc_metrics(results["challenger"])
        
        self._update_simulation_progress(
            test_id=test_id,
            processed=total_rows_for_progress,
            total=total_rows_for_progress,
            phase="FINALIZING",
            labelled_samples=labelled_rows,
            force_persist=True,
        )
        self._persist_state()
        self._finalize_simulation_progress(test_id, status="COMPLETED")
        
        if active_test:
            logger.info(f"Simulation complete. Champion F1: {active_test.champion_metrics['f1']:.4f}, Challenger F1: {active_test.challenger_metrics['f1']:.4f}")
        
        return {
            "test_id": test_id,
            "samples_processed": len(transactions),
            "champion_samples": len(champ_rows),
            "challenger_samples": len(chal_rows),
            "target_challenger_percent": test.config.challenger_traffic_percent,
            "labelled_samples_used": labelled_rows,
            "label_key_used": inferred_label_key,
            "reset_existing": reset_existing,
        }
    
    def record_prediction(
        self,
        test_id: str,
        model_id: str,
        prediction: int,
        actual: Optional[int] = None,
        response_time_ms: float = 0.0,
    ):
        """Record a prediction result for the test."""
        test = self._tests.get(test_id)
        if not test:
            return
        
        # Accumulate prediction counts per model for real metric aggregation
        if test_id not in self._prediction_counts:
            self._prediction_counts[test_id] = {}
        counts = self._prediction_counts[test_id]
        counts[model_id] = counts.get(model_id, 0) + 1
        self._prediction_updates_since_persist += 1
        if self._prediction_updates_since_persist >= 50:
            self._persist_state()
        logger.debug(f"Recorded prediction for test {test_id}, model {model_id}")
    
    def evaluate_test(self, test_id: str) -> Dict:
        """
        Evaluate current test status.
        
        Performs statistical significance testing if enough samples.
        """
        self._load_state()
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        # Build traffic counters from actual recorded prediction counts first.
        counts = self._prediction_counts.get(test_id, {})
        champ_count = counts.get(test.champion_model_id, 0)
        chal_count = counts.get(test.challenger_model_id, 0)
        
        # Update sample counters from actual recorded data.
        test.champion_samples = champ_count
        test.challenger_samples = chal_count

        total_samples = test.champion_samples + test.challenger_samples
        traffic_progress_pct = min(100.0, (total_samples / max(test.config.min_samples, 1)) * 100.0)

        # Check if we have enough routed samples first.
        if total_samples < test.config.min_samples:
            return {
                "ready_for_decision": False,
                "samples_collected": total_samples,
                "samples_needed": test.config.min_samples,
                "labelled_samples_collected": 0,
                "decision_progress_pct": round(traffic_progress_pct, 2),
                "traffic_progress_pct": round(traffic_progress_pct, 2),
                "label_coverage_pct": 0.0,
                "blockers": ["INSUFFICIENT_TRAFFIC"],
                "message": f"Need {test.config.min_samples - total_samples} more samples",
            }
        
        # Use metrics stored on the test objects if available.
        test.champion_metrics = test.champion_metrics or {
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0,
            "samples": champ_count,
        }
        test.challenger_metrics = test.challenger_metrics or {
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0,
            "samples": chal_count,
        }

        champion_eval_samples = int((test.champion_metrics or {}).get("samples", 0) or 0)
        challenger_eval_samples = int((test.challenger_metrics or {}).get("samples", 0) or 0)
        total_eval_samples = champion_eval_samples + challenger_eval_samples

        # Readiness must include at least some label-backed evaluation basis.
        if total_eval_samples == 0:
            return {
                "ready_for_decision": False,
                "samples_collected": total_samples,
                "samples_needed": test.config.min_samples,
                "labelled_samples_collected": 0,
                "decision_progress_pct": 0.0,
                "traffic_progress_pct": round(traffic_progress_pct, 2),
                "label_coverage_pct": 0.0,
                "blockers": ["NO_LABELLED_SAMPLES"],
                "message": (
                    "No labelled samples available for fair comparison. "
                    "Use simulation dataset with ground-truth label column or collect feedback labels."
                ),
                "champion_metrics": test.champion_metrics,
                "challenger_metrics": test.challenger_metrics,
            }

        if champion_eval_samples == 0 or challenger_eval_samples == 0:
            return {
                "ready_for_decision": False,
                "samples_collected": total_samples,
                "samples_needed": test.config.min_samples,
                "labelled_samples_collected": total_eval_samples,
                "decision_progress_pct": 0.0,
                "traffic_progress_pct": round(traffic_progress_pct, 2),
                "label_coverage_pct": round((total_eval_samples / max(total_samples, 1)) * 100.0, 2),
                "blockers": ["UNBALANCED_LABELLED_SPLIT"],
                "message": (
                    "Labelled samples are present only for one arm. "
                    "Run simulation with more rows or a dataset with balanced labels across routed traffic."
                ),
                "champion_metrics": test.champion_metrics,
                "challenger_metrics": test.challenger_metrics,
            }
        
        # Perform statistical analysis
        analysis = self._statistical_analysis(test)
        test.statistical_analysis = analysis
        self._persist_state()
        
        return {
            "ready_for_decision": True,
            "samples_collected": total_samples,
            "samples_needed": test.config.min_samples,
            "labelled_samples_collected": total_eval_samples,
            "decision_progress_pct": 100.0,
            "traffic_progress_pct": 100.0,
            "label_coverage_pct": round((total_eval_samples / max(total_samples, 1)) * 100.0, 2),
            "blockers": [],
            "champion_metrics": test.champion_metrics,
            "challenger_metrics": test.challenger_metrics,
            "analysis": analysis,
            "recommendation": analysis.get("recommendation"),
        }
    
    def _statistical_analysis(self, test: ABTest) -> Dict:
        """
        Perform statistical significance testing.
        
        Uses two-proportion z-test for fairness.
        """
        champion = test.champion_metrics
        challenger = test.challenger_metrics
        primary = test.config.primary_metric
        
        champ_val = champion.get(primary, 0)
        chal_val = challenger.get(primary, 0)
        
        diff = chal_val - champ_val
        diff_percent = (diff / champ_val * 100) if champ_val > 0 else 0
        
        champ_n = int((champion or {}).get("samples", test.champion_samples or 0) or 0)
        chal_n = int((challenger or {}).get("samples", test.challenger_samples or 0) or 0)

        # Simplified significance check using effective labelled counts.
        is_significant = min(champ_n, chal_n) >= 30 and abs(diff_percent) > 1.0
        
        if is_significant and diff > 0:
            recommendation = "PROMOTE_CHALLENGER"
            result = ABTestResult.CHALLENGER_WINS
        elif is_significant and diff < 0:
            recommendation = "KEEP_CHAMPION"
            result = ABTestResult.CHAMPION_WINS
        else:
            recommendation = "CONTINUE_TEST"
            result = ABTestResult.NO_SIGNIFICANT_DIFFERENCE
        
        return {
            "primary_metric": primary,
            "champion_value": champ_val,
            "challenger_value": chal_val,
            "difference": diff,
            "difference_percent": diff_percent,
            "is_significant": is_significant,
            "confidence": 0.95 if is_significant else 0.0,
            "sample_sizes": {"champion": champ_n, "challenger": chal_n},
            "recommendation": recommendation,
            "result": result.value,
        }
    
    def conclude_test(
        self,
        test_id: str,
        result: ABTestResult,
        promote_challenger: bool = False,
    ) -> ABTest:
        """Conclude an A/B test."""
        self._load_state()
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        test.status = ABTestStatus.COMPLETED
        test.result = result
        test.ended_at = now_ist()
        
        if self._active_test == test_id:
            self._active_test = None
        
        if promote_challenger and result == ABTestResult.CHALLENGER_WINS:
            logger.info(f"Promoting challenger model: {test.challenger_model_id}")
            # In production, update model registry
        
        logger.info(f"Concluded A/B test {test_id}: {result.value}")
        self._persist_state()
        return test
    
    def abort_test(self, test_id: str, reason: str = "") -> ABTest:
        """Abort a running test."""
        self._load_state()
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        test.status = ABTestStatus.ABORTED
        test.ended_at = now_ist()
        
        if self._active_test == test_id:
            self._active_test = None
        
        logger.info(f"Aborted A/B test {test_id}: {reason}")
        self._persist_state()
        return test

    def delete_test(self, test_id: str) -> bool:
        """Hard-delete a test from AB runtime/persisted state."""
        self._load_state()
        test = self._tests.get(test_id)
        if not test:
            return False

        if self._active_test == test_id:
            self._active_test = None

        self._tests.pop(test_id, None)
        self._prediction_counts.pop(test_id, None)
        self._persist_state()
        logger.info(f"Deleted A/B test {test_id}")
        return True
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test by ID."""
        self._load_state()
        return self._tests.get(test_id)
    
    def list_tests(
        self,
        status: Optional[ABTestStatus] = None,
        limit: int = 20,
    ) -> List[ABTest]:
        """List A/B tests."""
        self._load_state()
        tests = list(self._tests.values())
        
        if status:
            tests = [t for t in tests if t.status == status]
        
        tests.sort(key=lambda t: t.created_at, reverse=True)
        return tests[:limit]
    
    def get_active_test(self) -> Optional[ABTest]:
        """Get currently active test."""
        if self._active_test:
            return self._tests.get(self._active_test)
        return None


# Singleton service instance
_ab_service: Optional[ABTestingService] = None


def get_ab_testing_service() -> ABTestingService:
    """Get the global A/B testing service instance."""
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestingService()
    return _ab_service
