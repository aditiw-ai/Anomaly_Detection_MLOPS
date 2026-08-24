"""
Alert Service
Alert management and notification.
"""
from app.core.time import IST, now_ist
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum
import logging

import os
import httpx
from sqlalchemy import select, update, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.alert import Alert as AlertModel, AlertType, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

@dataclass
class AlertCreate:
    """Data for creating an alert."""
    model_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    details: Optional[Dict] = None


@dataclass
class Alert:
    """Alert representation."""
    id: str
    model_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    details: Optional[Dict]
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None


class AlertService:
    """
    Service for alert management.
    
    Features:
    - Create alerts from monitoring
    - Acknowledge and resolve alerts
    - Send notifications
    - Alert aggregation and deduplication
    """
    
    # Alert deduplication window
    DEDUP_WINDOW_HOURS = 1
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def _auto_resolve_informational_training_alerts(self) -> None:
        """
        Informational TRAINING alerts are notifications only.
        They should not require acknowledgement workflow.
        """
        result = await self.db.execute(
            update(AlertModel)
            .where(
                AlertModel.alert_type == AlertType.TRAINING,
                AlertModel.severity == AlertSeverity.INFO,
                AlertModel.status.in_([AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]),
            )
            .values(
                status=AlertStatus.RESOLVED,
                resolved_at=now_ist(),
                resolved_notes="Auto-resolved informational training completion notification.",
            )
        )
        if (result.rowcount or 0) > 0:
            await self.db.commit()
       
    async def create_alert(self, data: AlertCreate) -> Alert:
        """
        Create a new alert.
        
        Applies deduplication - if similar alert exists within window,
        updates existing instead of creating new.
        """
        # Check for duplicate
        existing = await self._find_duplicate(data)
        if existing:
            logger.info(f"Deduplicating alert: {data.title}")
            return existing
        
        # Create new alert
        alert = AlertModel(
            id=str(uuid4()),
            model_id=data.model_id,
            alert_type=data.alert_type,
            severity=data.severity,
            status=AlertStatus.ACTIVE,
            title=data.title,
            message=data.message,
            details=data.details,
            created_at=now_ist(),
        )
        
        self.db.add(alert)
        await self.db.commit()
        await self.db.refresh(alert)
        
        # Send notification
        await self._send_notification(alert)
        
        logger.info(f"Created alert: {alert.id} - {alert.title}")
        return alert
    
    async def _find_duplicate(self, data: AlertCreate) -> Optional[AlertModel]:
        """Find duplicate alert within dedup window using DB query."""
        cutoff = now_ist() - timedelta(hours=self.DEDUP_WINDOW_HOURS)
        
        query = select(AlertModel).where(
            AlertModel.model_id == data.model_id,
            AlertModel.alert_type == data.alert_type,
            AlertModel.title == data.title,
            AlertModel.status == AlertStatus.ACTIVE,
            AlertModel.created_at > cutoff
        ).limit(1)
        
        result = await self.db.execute(query)
        return result.scalars().first()
    
    async def _send_notification(self, alert: AlertModel):
        """Send rich Block Kit Slack notification for an alert."""
        if alert.severity == AlertSeverity.CRITICAL:
            logger.warning(f"[CRITICAL ALERT] {alert.title}: {alert.message}")
        else:
            logger.info(f"[{alert.severity.value}] {alert.title}")

        from app.core.config import settings
        webhook_url = settings.SLACK_WEBHOOK_URL or ""
        if not webhook_url:
            return

        color_map = {
            AlertSeverity.CRITICAL: "#E53935",
            AlertSeverity.WARNING: "#FB8C00",
            AlertSeverity.INFO: "#1E88E5",
        }
        severity_emoji = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.WARNING: "🟡",
            AlertSeverity.INFO: "🔵",
        }
        type_emoji = {
            "DRIFT": "📊",
            "BIAS": "⚖️",
            "PERFORMANCE": "📉",
            "SYSTEM": "⚙️",
            "TRAINING": "🎓",
        }

        sev_label  = alert.severity.value
        type_label = alert.alert_type.value if hasattr(alert.alert_type, "value") else str(alert.alert_type)
        sev_icon   = severity_emoji.get(alert.severity, "⚠️")
        type_icon  = type_emoji.get(type_label, "🔔")
        color      = color_map.get(alert.severity, "#CCCCCC")

        # Build detail fields from alert.details dict
        detail_fields = []
        if alert.details and isinstance(alert.details, dict):
            for k, v in alert.details.items():
                detail_fields.append({
                    "title": k.replace("_", " ").title(),
                    "value": f"`{round(v, 4) if isinstance(v, float) else v}`",
                    "short": True,
                })

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{sev_icon} {type_icon} {alert.title}",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Severity:*\n{sev_icon} {sev_label}"},
                                {"type": "mrkdwn", "text": f"*Alert Type:*\n{type_icon} {type_label}"},
                                {"type": "mrkdwn", "text": f"*Model ID:*\n`{alert.model_id}`"},
                                {"type": "mrkdwn", "text": f"*Status:*\n🟠 ACTIVE"},
                            ],
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*Details:*\n{alert.message}"},
                        },
                    ]
                    + (
                        [
                            {
                                "type": "section",
                                "fields": [
                                    {"type": "mrkdwn", "text": f"*{f['title']}:*\n{f['value']}"}
                                    for f in detail_fields
                                ],
                            }
                        ]
                        if detail_fields
                        else []
                    )
                    + [
                        {"type": "divider"},
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"🏦 *Fraud Detection MLOps Platform* | {now_ist().strftime('%Y-%m-%d %H:%M UTC')}",
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(webhook_url, json=payload)
        except Exception as e:
            logger.error(f"Failed to push alert to Slack webhook: {e}")
    
    async def list_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        model_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[AlertModel]:
        """List alerts with DB filtering."""
        await self._auto_resolve_informational_training_alerts()
        query = select(AlertModel)
        
        if status:
            query = query.where(AlertModel.status == status)
        if severity:
            query = query.where(AlertModel.severity == severity)
        if model_id:
            query = query.where(AlertModel.model_id == model_id)
            
        query = query.order_by(AlertModel.created_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_alert(self, alert_id: str) -> Optional[AlertModel]:
        """Get alert by ID from DB."""
        query = select(AlertModel).where(AlertModel.id == alert_id)
        result = await self.db.execute(query)
        return result.scalars().first()
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        note: Optional[str] = None,
    ) -> Optional[AlertModel]:
        """Acknowledge an alert in DB."""
        alert = await self.get_alert(alert_id)
        if not alert:
            return None

        # TRAINING INFO alerts are informational only and should not be acknowledged.
        if alert.alert_type == AlertType.TRAINING and alert.severity == AlertSeverity.INFO:
            alert.status = AlertStatus.RESOLVED
            if not alert.resolved_at:
                alert.resolved_at = now_ist()
            if not alert.resolved_notes:
                alert.resolved_notes = "Auto-resolved informational training completion notification."
            await self.db.commit()
            await self.db.refresh(alert)
            return alert
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = now_ist()
        alert.acknowledged_by = user_id
        if note:
            alert.resolved_notes = note
            
        await self.db.commit()
        await self.db.refresh(alert)
        
        logger.info(f"Alert {alert_id} acknowledged by {user_id}")
        return alert
    
    async def resolve_alert(
        self,
        alert_id: str,
        note: Optional[str] = None,
    ) -> Optional[AlertModel]:
        """Resolve an alert in DB."""
        alert = await self.get_alert(alert_id)
        if not alert:
            return None
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = now_ist()
        alert.resolution_note = note
        
        await self.db.commit()
        await self.db.refresh(alert)
        
        logger.info(f"Alert {alert_id} resolved: {note}")
        return alert
    
    async def get_alert_summary(
        self,
        model_id: Optional[str] = None,
    ) -> Dict:
        """Get alert statistics summary directly from DB aggregations."""
        await self._auto_resolve_informational_training_alerts()
        
        # Base query for all aggregations
        base_where = [AlertModel.model_id == model_id] if model_id else []
        
        def build_count_query(field=None, value=None):
            q = select(func.count(AlertModel.id))
            filters = list(base_where)
            if field is not None and value is not None:
                filters.append(field == value)
            if filters:
                q = q.where(*filters)
            return q

        # Execute all counts concurrently
        total = await self.db.scalar(build_count_query())
        active = await self.db.scalar(build_count_query(AlertModel.status, AlertStatus.ACTIVE))
        ack = await self.db.scalar(build_count_query(AlertModel.status, AlertStatus.ACKNOWLEDGED))
        res = await self.db.scalar(build_count_query(AlertModel.status, AlertStatus.RESOLVED))
        
        crit = await self.db.scalar(build_count_query(AlertModel.severity, AlertSeverity.CRITICAL))
        warn = await self.db.scalar(build_count_query(AlertModel.severity, AlertSeverity.WARNING))
        info = await self.db.scalar(build_count_query(AlertModel.severity, AlertSeverity.INFO))
        
        drift = await self.db.scalar(build_count_query(AlertModel.alert_type, AlertType.DRIFT))
        perf = await self.db.scalar(build_count_query(AlertModel.alert_type, AlertType.PERFORMANCE))
        bias = await self.db.scalar(build_count_query(AlertModel.alert_type, AlertType.BIAS))
        
        return {
            "total": total or 0,
            "active": active or 0,
            "acknowledged": ack or 0,
            "resolved": res or 0,
            "by_severity": {"critical": crit or 0, "warning": warn or 0, "info": info or 0},
            "by_type": {"drift": drift or 0, "performance": perf or 0, "bias": bias or 0},
        }
    
    async def create_training_success_alert(
        self,
        model_id: str,
        job_id: str,
        algorithm: str,
        duration: float,
        metrics: Dict[str, float],
        event_time: Optional[datetime] = None,
    ) -> AlertModel:
        """Create a training success notification."""
        f1 = metrics.get("f1", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        
        alert = await self.create_alert(AlertCreate(
            model_id=model_id,
            alert_type=AlertType.TRAINING,
            severity=AlertSeverity.INFO,
            title=f"Model Training Complete: {algorithm.upper()}",
            message=(
                f"Model successfully trained in {duration:.1f} seconds. "
                f"Performance: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}"
            ),
            details={
                "job_id": job_id,
                "algorithm": algorithm,
                "duration_seconds": duration,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            },
        ))

        # Preserve original workflow event time when supplied (e.g., job completed_at),
        # instead of the current insertion time used by backfills/replays.
        if event_time:
            # DB uses naive IST timestamps.
            if event_time.tzinfo is not None:
                event_time = event_time.astimezone(IST).replace(tzinfo=None)
            alert.created_at = event_time

        # Informational training alerts should be closed by default.
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = now_ist()
        if not alert.resolved_notes:
            alert.resolved_notes = "Informational notification; no acknowledgement required."
        await self.db.commit()
        await self.db.refresh(alert)
        return alert


    async def create_drift_alert(
        self,
        model_id: str,
        feature: str,
        psi: float,
        threshold: float,
    ) -> AlertModel:
        """Create a drift-specific alert."""
        severity = AlertSeverity.CRITICAL if psi > 0.25 else AlertSeverity.WARNING
        
        return await self.create_alert(AlertCreate(
            model_id=model_id,
            alert_type=AlertType.DRIFT,
            severity=severity,
            title=f"Data Drift Detected: {feature}",
            message=f"PSI={psi:.3f} exceeds threshold {threshold}",
            details={
                "feature": feature,
                "psi": psi,
                "threshold": threshold,
            },
        ))
    
    async def create_performance_alert(
        self,
        model_id: str,
        metric: str,
        current_value: float,
        baseline_value: float,
    ) -> AlertModel:
        """Create a performance degradation alert."""
        drop = baseline_value - current_value
        drop_pct = (drop / baseline_value) * 100 if baseline_value > 0 else 0
        
        severity = AlertSeverity.CRITICAL if drop_pct > 10 else AlertSeverity.WARNING
        
        return await self.create_alert(AlertCreate(
            model_id=model_id,
            alert_type=AlertType.PERFORMANCE,
            severity=severity,
            title=f"Performance Degradation: {metric}",
            message=f"{metric} dropped from {baseline_value:.3f} to {current_value:.3f} ({drop_pct:.1f}%)",
            details={
                "metric": metric,
                "current": current_value,
                "baseline": baseline_value,
                "drop_percent": drop_pct,
            },
        ))
