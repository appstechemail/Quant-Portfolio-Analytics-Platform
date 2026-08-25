"""
============================================================
MONITORING
============================================================

Institutional Monitoring Layer

Responsibilities
----------------

Audit Trail
Configuration Tracking
Model Lineage
Runtime Monitoring
Health Checks
Compliance Monitoring
Alerting
Operational Diagnostics

This module NEVER

• modifies portfolios
• modifies attribution
• modifies stress results
• modifies forecasts

It only observes and reports.

============================================================
"""

from __future__ import annotations

# ============================================================
# STANDARD LIBRARY
# ============================================================

from dataclasses import (
    dataclass,
    field,
)

from datetime import (
    datetime,
    timezone,
)

from enum import (
    Enum,
)

from typing import (
    Any,
)

import uuid

# ============================================================
# THIRD PARTY
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# GLOBAL CONSTANTS
# ============================================================

UTC = timezone.utc

DEFAULT_MONITORING_VERSION = "1.0.0"

DEFAULT_HEALTH_THRESHOLD = 0.95

DEFAULT_ALERT_THRESHOLD = 0.90


# ============================================================
# ENUMS
# ============================================================

class MonitoringStatus(
    str,
    Enum,
):
    """
    Monitoring result status.
    """

    PASSED = "PASSED"

    WARNING = "WARNING"

    FAILED = "FAILED"


class MonitoringSeverity(
    str,
    Enum,
):
    """
    Monitoring severity.
    """

    LOW = "LOW"

    MEDIUM = "MEDIUM"

    HIGH = "HIGH"

    CRITICAL = "CRITICAL"


class MonitoringCategory(
    str,
    Enum,
):
    """
    Monitoring categories.
    """

    AUDIT = "AUDIT"

    CONFIGURATION = "CONFIGURATION"

    LINEAGE = "LINEAGE"

    RUNTIME = "RUNTIME"

    HEALTH = "HEALTH"

    COMPLIANCE = "COMPLIANCE"

    ALERTING = "ALERTING"


class AlertLevel(
    str,
    Enum,
):
    """
    Alert level.
    """

    INFO = "INFO"

    WARNING = "WARNING"

    ERROR = "ERROR"

    CRITICAL = "CRITICAL"


# ============================================================
# METADATA
# ============================================================

@dataclass(slots=True)
class MonitoringMetadata:
    """
    Metadata for monitoring runs.
    """

    run_id: str

    timestamp: datetime

    platform_name: str

    environment: str

    version: str = (
        DEFAULT_MONITORING_VERSION
    )

    owner: str = ""

    tags: dict[str, str] = field(
        default_factory=dict
    )

    # --------------------------------------------------------

    @staticmethod
    def create(
        *,
        platform_name: str,
        environment: str,
        owner: str = "",
    ) -> "MonitoringMetadata":

        return MonitoringMetadata(

            run_id=
            str(
                uuid.uuid4()
            ),

            timestamp=
            datetime.now(
                UTC
            ),

            platform_name=
            platform_name,

            environment=
            environment,

            owner=
            owner,
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(slots=True)
class MonitoringConfig:
    """
    Monitoring configuration.
    """

    health_threshold: float = (
        DEFAULT_HEALTH_THRESHOLD
    )

    alert_threshold: float = (
        DEFAULT_ALERT_THRESHOLD
    )

    enable_audit: bool = True

    enable_lineage: bool = True

    enable_runtime: bool = True

    enable_compliance: bool = True

    enable_alerting: bool = True

    retain_history_days: int = 365

    diagnostics_depth: int = 5


# ============================================================
# BASE OBJECT
# ============================================================

@dataclass(slots=True)
class MonitoringObject:
    """
    Base monitoring object.
    """

    metadata: MonitoringMetadata

    category: MonitoringCategory


# ============================================================
# BASE RESULT
# ============================================================

@dataclass(slots=True)
class MonitoringResult:
    """
    Base monitoring result.
    """

    metadata: MonitoringMetadata

    category: MonitoringCategory

    status: MonitoringStatus

    severity: MonitoringSeverity

    score: float

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    runtime_seconds: float = 0.0


# ============================================================
# BASE ENGINE
# ============================================================

class BaseMonitoringEngine:
    """
    Base class for all monitoring engines.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = (
            config
            if config is not None
            else MonitoringConfig()
        )

    # --------------------------------------------------------

    @staticmethod
    def current_time() -> datetime:

        return datetime.now(
            UTC
        )

    # --------------------------------------------------------

    @staticmethod
    def clamp_score(
        value: float,
    ) -> float:

        return float(
            max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            )
        )

    # --------------------------------------------------------

    def determine_status(
        self,
        score: float,
    ) -> MonitoringStatus:

        if score >= (
            self.config
            .health_threshold
        ):

            return (
                MonitoringStatus
                .PASSED
            )

        if score >= (
            self.config
            .alert_threshold
        ):

            return (
                MonitoringStatus
                .WARNING
            )

        return (
            MonitoringStatus
            .FAILED
        )

    # --------------------------------------------------------

    @staticmethod
    def determine_severity(
        score: float,
    ) -> MonitoringSeverity:

        if score >= 0.95:

            return (
                MonitoringSeverity
                .LOW
            )

        if score >= 0.85:

            return (
                MonitoringSeverity
                .MEDIUM
            )

        if score >= 0.70:

            return (
                MonitoringSeverity
                .HIGH
            )

        return (
            MonitoringSeverity
            .CRITICAL
        )
    
# ============================================================
# PART 2 — MONITORING RESULT OBJECTS
# ============================================================

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
)

# ============================================================
# AUDIT RECORD
# ============================================================

@dataclass(slots=True)
class AuditRecord:
    """
    Immutable audit event.
    """

    event_id: str

    event_timestamp: datetime

    event_name: str

    category: MonitoringCategory

    status: MonitoringStatus

    source_system: str

    user_id: str | None = None

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CONFIGURATION SNAPSHOT
# ============================================================

@dataclass(slots=True)
class ConfigurationSnapshot:
    """
    Configuration image captured
    at run time.
    """

    snapshot_id: str

    created_at: datetime

    configuration: dict[str, Any]

    checksum: str

    version: str


# ============================================================
# MODEL LINEAGE RECORD
# ============================================================

@dataclass(slots=True)
class ModelLineageRecord:
    """
    Model lineage metadata.
    """

    model_id: str

    model_name: str

    model_version: str

    training_timestamp: datetime

    feature_count: int

    training_dataset_id: str

    checksum: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# RUNTIME METRICS
# ============================================================

@dataclass(slots=True)
class RuntimeMetrics:
    """
    Runtime monitoring metrics.
    """

    runtime_seconds: float

    cpu_utilization: float

    memory_utilization: float

    peak_memory_mb: float

    io_operations: int

    thread_count: int

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HEALTH CHECK RESULT
# ============================================================

@dataclass(slots=True)
class HealthCheckResult:
    """
    Health-check result.
    """

    component_name: str

    score: float

    status: MonitoringStatus

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )


# ============================================================
# COMPLIANCE RESULT
# ============================================================

@dataclass(slots=True)
class ComplianceCheckResult:
    """
    Compliance monitoring result.
    """

    rule_name: str

    passed: bool

    severity: MonitoringSeverity

    message: str

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# ALERT RECORD
# ============================================================

@dataclass(slots=True)
class AlertRecord:
    """
    Monitoring alert.
    """

    alert_id: str

    created_at: datetime

    level: AlertLevel

    title: str

    message: str

    source_component: str

    acknowledged: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# AUDIT TRAIL RESULT
# ============================================================

@dataclass(slots=True)
class AuditTrailResult(
    MonitoringResult
):
    """
    Audit-trail monitoring result.
    """

    audit_records: list[AuditRecord] = field(
        default_factory=list
    )

    total_events: int = 0


# ============================================================
# CONFIGURATION MONITOR RESULT
# ============================================================

@dataclass(slots=True)
class ConfigurationMonitoringResult(
    MonitoringResult
):
    """
    Configuration monitoring result.
    """

    snapshot: ConfigurationSnapshot | None = None

    configuration_changes: int = 0


# ============================================================
# LINEAGE RESULT
# ============================================================

@dataclass(slots=True)
class ModelLineageMonitoringResult(
    MonitoringResult
):
    """
    Model-lineage result.
    """

    lineage_records: list[
        ModelLineageRecord
    ] = field(
        default_factory=list
    )

    model_count: int = 0


# ============================================================
# RUNTIME MONITOR RESULT
# ============================================================

@dataclass(slots=True)
class RuntimeMonitoringResult(
    MonitoringResult
):
    """
    Runtime monitoring result.
    """

    runtime_metrics: RuntimeMetrics | None = None


# ============================================================
# HEALTH MONITOR RESULT
# ============================================================

@dataclass(slots=True)
class HealthMonitoringResult(
    MonitoringResult
):
    """
    Health monitoring result.
    """

    health_checks: list[
        HealthCheckResult
    ] = field(
        default_factory=list
    )

    average_health_score: float = 0.0


# ============================================================
# COMPLIANCE MONITOR RESULT
# ============================================================

@dataclass(slots=True)
class ComplianceMonitoringResult(
    MonitoringResult
):
    """
    Compliance monitoring result.
    """

    compliance_checks: list[
        ComplianceCheckResult
    ] = field(
        default_factory=list
    )

    violations: int = 0


# ============================================================
# ALERTING RESULT
# ============================================================

@dataclass(slots=True)
class AlertMonitoringResult(
    MonitoringResult
):
    """
    Alerting result.
    """

    alerts: list[
        AlertRecord
    ] = field(
        default_factory=list
    )

    critical_alerts: int = 0


# ============================================================
# DIAGNOSTICS RESULT
# ============================================================

@dataclass(slots=True)
class MonitoringDiagnosticsResult(
    MonitoringResult
):
    """
    Monitoring diagnostics.
    """

    diagnostic_summary: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER MONITORING RESULT
# ============================================================

@dataclass(slots=True)
class InstitutionalMonitoringResult:
    """
    Aggregated monitoring result.
    """

    metadata: MonitoringMetadata

    audit_result: (
        AuditTrailResult
        | None
    ) = None

    configuration_result: (
        ConfigurationMonitoringResult
        | None
    ) = None

    lineage_result: (
        ModelLineageMonitoringResult
        | None
    ) = None

    runtime_result: (
        RuntimeMonitoringResult
        | None
    ) = None

    health_result: (
        HealthMonitoringResult
        | None
    ) = None

    compliance_result: (
        ComplianceMonitoringResult
        | None
    ) = None

    alert_result: (
        AlertMonitoringResult
        | None
    ) = None

    diagnostics_result: (
        MonitoringDiagnosticsResult
        | None
    ) = None


# ============================================================
# PART 3 — AUDIT TRAIL ENGINE
# ============================================================

import hashlib
import json

from typing import Any


# ============================================================
# AUDIT EVENT BUILDER
# ============================================================

class AuditEventBuilder:
    """
    Utility builder for audit events.
    """

    # --------------------------------------------------------

    @staticmethod
    def build_event(
        *,
        event_name: str,

        category: MonitoringCategory,

        status: MonitoringStatus,

        source_system: str,

        details: dict[str, Any]
        | None = None,

        user_id: str | None = None,
    ) -> AuditRecord:

        return AuditRecord(

            event_id=
            str(
                uuid.uuid4()
            ),

            event_timestamp=
            datetime.now(
                UTC
            ),

            event_name=
            event_name,

            category=
            category,

            status=
            status,

            source_system=
            source_system,

            user_id=
            user_id,

            details=
            details
            if details is not None
            else {},
        )


# ============================================================
# AUDIT STORE
# ============================================================

class AuditStore:
    """
    In-memory audit repository.

    Can later be replaced by:

        SQL
        Kafka
        Snowflake
        EventHub
        Elastic

    without changing engine code.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._records: list[
            AuditRecord
        ] = []

    # --------------------------------------------------------

    def add(
        self,
        record: AuditRecord,
    ) -> None:

        self._records.append(
            record
        )

    # --------------------------------------------------------

    def extend(
        self,
        records: list[
            AuditRecord
        ],
    ) -> None:

        self._records.extend(
            records
        )

    # --------------------------------------------------------

    def records(
        self,
    ) -> list[
        AuditRecord
    ]:

        return list(
            self._records
        )

    # --------------------------------------------------------

    def clear(
        self,
    ) -> None:

        self._records.clear()


# ============================================================
# AUDIT CHECKSUM ENGINE
# ============================================================

class AuditChecksumEngine:
    """
    Creates deterministic
    audit checksums.
    """

    # --------------------------------------------------------

    @staticmethod
    def checksum(
        payload: dict[str, Any],
    ) -> str:

        try:

            encoded = json.dumps(

                payload,

                sort_keys=True,

                default=str,
            )

        except Exception:

            encoded = str(
                payload
            )

        return hashlib.sha256(

            encoded.encode(
                "utf-8"
            )

        ).hexdigest()


# ============================================================
# AUDIT TRAIL ENGINE
# ============================================================

class AuditTrailEngine(
    BaseMonitoringEngine
):
    """
    Institutional audit engine.

    Responsibilities
    ----------------

    Pipeline events

    Model events

    Configuration events

    Portfolio events

    Execution events

    Compliance events
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata: MonitoringMetadata,

        config:
        MonitoringConfig | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.store = (
            AuditStore()
        )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    def log_event(
        self,
        *,
        event_name: str,

        source_system: str,

        category:
        MonitoringCategory,

        status:
        MonitoringStatus,

        details:
        dict[str, Any]
        | None = None,

        user_id:
        str | None = None,
    ) -> AuditRecord:

        record = (

            AuditEventBuilder
            .build_event(

                event_name=
                event_name,

                category=
                category,

                status=
                status,

                source_system=
                source_system,

                details=
                details,

                user_id=
                user_id,
            )
        )

        self.store.add(
            record
        )

        return record

    # --------------------------------------------------------
    # PIPELINE START
    # --------------------------------------------------------

    def log_pipeline_start(
        self,
        *,
        pipeline_name: str,
    ) -> None:

        self.log_event(

            event_name=
            "PIPELINE_START",

            source_system=
            pipeline_name,

            category=
            MonitoringCategory.AUDIT,

            status=
            MonitoringStatus.PASSED,

            details={

                "run_id":
                self.metadata.run_id,
            },
        )

    # --------------------------------------------------------
    # PIPELINE END
    # --------------------------------------------------------

    def log_pipeline_end(
        self,
        *,
        pipeline_name: str,

        success: bool,

        runtime_seconds:
        float,
    ) -> None:

        self.log_event(

            event_name=
            "PIPELINE_END",

            source_system=
            pipeline_name,

            category=
            MonitoringCategory.AUDIT,

            status=(

                MonitoringStatus.PASSED

                if success

                else MonitoringStatus.FAILED
            ),

            details={

                "runtime_seconds":
                runtime_seconds,

                "success":
                success,
            },
        )

    # --------------------------------------------------------
    # MODEL EVENT
    # --------------------------------------------------------

    def log_model_event(
        self,
        *,
        model_name: str,

        model_version: str,

        action: str,
    ) -> None:

        self.log_event(

            event_name=
            f"MODEL_{action}",

            source_system=
            model_name,

            category=
            MonitoringCategory.LINEAGE,

            status=
            MonitoringStatus.PASSED,

            details={

                "model_name":
                model_name,

                "model_version":
                model_version,
            },
        )

    # --------------------------------------------------------
    # CONFIG EVENT
    # --------------------------------------------------------

    def log_configuration_event(
        self,
        *,
        configuration:
        dict[str, Any],
    ) -> None:

        checksum = (
            AuditChecksumEngine
            .checksum(
                configuration
            )
        )

        self.log_event(

            event_name=
            "CONFIGURATION_CAPTURED",

            source_system=
            "CONFIGURATION",

            category=
            MonitoringCategory.CONFIGURATION,

            status=
            MonitoringStatus.PASSED,

            details={

                "checksum":
                checksum,
            },
        )

    # --------------------------------------------------------
    # EXPORT DF
    # --------------------------------------------------------

    def to_dataframe(
        self,
    ) -> pd.DataFrame:

        rows = []

        for record in (
            self.store.records()
        ):

            rows.append(

                {

                    "event_id":
                    record.event_id,

                    "timestamp":
                    record.event_timestamp,

                    "event_name":
                    record.event_name,

                    "category":
                    record.category.value,

                    "status":
                    record.status.value,

                    "source":
                    record.source_system,

                    "user":
                    record.user_id,
                }
            )

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
    ) -> AuditTrailResult:

        records = (
            self.store.records()
        )

        total_events = len(
            records
        )

        if total_events == 0:

            score = 1.0

        else:

            failures = sum(

                1

                for r in records

                if r.status
                == MonitoringStatus.FAILED
            )

            score = (

                1.0
                -
                failures
                /
                total_events
            )

        score = (
            self.clamp_score(
                score
            )
        )

        status = (
            self.determine_status(
                score
            )
        )

        severity = (
            self.determine_severity(
                score
            )
        )

        return AuditTrailResult(

            metadata=
            self.metadata,

            category=
            MonitoringCategory.AUDIT,

            status=
            status,

            severity=
            severity,

            score=
            score,

            diagnostics={

                "event_count":
                total_events,
            },

            audit_records=
            records,

            total_events=
            total_events,
        )
    

# ============================================================
# PART 4 — CONFIGURATION SNAPSHOT ENGINE
# ============================================================

import copy
import hashlib
import json

from typing import Any


# ============================================================
# CONFIG SERIALIZER
# ============================================================

class ConfigurationSerializer:
    """
    Converts configuration objects
    into deterministic dictionaries.
    """

    # --------------------------------------------------------

    @staticmethod
    def to_dict(
        configuration: Any,
    ) -> dict[str, Any]:

        if configuration is None:

            return {}

        # ------------------------------------------
        # dataclass
        # ------------------------------------------

        if hasattr(
            configuration,
            "__dataclass_fields__",
        ):

            result = {}

            for key in (
                configuration
                .__dataclass_fields__
            ):

                value = getattr(
                    configuration,
                    key,
                )

                result[key] = (
                    ConfigurationSerializer
                    .serialize_value(
                        value
                    )
                )

            return result

        # ------------------------------------------
        # dict
        # ------------------------------------------

        if isinstance(
            configuration,
            dict,
        ):

            return {

                str(k):
                ConfigurationSerializer
                .serialize_value(v)

                for k, v
                in configuration.items()
            }

        # ------------------------------------------
        # object
        # ------------------------------------------

        if hasattr(
            configuration,
            "__dict__",
        ):

            return {

                str(k):
                ConfigurationSerializer
                .serialize_value(v)

                for k, v
                in vars(
                    configuration
                ).items()
            }

        return {

            "value":
            str(configuration)
        }

    # --------------------------------------------------------

    @staticmethod
    def serialize_value(
        value: Any,
    ) -> Any:

        if value is None:

            return None

        if isinstance(
            value,
            (
                int,
                float,
                str,
                bool,
            ),
        ):

            return value

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):

            return [

                ConfigurationSerializer
                .serialize_value(v)

                for v in value
            ]

        if isinstance(
            value,
            dict,
        ):

            return {

                str(k):
                ConfigurationSerializer
                .serialize_value(v)

                for k, v
                in value.items()
            }

        if isinstance(
            value,
            datetime,
        ):

            return (
                value.isoformat()
            )

        if hasattr(
            value,
            "__dataclass_fields__",
        ):

            return (
                ConfigurationSerializer
                .to_dict(value)
            )

        return str(value)


# ============================================================
# CONFIG CHECKSUM ENGINE
# ============================================================

class ConfigurationChecksumEngine:
    """
    Deterministic configuration hash.
    """

    # --------------------------------------------------------

    @staticmethod
    def checksum(
        configuration:
        dict[str, Any],
    ) -> str:

        payload = json.dumps(

            configuration,

            sort_keys=True,

            default=str,
        )

        return hashlib.sha256(

            payload.encode(
                "utf-8"
            )

        ).hexdigest()


# ============================================================
# SNAPSHOT REPOSITORY
# ============================================================

class ConfigurationSnapshotStore:
    """
    In-memory snapshot repository.

    Can later be replaced by:

        SQL
        NoSQL
        S3
        Snowflake

    without changing engine code.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._snapshots: list[
            ConfigurationSnapshot
        ] = []

    # --------------------------------------------------------

    def add(
        self,
        snapshot:
        ConfigurationSnapshot,
    ) -> None:

        self._snapshots.append(
            snapshot
        )

    # --------------------------------------------------------

    def snapshots(
        self,
    ) -> list[
        ConfigurationSnapshot
    ]:

        return list(
            self._snapshots
        )

    # --------------------------------------------------------

    def latest(
        self,
    ) -> (
        ConfigurationSnapshot
        | None
    ):

        if (
            len(
                self._snapshots
            )
            == 0
        ):

            return None

        return (
            self._snapshots[-1]
        )


# ============================================================
# CONFIGURATION SNAPSHOT ENGINE
# ============================================================

class ConfigurationSnapshotEngine(
    BaseMonitoringEngine
):
    """
    Institutional configuration
    monitoring engine.

    Responsibilities
    ----------------

    Capture configuration

    Version control

    Checksum validation

    Drift detection

    Snapshot history
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.store = (
            ConfigurationSnapshotStore()
        )

    # --------------------------------------------------------
    # CREATE SNAPSHOT
    # --------------------------------------------------------

    def create_snapshot(
        self,
        *,
        configuration:
        Any,

        version:
        str = "1.0",
    ) -> ConfigurationSnapshot:

        config_dict = (

            ConfigurationSerializer
            .to_dict(
                configuration
            )
        )

        checksum = (

            ConfigurationChecksumEngine
            .checksum(
                config_dict
            )
        )

        snapshot = (
            ConfigurationSnapshot(

                snapshot_id=
                str(
                    uuid.uuid4()
                ),

                created_at=
                datetime.now(
                    UTC
                ),

                configuration=
                copy.deepcopy(
                    config_dict
                ),

                checksum=
                checksum,

                version=
                version,
            )
        )

        self.store.add(
            snapshot
        )

        return snapshot

    # --------------------------------------------------------
    # COMPARE SNAPSHOTS
    # --------------------------------------------------------

    def compare_snapshots(
        self,
        *,
        old_snapshot:
        ConfigurationSnapshot,

        new_snapshot:
        ConfigurationSnapshot,
    ) -> dict[str, Any]:

        old_cfg = (
            old_snapshot
            .configuration
        )

        new_cfg = (
            new_snapshot
            .configuration
        )

        changes = {}

        keys = set(
            old_cfg.keys()
        ) | set(
            new_cfg.keys()
        )

        for key in keys:

            old_value = (
                old_cfg.get(
                    key
                )
            )

            new_value = (
                new_cfg.get(
                    key
                )
            )

            if (
                old_value
                !=
                new_value
            ):

                changes[key] = {

                    "old":
                    old_value,

                    "new":
                    new_value,
                }

        return changes

    # --------------------------------------------------------
    # CONFIG DRIFT SCORE
    # --------------------------------------------------------

    def drift_score(
        self,
        *,
        changes:
        dict[str, Any],
        total_fields:
        int,
    ) -> float:

        if total_fields <= 0:

            return 0.0

        return float(
            len(changes)
            /
            total_fields
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        configuration:
        Any,

        version:
        str = "1.0",
    ) -> ConfigurationMonitoringResult:

        snapshot = (
            self.create_snapshot(

                configuration=
                configuration,

                version=
                version,
            )
        )

        snapshot_count = len(
            self.store.snapshots()
        )

        configuration_changes = 0

        drift = 0.0

        latest = None

        if snapshot_count >= 2:

            latest = (
                self.store
                .snapshots()[-2]
            )

            changes = (
                self.compare_snapshots(

                    old_snapshot=
                    latest,

                    new_snapshot=
                    snapshot,
                )
            )

            configuration_changes = (
                len(changes)
            )

            drift = (
                self.drift_score(

                    changes=
                    changes,

                    total_fields=
                    max(
                        1,
                        len(
                            snapshot
                            .configuration
                        )
                    ),
                )
            )

        score = (
            self.clamp_score(
                1.0 - drift
            )
        )

        status = (
            self.determine_status(
                score
            )
        )

        severity = (
            self.determine_severity(
                score
            )
        )

        return (
            ConfigurationMonitoringResult(

                metadata=
                self.metadata,

                category=
                MonitoringCategory
                .CONFIGURATION,

                status=
                status,

                severity=
                severity,

                score=
                score,

                diagnostics={

                    "snapshot_count":
                    snapshot_count,

                    "drift_score":
                    drift,
                },

                snapshot=
                snapshot,

                configuration_changes=
                configuration_changes,
            )
        )
    
# ============================================================
# PART 5 — MODEL LINEAGE TRACKING
# ============================================================

import hashlib
import inspect

from typing import Any


# ============================================================
# MODEL SIGNATURE ENGINE
# ============================================================

class ModelSignatureEngine:
    """
    Creates deterministic model signatures.
    """

    # --------------------------------------------------------

    @staticmethod
    def build_signature(
        *,
        model_name: str,

        model_version: str,

        feature_names: list[str],

        hyperparameters:
        dict[str, Any],
    ) -> str:

        payload = {

            "model_name":
            model_name,

            "model_version":
            model_version,

            "feature_names":
            sorted(
                feature_names
            ),

            "hyperparameters":
            hyperparameters,
        }

        encoded = json.dumps(

            payload,

            sort_keys=True,

            default=str,
        )

        return hashlib.sha256(

            encoded.encode(
                "utf-8"
            )

        ).hexdigest()


# ============================================================
# MODEL SOURCE HASH
# ============================================================

class ModelSourceHashEngine:
    """
    Hashes source code.

    Useful for:

        reproducibility
        governance
        model risk
    """

    # --------------------------------------------------------

    @staticmethod
    def source_hash(
        model_object: Any,
    ) -> str:

        try:

            source = inspect.getsource(
                model_object.__class__
            )

        except Exception:

            source = str(
                type(
                    model_object
                )
            )

        return hashlib.sha256(

            source.encode(
                "utf-8"
            )

        ).hexdigest()


# ============================================================
# LINEAGE STORE
# ============================================================

class ModelLineageStore:
    """
    In-memory lineage repository.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._records: list[
            ModelLineageRecord
        ] = []

    # --------------------------------------------------------

    def add(
        self,
        record:
        ModelLineageRecord,
    ) -> None:

        self._records.append(
            record
        )

    # --------------------------------------------------------

    def records(
        self,
    ) -> list[
        ModelLineageRecord
    ]:

        return list(
            self._records
        )

    # --------------------------------------------------------

    def latest(
        self,
    ) -> (
        ModelLineageRecord
        | None
    ):

        if (
            len(
                self._records
            )
            == 0
        ):

            return None

        return (
            self._records[-1]
        )


# ============================================================
# DATASET HASH ENGINE
# ============================================================

class DatasetHashEngine:
    """
    Dataset fingerprint.
    """

    # --------------------------------------------------------

    @staticmethod
    def dataframe_hash(
        dataframe:
        pd.DataFrame,
    ) -> str:

        if dataframe.empty:

            return "EMPTY"

        payload = (

            pd.util.hash_pandas_object(

                dataframe,

                index=True,
            )
            .values
            .tobytes()
        )

        return hashlib.sha256(
            payload
        ).hexdigest()


# ============================================================
# MODEL LINEAGE ENGINE
# ============================================================

class ModelLineageEngine(
    BaseMonitoringEngine
):
    """
    Institutional model lineage engine.

    Responsibilities
    ----------------

    Model registry

    Model provenance

    Dataset lineage

    Feature lineage

    Hyperparameter lineage

    Governance
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.store = (
            ModelLineageStore()
        )

    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    def register_model(
        self,
        *,
        model_name: str,

        model_version: str,

        model_object: Any,

        feature_names:
        list[str],

        training_dataset:
        pd.DataFrame
        | None = None,

        hyperparameters:
        dict[str, Any]
        | None = None,
    ) -> ModelLineageRecord:

        hyperparameters = (

            hyperparameters
            if hyperparameters
            is not None
            else {}
        )

        dataset_id = (
            "UNKNOWN"
        )

        if (
            training_dataset
            is not None
        ):

            dataset_id = (

                DatasetHashEngine
                .dataframe_hash(
                    training_dataset
                )
            )

        model_checksum = (

            ModelSignatureEngine
            .build_signature(

                model_name=
                model_name,

                model_version=
                model_version,

                feature_names=
                feature_names,

                hyperparameters=
                hyperparameters,
            )
        )

        source_hash = (

            ModelSourceHashEngine
            .source_hash(
                model_object
            )
        )

        lineage_record = (
            ModelLineageRecord(

                model_id=
                str(
                    uuid.uuid4()
                ),

                model_name=
                model_name,

                model_version=
                model_version,

                training_timestamp=
                datetime.now(
                    UTC
                ),

                feature_count=
                len(
                    feature_names
                ),

                training_dataset_id=
                dataset_id,

                checksum=
                model_checksum,

                metadata={

                    "feature_names":
                    feature_names,

                    "hyperparameters":
                    hyperparameters,

                    "source_hash":
                    source_hash,
                },
            )
        )

        self.store.add(
            lineage_record
        )

        return (
            lineage_record
        )

    # --------------------------------------------------------
    # FIND MODEL
    # --------------------------------------------------------

    def find_model(
        self,
        *,
        model_name: str,
    ) -> list[
        ModelLineageRecord
    ]:

        return [

            record

            for record
            in self.store.records()

            if (
                record.model_name
                ==
                model_name
            )
        ]

    # --------------------------------------------------------
    # VERSION COUNT
    # --------------------------------------------------------

    def version_count(
        self,
        *,
        model_name: str,
    ) -> int:

        return len(

            self.find_model(

                model_name=
                model_name
            )
        )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    def duplicate_models(
        self,
    ) -> list[
        ModelLineageRecord
    ]:

        seen = set()

        duplicates = []

        for record in (
            self.store.records()
        ):

            if (
                record.checksum
                in seen
            ):

                duplicates.append(
                    record
                )

            seen.add(
                record.checksum
            )

        return duplicates

    # --------------------------------------------------------
    # LINEAGE DATAFRAME
    # --------------------------------------------------------

    def to_dataframe(
        self,
    ) -> pd.DataFrame:

        rows = []

        for record in (
            self.store.records()
        ):

            rows.append(

                {

                    "model_id":
                    record.model_id,

                    "model_name":
                    record.model_name,

                    "model_version":
                    record.model_version,

                    "timestamp":
                    record.training_timestamp,

                    "feature_count":
                    record.feature_count,

                    "dataset_id":
                    record.training_dataset_id,

                    "checksum":
                    record.checksum,
                }
            )

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
    ) -> (
        ModelLineageMonitoringResult
    ):

        records = (
            self.store.records()
        )

        model_count = len(
            records
        )

        duplicates = (
            self.duplicate_models()
        )

        duplicate_count = len(
            duplicates
        )

        score = 1.0

        if model_count > 0:

            score = (

                1.0

                -
                (
                    duplicate_count
                    /
                    model_count
                )
            )

        score = (
            self.clamp_score(
                score
            )
        )

        status = (
            self.determine_status(
                score
            )
        )

        severity = (
            self.determine_severity(
                score
            )
        )

        return (
            ModelLineageMonitoringResult(

                metadata=
                self.metadata,

                category=
                MonitoringCategory
                .LINEAGE,

                status=
                status,

                severity=
                severity,

                score=
                score,

                diagnostics={

                    "model_count":
                    model_count,

                    "duplicate_count":
                    duplicate_count,
                },

                lineage_records=
                records,

                model_count=
                model_count,
            )
        )
    
# ============================================================
# PART 6 — RUNTIME MONITORING
# ============================================================

import gc
import os
import threading
import time

from typing import Any


# ============================================================
# PROCESS METRICS COLLECTOR
# ============================================================

class ProcessMetricsCollector:
    """
    Runtime metric collector.

    Uses standard library only.
    """

    # --------------------------------------------------------

    @staticmethod
    def cpu_count() -> int:

        try:

            return (
                os.cpu_count()
                or 0
            )

        except Exception:

            return 0

    # --------------------------------------------------------

    @staticmethod
    def thread_count() -> int:

        try:

            return (
                threading.active_count()
            )

        except Exception:

            return 0

    # --------------------------------------------------------

    @staticmethod
    def process_id() -> int:

        try:

            return os.getpid()

        except Exception:

            return -1

    # --------------------------------------------------------

    @staticmethod
    def garbage_objects() -> int:

        try:

            return len(
                gc.get_objects()
            )

        except Exception:

            return 0


# ============================================================
# RUNTIME TIMER
# ============================================================

class RuntimeTimer:
    """
    High precision runtime timer.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._start = (
            time.perf_counter()
        )

    # --------------------------------------------------------

    def elapsed(
        self,
    ) -> float:

        return float(

            time.perf_counter()

            -
            self._start
        )


# ============================================================
# RUNTIME REGISTRY
# ============================================================

class RuntimeRegistry:
    """
    Runtime history store.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._history: list[
            RuntimeMetrics
        ] = []

    # --------------------------------------------------------

    def add(
        self,
        metrics:
        RuntimeMetrics,
    ) -> None:

        self._history.append(
            metrics
        )

    # --------------------------------------------------------

    def history(
        self,
    ) -> list[
        RuntimeMetrics
    ]:

        return list(
            self._history
        )

    # --------------------------------------------------------

    def latest(
        self,
    ) -> (
        RuntimeMetrics
        | None
    ):

        if (
            len(
                self._history
            )
            == 0
        ):

            return None

        return (
            self._history[-1]
        )


# ============================================================
# RESOURCE SCORING ENGINE
# ============================================================

class ResourceScoringEngine:
    """
    Institutional runtime scoring.
    """

    # --------------------------------------------------------

    @staticmethod
    def runtime_score(
        runtime_seconds:
        float,
    ) -> float:

        if runtime_seconds <= 1:

            return 1.0

        if runtime_seconds <= 5:

            return 0.95

        if runtime_seconds <= 15:

            return 0.85

        if runtime_seconds <= 60:

            return 0.70

        return 0.50

    # --------------------------------------------------------

    @staticmethod
    def thread_score(
        thread_count:
        int,
    ) -> float:

        if thread_count <= 20:

            return 1.0

        if thread_count <= 50:

            return 0.90

        if thread_count <= 100:

            return 0.80

        return 0.60

    # --------------------------------------------------------

    @staticmethod
    def object_score(
        object_count:
        int,
    ) -> float:

        if object_count <= 50000:

            return 1.0

        if object_count <= 100000:

            return 0.90

        if object_count <= 200000:

            return 0.80

        return 0.60


# ============================================================
# RUNTIME MONITORING ENGINE
# ============================================================

class RuntimeMonitoringEngine(
    BaseMonitoringEngine
):
    """
    Institutional runtime monitoring.

    Tracks:

        Runtime
        Threads
        Objects
        Process metadata
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.registry = (
            RuntimeRegistry()
        )

    # --------------------------------------------------------
    # CAPTURE
    # --------------------------------------------------------

    def capture_metrics(
        self,
        *,
        runtime_seconds:
        float,
    ) -> RuntimeMetrics:

        thread_count = (

            ProcessMetricsCollector
            .thread_count()
        )

        object_count = (

            ProcessMetricsCollector
            .garbage_objects()
        )

        cpu_count = (

            ProcessMetricsCollector
            .cpu_count()
        )

        metrics = (
            RuntimeMetrics(

                runtime_seconds=
                runtime_seconds,

                cpu_utilization=
                float(cpu_count),

                memory_utilization=
                0.0,

                peak_memory_mb=
                0.0,

                io_operations=
                0,

                thread_count=
                thread_count,

                diagnostics={

                    "object_count":
                    object_count,

                    "process_id":
                    (
                        ProcessMetricsCollector
                        .process_id()
                    ),

                    "cpu_count":
                    cpu_count,
                },
            )
        )

        self.registry.add(
            metrics
        )

        return metrics

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    def score_metrics(
        self,
        metrics:
        RuntimeMetrics,
    ) -> float:

        runtime_score = (

            ResourceScoringEngine
            .runtime_score(

                metrics
                .runtime_seconds
            )
        )

        thread_score = (

            ResourceScoringEngine
            .thread_score(

                metrics
                .thread_count
            )
        )

        object_score = (

            ResourceScoringEngine
            .object_score(

                metrics
                .diagnostics
                .get(
                    "object_count",
                    0,
                )
            )
        )

        score = float(

            np.mean(

                [

                    runtime_score,

                    thread_score,

                    object_score,
                ]
            )
        )

        return (
            self.clamp_score(
                score
            )
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        runtime_seconds:
        float,
    ) -> RuntimeMonitoringResult:

        metrics = (
            self.capture_metrics(

                runtime_seconds=
                runtime_seconds
            )
        )

        score = (
            self.score_metrics(
                metrics
            )
        )

        status = (
            self.determine_status(
                score
            )
        )

        severity = (
            self.determine_severity(
                score
            )
        )

        return (
            RuntimeMonitoringResult(

                metadata=
                self.metadata,

                category=
                MonitoringCategory
                .RUNTIME,

                status=
                status,

                severity=
                severity,

                score=
                score,

                diagnostics={

                    "history_size":
                    len(
                        self.registry
                        .history()
                    )
                },

                runtime_metrics=
                metrics,
            )
        )


# ============================================================
# CONTEXT MANAGER
# ============================================================

class RuntimeMonitorContext:
    """
    Convenience runtime tracker.

    Example
    -------
    with RuntimeMonitorContext()
        as monitor:

        run_pipeline()

    runtime =
        monitor.runtime_seconds
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self.timer = RuntimeTimer()

        self.runtime_seconds = 0.0

    # --------------------------------------------------------

    def __enter__(
        self,
    ) -> "RuntimeMonitorContext":

        self.timer = (
            RuntimeTimer()
        )

        return self

    # --------------------------------------------------------

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:

        self.runtime_seconds = (

            self.timer.elapsed()
        )


# ============================================================
# PART 7 — HEALTH CHECKS
# ============================================================

from typing import Callable
from typing import Any


# ============================================================
# HEALTH CHECK REGISTRY
# ============================================================

class HealthCheckRegistry:
    """
    Registry of all health checks.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._checks: dict[
            str,
            Callable[[], HealthCheckResult]
        ] = {}

    # --------------------------------------------------------

    def register(
        self,
        *,
        name: str,

        check:
        Callable[
            [],
            HealthCheckResult
        ],
    ) -> None:

        self._checks[
            name
        ] = check

    # --------------------------------------------------------

    def checks(
        self,
    ) -> dict[
        str,
        Callable[
            [],
            HealthCheckResult
        ]
    ]:

        return dict(
            self._checks
        )


# ============================================================
# STANDARD HEALTH CHECKS
# ============================================================

class StandardHealthChecks:
    """
    Built-in health checks.
    """

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    @staticmethod
    def memory_check(
        *,
        object_limit:
        int = 200000,
    ) -> HealthCheckResult:

        object_count = len(
            gc.get_objects()
        )

        score = max(

            0.0,

            1.0
            -
            (
                object_count
                /
                object_limit
            )
        )

        status = (

            MonitoringStatus
            .PASSED

            if score >= 0.95

            else (
                MonitoringStatus
                .WARNING

                if score >= 0.80

                else
                MonitoringStatus
                .FAILED
            )
        )

        return (
            HealthCheckResult(

                component_name=
                "MEMORY",

                score=
                float(score),

                status=
                status,

                diagnostics={

                    "object_count":
                    object_count,

                    "limit":
                    object_limit,
                },
            )
        )

    # --------------------------------------------------------
    # THREADS
    # --------------------------------------------------------

    @staticmethod
    def thread_check(
        *,
        thread_limit:
        int = 100,
    ) -> HealthCheckResult:

        thread_count = (
            threading.active_count()
        )

        score = max(

            0.0,

            1.0
            -
            (
                thread_count
                /
                thread_limit
            )
        )

        status = (

            MonitoringStatus
            .PASSED

            if score >= 0.95

            else (
                MonitoringStatus
                .WARNING

                if score >= 0.80

                else
                MonitoringStatus
                .FAILED
            )
        )

        return (
            HealthCheckResult(

                component_name=
                "THREADS",

                score=
                float(score),

                status=
                status,

                diagnostics={

                    "thread_count":
                    thread_count,

                    "limit":
                    thread_limit,
                },
            )
        )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    @staticmethod
    def process_check(
    ) -> HealthCheckResult:

        pid = os.getpid()

        score = (
            1.0
            if pid > 0
            else 0.0
        )

        status = (

            MonitoringStatus
            .PASSED

            if score >= 1.0

            else
            MonitoringStatus
            .FAILED
        )

        return (
            HealthCheckResult(

                component_name=
                "PROCESS",

                score=
                score,

                status=
                status,

                diagnostics={

                    "pid":
                    pid,
                },
            )
        )

    # --------------------------------------------------------
    # FILESYSTEM
    # --------------------------------------------------------

    @staticmethod
    def filesystem_check(
        *,
        path:
        str = ".",
    ) -> HealthCheckResult:

        exists = os.path.exists(
            path
        )

        score = (
            1.0
            if exists
            else 0.0
        )

        status = (

            MonitoringStatus
            .PASSED

            if exists

            else
            MonitoringStatus
            .FAILED
        )

        return (
            HealthCheckResult(

                component_name=
                "FILESYSTEM",

                score=
                score,

                status=
                status,

                diagnostics={

                    "path":
                    path,

                    "exists":
                    exists,
                },
            )
        )


# ============================================================
# HEALTH SCORE AGGREGATOR
# ============================================================

class HealthScoreAggregator:
    """
    Aggregates component health.
    """

    # --------------------------------------------------------

    @staticmethod
    def aggregate(
        checks:
        list[
            HealthCheckResult
        ],
    ) -> float:

        if len(checks) == 0:

            return 1.0

        scores = [

            c.score

            for c in checks
        ]

        return float(
            np.mean(
                scores
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def warnings(
        checks:
        list[
            HealthCheckResult
        ],
    ) -> list[str]:

        warnings = []

        for check in checks:

            if (
                check.status
                ==
                MonitoringStatus
                .WARNING
            ):

                warnings.append(

                    f"{check.component_name}"
                    f" WARNING"
                )

        return warnings

    # --------------------------------------------------------

    @staticmethod
    def failures(
        checks:
        list[
            HealthCheckResult
        ],
    ) -> list[str]:

        failures = []

        for check in checks:

            if (
                check.status
                ==
                MonitoringStatus
                .FAILED
            ):

                failures.append(

                    f"{check.component_name}"
                    f" FAILED"
                )

        return failures


# ============================================================
# HEALTH MONITORING ENGINE
# ============================================================

class HealthMonitoringEngine(
    BaseMonitoringEngine
):
    """
    Institutional health engine.

    Executes:

        memory checks
        thread checks
        process checks
        filesystem checks

    Produces:

        HealthMonitoringResult
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.registry = (
            HealthCheckRegistry()
        )

        self._register_defaults()

    # --------------------------------------------------------

    def _register_defaults(
        self,
    ) -> None:

        self.registry.register(

            name="MEMORY",

            check=lambda:
            StandardHealthChecks
            .memory_check(),
        )

        self.registry.register(

            name="THREADS",

            check=lambda:
            StandardHealthChecks
            .thread_check(),
        )

        self.registry.register(

            name="PROCESS",

            check=lambda:
            StandardHealthChecks
            .process_check(),
        )

        self.registry.register(

            name="FILESYSTEM",

            check=lambda:
            StandardHealthChecks
            .filesystem_check(),
        )

    # --------------------------------------------------------
    # REGISTER CUSTOM CHECK
    # --------------------------------------------------------

    def register_check(
        self,
        *,
        name: str,

        check:
        Callable[
            [],
            HealthCheckResult
        ],
    ) -> None:

        self.registry.register(

            name=name,

            check=check,
        )

    # --------------------------------------------------------
    # RUN CHECKS
    # --------------------------------------------------------

    def run_checks(
        self,
    ) -> list[
        HealthCheckResult
    ]:

        results = []

        for (
            _,
            check,
        ) in (
            self.registry
            .checks()
            .items()
        ):

            try:

                results.append(
                    check()
                )

            except Exception as exc:

                results.append(

                    HealthCheckResult(

                        component_name=
                        "UNKNOWN",

                        score=0.0,

                        status=
                        MonitoringStatus
                        .FAILED,

                        diagnostics={

                            "error":
                            str(exc)
                        },
                    )
                )

        return results

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
    ) -> HealthMonitoringResult:

        checks = (
            self.run_checks()
        )

        score = (

            HealthScoreAggregator
            .aggregate(
                checks
            )
        )

        status = (
            self.determine_status(
                score
            )
        )

        severity = (
            self.determine_severity(
                score
            )
        )

        warnings = (

            HealthScoreAggregator
            .warnings(
                checks
            )
        )

        failures = (

            HealthScoreAggregator
            .failures(
                checks
            )
        )

        return (
            HealthMonitoringResult(

                metadata=
                self.metadata,

                category=
                MonitoringCategory
                .HEALTH,

                status=
                status,

                severity=
                severity,

                score=
                score,

                diagnostics={

                    "warning_count":
                    len(
                        warnings
                    ),

                    "failure_count":
                    len(
                        failures
                    ),
                },

                warnings=
                warnings,

                errors=
                failures,

                health_checks=
                checks,

                average_health_score=
                score,
            )
        )
    
# ============================================================
# PART 8 — COMPLIANCE MONITORING
# ============================================================

from typing import Any
from typing import Callable


# ============================================================
# COMPLIANCE RULE
# ============================================================

@dataclass(slots=True)
class ComplianceRule:
    """
    Compliance rule definition.
    """

    rule_id: str

    rule_name: str

    description: str

    severity: MonitoringSeverity

    enabled: bool = True


# ============================================================
# COMPLIANCE REGISTRY
# ============================================================

class ComplianceRuleRegistry:
    """
    Registry of compliance rules.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._rules: dict[
            str,
            tuple[
                ComplianceRule,
                Callable[
                    [dict[str, Any]],
                    ComplianceCheckResult
                ],
            ],
        ] = {}

    # --------------------------------------------------------

    def register(
        self,
        *,
        rule: ComplianceRule,

        evaluator:
        Callable[
            [dict[str, Any]],
            ComplianceCheckResult
        ],
    ) -> None:

        self._rules[
            rule.rule_id
        ] = (
            rule,
            evaluator,
        )

    # --------------------------------------------------------

    def rules(
        self,
    ) -> dict[
        str,
        tuple[
            ComplianceRule,
            Callable[
                [dict[str, Any]],
                ComplianceCheckResult
            ],
        ],
    ]:

        return dict(
            self._rules
        )


# ============================================================
# STANDARD COMPLIANCE RULES
# ============================================================

class StandardComplianceRules:
    """
    Institutional compliance checks.
    """

    # --------------------------------------------------------
    # POSITION LIMIT
    # --------------------------------------------------------

    @staticmethod
    def position_limit_rule(
        context:
        dict[str, Any],
    ) -> ComplianceCheckResult:

        max_weight = float(
            context.get(
                "max_weight",
                0.0,
            )
        )

        limit = float(
            context.get(
                "position_limit",
                0.10,
            )
        )

        passed = (
            max_weight
            <= limit
        )

        return ComplianceCheckResult(

            rule_name=
            "POSITION_LIMIT",

            passed=
            passed,

            severity=
            MonitoringSeverity.HIGH,

            message=(

                "Position limit respected"

                if passed

                else
                "Position limit breached"
            ),

            diagnostics={

                "max_weight":
                max_weight,

                "limit":
                limit,
            },
        )

    # --------------------------------------------------------
    # CONCENTRATION LIMIT
    # --------------------------------------------------------

    @staticmethod
    def concentration_rule(
        context:
        dict[str, Any],
    ) -> ComplianceCheckResult:

        hhi = float(
            context.get(
                "hhi",
                0.0,
            )
        )

        limit = float(
            context.get(
                "hhi_limit",
                0.15,
            )
        )

        passed = (
            hhi
            <= limit
        )

        return ComplianceCheckResult(

            rule_name=
            "CONCENTRATION",

            passed=
            passed,

            severity=
            MonitoringSeverity.HIGH,

            message=(

                "Concentration acceptable"

                if passed

                else
                "Concentration too high"
            ),

            diagnostics={

                "hhi":
                hhi,

                "limit":
                limit,
            },
        )

    # --------------------------------------------------------
    # TURNOVER LIMIT
    # --------------------------------------------------------

    @staticmethod
    def turnover_rule(
        context:
        dict[str, Any],
    ) -> ComplianceCheckResult:

        turnover = float(
            context.get(
                "turnover",
                0.0,
            )
        )

        limit = float(
            context.get(
                "turnover_limit",
                0.50,
            )
        )

        passed = (
            turnover
            <= limit
        )

        return ComplianceCheckResult(

            rule_name=
            "TURNOVER",

            passed=
            passed,

            severity=
            MonitoringSeverity.MEDIUM,

            message=(

                "Turnover acceptable"

                if passed

                else
                "Turnover exceeds policy"
            ),

            diagnostics={

                "turnover":
                turnover,

                "limit":
                limit,
            },
        )

    # --------------------------------------------------------
    # LEVERAGE LIMIT
    # --------------------------------------------------------

    @staticmethod
    def leverage_rule(
        context:
        dict[str, Any],
    ) -> ComplianceCheckResult:

        gross_exposure = float(
            context.get(
                "gross_exposure",
                0.0,
            )
        )

        limit = float(
            context.get(
                "gross_limit",
                1.50,
            )
        )

        passed = (
            gross_exposure
            <= limit
        )

        return ComplianceCheckResult(

            rule_name=
            "LEVERAGE",

            passed=
            passed,

            severity=
            MonitoringSeverity.CRITICAL,

            message=(

                "Leverage acceptable"

                if passed

                else
                "Leverage breach"
            ),

            diagnostics={

                "gross_exposure":
                gross_exposure,

                "limit":
                limit,
            },
        )

    # --------------------------------------------------------
    # LIQUIDITY RULE
    # --------------------------------------------------------

    @staticmethod
    def liquidity_rule(
        context:
        dict[str, Any],
    ) -> ComplianceCheckResult:

        liquidity_score = float(
            context.get(
                "liquidity_score",
                1.0,
            )
        )

        minimum = float(
            context.get(
                "minimum_liquidity",
                0.50,
            )
        )

        passed = (
            liquidity_score
            >= minimum
        )

        return ComplianceCheckResult(

            rule_name=
            "LIQUIDITY",

            passed=
            passed,

            severity=
            MonitoringSeverity.HIGH,

            message=(

                "Liquidity acceptable"

                if passed

                else
                "Liquidity insufficient"
            ),

            diagnostics={

                "liquidity_score":
                liquidity_score,

                "minimum":
                minimum,
            },
        )


# ============================================================
# COMPLIANCE SCORE ENGINE
# ============================================================

class ComplianceScoreEngine:
    """
    Converts rule results
    into portfolio compliance score.
    """

    # --------------------------------------------------------

    @staticmethod
    def score(
        checks:
        list[
            ComplianceCheckResult
        ],
    ) -> float:

        if len(checks) == 0:

            return 1.0

        passed = sum(

            1

            for c in checks

            if c.passed
        )

        return float(
            passed
            /
            len(checks)
        )


# ============================================================
# COMPLIANCE ENGINE
# ============================================================

class ComplianceMonitoringEngine(
    BaseMonitoringEngine
):
    """
    Institutional compliance engine.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata: MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.registry = (
            ComplianceRuleRegistry()
        )

        self._register_defaults()

    # --------------------------------------------------------

    def _register_defaults(
        self,
    ) -> None:

        self.registry.register(

            rule=
            ComplianceRule(

                rule_id=
                "POSITION_LIMIT",

                rule_name=
                "Position Limit",

                description=
                "Maximum position weight",

                severity=
                MonitoringSeverity.HIGH,
            ),

            evaluator=
            StandardComplianceRules
            .position_limit_rule,
        )

        self.registry.register(

            rule=
            ComplianceRule(

                rule_id=
                "CONCENTRATION",

                rule_name=
                "Concentration",

                description=
                "HHI concentration",

                severity=
                MonitoringSeverity.HIGH,
            ),

            evaluator=
            StandardComplianceRules
            .concentration_rule,
        )

        self.registry.register(

            rule=
            ComplianceRule(

                rule_id=
                "TURNOVER",

                rule_name=
                "Turnover",

                description=
                "Portfolio turnover",

                severity=
                MonitoringSeverity.MEDIUM,
            ),

            evaluator=
            StandardComplianceRules
            .turnover_rule,
        )

        self.registry.register(

            rule=
            ComplianceRule(

                rule_id=
                "LEVERAGE",

                rule_name=
                "Leverage",

                description=
                "Gross exposure",

                severity=
                MonitoringSeverity.CRITICAL,
            ),

            evaluator=
            StandardComplianceRules
            .leverage_rule,
        )

        self.registry.register(

            rule=
            ComplianceRule(

                rule_id=
                "LIQUIDITY",

                rule_name=
                "Liquidity",

                description=
                "Liquidity threshold",

                severity=
                MonitoringSeverity.HIGH,
            ),

            evaluator=
            StandardComplianceRules
            .liquidity_rule,
        )

    # --------------------------------------------------------
    # CUSTOM RULE
    # --------------------------------------------------------

    def register_rule(
        self,
        *,
        rule: ComplianceRule,

        evaluator:
        Callable[
            [dict[str, Any]],
            ComplianceCheckResult
        ],
    ) -> None:

        self.registry.register(

            rule=rule,

            evaluator=evaluator,
        )

    # --------------------------------------------------------
    # RUN CHECKS
    # --------------------------------------------------------

    def run_checks(
        self,
        *,
        context:
        dict[str, Any],
    ) -> list[
        ComplianceCheckResult
    ]:

        results = []

        for (
            _,
            (
                rule,
                evaluator,
            ),
        ) in (
            self.registry
            .rules()
            .items()
        ):

            if not rule.enabled:

                continue

            try:

                results.append(
                    evaluator(
                        context
                    )
                )

            except Exception as exc:

                results.append(

                    ComplianceCheckResult(

                        rule_name=
                        rule.rule_name,

                        passed=
                        False,

                        severity=
                        MonitoringSeverity
                        .CRITICAL,

                        message=
                        str(exc),
                    )
                )

        return results

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        context:
        dict[str, Any],
    ) -> ComplianceMonitoringResult:

        checks = (
            self.run_checks(

                context=
                context
            )
        )

        score = (
            ComplianceScoreEngine
            .score(
                checks
            )
        )

        status = (
            self.determine_status(
                score
            )
        )

        severity = (
            self.determine_severity(
                score
            )
        )

        violations = sum(

            1

            for c in checks

            if not c.passed
        )

        return (
            ComplianceMonitoringResult(

                metadata=
                self.metadata,

                category=
                MonitoringCategory
                .COMPLIANCE,

                status=
                status,

                severity=
                severity,

                score=
                score,

                diagnostics={

                    "rule_count":
                    len(checks),

                    "violations":
                    violations,
                },

                compliance_checks=
                checks,

                violations=
                violations,
            )
        )
    

# ============================================================
# PART 9 — ALERTING ENGINE
# ============================================================

from typing import Any
from typing import Iterable


# ============================================================
# ALERT RULE
# ============================================================

@dataclass(slots=True)
class AlertRule:
    """
    Alert rule definition.
    """

    rule_id: str

    rule_name: str

    severity: MonitoringSeverity

    category: MonitoringCategory

    enabled: bool = True

    threshold: float | None = None

    description: str = ""


# ============================================================
# ALERT REGISTRY
# ============================================================

class AlertRuleRegistry:
    """
    Stores alert rules.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._rules: dict[
            str,
            AlertRule
        ] = {}

    # --------------------------------------------------------

    def register(
        self,
        rule: AlertRule,
    ) -> None:

        self._rules[
            rule.rule_id
        ] = rule

    # --------------------------------------------------------

    def rules(
        self,
    ) -> list[
        AlertRule
    ]:

        return list(
            self._rules.values()
        )


# ============================================================
# ALERT STORE
# ============================================================

class AlertStore:
    """
    Persistent in-memory alert history.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._alerts: list[
            AlertRecord
        ] = []

    # --------------------------------------------------------

    def add(
        self,
        alert:
        AlertRecord,
    ) -> None:

        self._alerts.append(
            alert
        )

    # --------------------------------------------------------

    def history(
        self,
    ) -> list[
        AlertRecord
    ]:

        return list(
            self._alerts
        )

    # --------------------------------------------------------

    def latest(
        self,
    ) -> (
        AlertRecord
        | None
    ):

        if (
            len(
                self._alerts
            )
            == 0
        ):

            return None

        return (
            self._alerts[-1]
        )


# ============================================================
# ALERT FACTORY
# ============================================================

class AlertFactory:
    """
    Creates standardized alerts.
    """

    # --------------------------------------------------------

    @staticmethod
    def create(
        *,
        category:
        MonitoringCategory,

        severity:
        MonitoringSeverity,

        title: str,

        message: str,

        source: str,

        diagnostics:
        dict[str, Any]
        | None = None,
    ) -> AlertRecord:

        return AlertRecord(

            alert_id=
            str(
                uuid.uuid4()
            ),

            timestamp=
            datetime.now(
                UTC
            ),

            category=
            category,

            severity=
            severity,

            title=
            title,

            message=
            message,

            source=
            source,

            diagnostics=(
                diagnostics
                if diagnostics
                is not None
                else {}
            ),
        )


# ============================================================
# ALERT EVALUATOR
# ============================================================

class AlertEvaluator:
    """
    Converts monitoring results
    into alert records.
    """

    # --------------------------------------------------------

    @staticmethod
    def evaluate_score(
        *,
        score: float,

        category:
        MonitoringCategory,

        source: str,
    ) -> list[
        AlertRecord
    ]:

        alerts = []

        if score < 0.50:

            alerts.append(

                AlertFactory.create(

                    category=
                    category,

                    severity=
                    MonitoringSeverity
                    .CRITICAL,

                    title=
                    "Critical Monitoring Score",

                    message=(
                        f"Score={score:.3f}"
                    ),

                    source=
                    source,
                )
            )

        elif score < 0.75:

            alerts.append(

                AlertFactory.create(

                    category=
                    category,

                    severity=
                    MonitoringSeverity
                    .HIGH,

                    title=
                    "Monitoring Warning",

                    message=(
                        f"Score={score:.3f}"
                    ),

                    source=
                    source,
                )
            )

        return alerts

    # --------------------------------------------------------

    @staticmethod
    def evaluate_status(
        *,
        status:
        MonitoringStatus,

        category:
        MonitoringCategory,

        source: str,
    ) -> list[
        AlertRecord
    ]:

        alerts = []

        if (
            status
            ==
            MonitoringStatus
            .FAILED
        ):

            alerts.append(

                AlertFactory.create(

                    category=
                    category,

                    severity=
                    MonitoringSeverity
                    .CRITICAL,

                    title=
                    "Monitoring Failure",

                    message=
                    "Component failed",

                    source=
                    source,
                )
            )

        return alerts


# ============================================================
# ALERT ROUTER
# ============================================================

class AlertRouter:
    """
    Routes alerts to channels.

    Current implementation:

        console
        memory store

    Future:

        email
        slack
        teams
        pagerduty
    """

    # --------------------------------------------------------

    def route(
        self,
        alert:
        AlertRecord,
    ) -> None:

        print(

            "[ALERT]",

            alert.severity,

            alert.category,

            alert.title,

            alert.message,
        )


# ============================================================
# ALERT ENGINE
# ============================================================

class AlertingEngine(
    BaseMonitoringEngine
):
    """
    Institutional alert platform.

    Responsibilities
    ----------------

    Alert generation

    Alert routing

    Alert storage

    Alert aggregation
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.registry = (
            AlertRuleRegistry()
        )

        self.store = (
            AlertStore()
        )

        self.router = (
            AlertRouter()
        )

    # --------------------------------------------------------
    # PROCESS RESULT
    # --------------------------------------------------------

    def process_result(
        self,
        *,
        score: float,

        status:
        MonitoringStatus,

        category:
        MonitoringCategory,

        source: str,
    ) -> list[
        AlertRecord
    ]:

        alerts = []

        alerts.extend(

            AlertEvaluator
            .evaluate_score(

                score=
                score,

                category=
                category,

                source=
                source,
            )
        )

        alerts.extend(

            AlertEvaluator
            .evaluate_status(

                status=
                status,

                category=
                category,

                source=
                source,
            )
        )

        return alerts

    # --------------------------------------------------------
    # STORE ALERTS
    # --------------------------------------------------------

    def persist(
        self,
        alerts:
        Iterable[
            AlertRecord
        ],
    ) -> None:

        for alert in alerts:

            self.store.add(
                alert
            )

            self.router.route(
                alert
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        score: float,

        status:
        MonitoringStatus,

        category:
        MonitoringCategory,

        source: str,
    ) -> AlertMonitoringResult:

        alerts = (
            self.process_result(

                score=
                score,

                status=
                status,

                category=
                category,

                source=
                source,
            )
        )

        self.persist(
            alerts
        )

        alert_count = len(
            alerts
        )

        monitoring_score = (
            1.0

            if alert_count == 0

            else max(
                0.0,
                1.0
                -
                (
                    alert_count
                    * 0.10
                ),
            )
        )

        overall_status = (
            self.determine_status(
                monitoring_score
            )
        )

        overall_severity = (
            self.determine_severity(
                monitoring_score
            )
        )

        return (
            AlertMonitoringResult(

                metadata=
                self.metadata,

                category=
                MonitoringCategory
                .ALERTING,

                status=
                overall_status,

                severity=
                overall_severity,

                score=
                monitoring_score,

                diagnostics={

                    "alert_count":
                    alert_count,

                    "history_size":
                    len(
                        self.store
                        .history()
                    ),
                },

                alerts=
                alerts,

                total_alerts=
                alert_count,
            )
        )


# ============================================================
# ALERT QUERY ENGINE
# ============================================================

class AlertQueryEngine:
    """
    Query alert history.
    """

    # --------------------------------------------------------

    @staticmethod
    def filter_by_severity(
        alerts:
        list[
            AlertRecord
        ],

        severity:
        MonitoringSeverity,
    ) -> list[
        AlertRecord
    ]:

        return [

            alert

            for alert
            in alerts

            if (
                alert.severity
                ==
                severity
            )
        ]

    # --------------------------------------------------------

    @staticmethod
    def filter_by_category(
        alerts:
        list[
            AlertRecord
        ],

        category:
        MonitoringCategory,
    ) -> list[
        AlertRecord
    ]:

        return [

            alert

            for alert
            in alerts

            if (
                alert.category
                ==
                category
            )
        ]

    # --------------------------------------------------------

    @staticmethod
    def critical_alerts(
        alerts:
        list[
            AlertRecord
        ],
    ) -> list[
        AlertRecord
    ]:

        return [

            alert

            for alert
            in alerts

            if (
                alert.severity
                ==
                MonitoringSeverity
                .CRITICAL
            )
        ]
    
# ============================================================
# PART 10 — MONITORING DIAGNOSTICS
# ============================================================

from dataclasses import dataclass
from dataclasses import field

from typing import Any


# ============================================================
# DIAGNOSTIC SNAPSHOT
# ============================================================

@dataclass(slots=True)
class MonitoringDiagnosticSnapshot:
    """
    Unified monitoring snapshot.
    """

    timestamp: datetime

    overall_score: float

    overall_status: MonitoringStatus

    runtime_score: float

    health_score: float

    compliance_score: float

    alert_score: float

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# DIAGNOSTIC TREND
# ============================================================

@dataclass(slots=True)
class MonitoringTrend:
    """
    Trend statistics.
    """

    metric_name: str

    latest_value: float

    average_value: float

    minimum_value: float

    maximum_value: float

    trend_direction: str


# ============================================================
# SCORE AGGREGATOR
# ============================================================

class MonitoringScoreAggregator:
    """
    Aggregates monitoring scores.
    """

    # --------------------------------------------------------

    @staticmethod
    def aggregate_scores(
        *,
        runtime_score: float,

        health_score: float,

        compliance_score: float,

        alert_score: float,
    ) -> float:

        scores = [

            runtime_score,

            health_score,

            compliance_score,

            alert_score,
        ]

        scores = [

            float(x)

            for x in scores

            if x is not None
        ]

        if len(scores) == 0:

            return 0.0

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------

    @staticmethod
    def determine_status(
        score: float,
    ) -> MonitoringStatus:

        if score >= 0.95:

            return (
                MonitoringStatus
                .PASSED
            )

        if score >= 0.75:

            return (
                MonitoringStatus
                .WARNING
            )

        return (
            MonitoringStatus
            .FAILED
        )


# ============================================================
# TREND ANALYZER
# ============================================================

class MonitoringTrendAnalyzer:
    """
    Monitoring trend analysis.
    """

    # --------------------------------------------------------

    @staticmethod
    def analyze(
        *,
        metric_name: str,

        values: list[float],
    ) -> MonitoringTrend:

        if len(values) == 0:

            return MonitoringTrend(

                metric_name=
                metric_name,

                latest_value=
                0.0,

                average_value=
                0.0,

                minimum_value=
                0.0,

                maximum_value=
                0.0,

                trend_direction=
                "UNKNOWN",
            )

        latest = float(
            values[-1]
        )

        average = float(
            np.mean(values)
        )

        minimum = float(
            np.min(values)
        )

        maximum = float(
            np.max(values)
        )

        trend = "FLAT"

        if len(values) >= 2:

            if (
                values[-1]
                >
                values[0]
            ):

                trend = "UP"

            elif (
                values[-1]
                <
                values[0]
            ):

                trend = "DOWN"

        return MonitoringTrend(

            metric_name=
            metric_name,

            latest_value=
            latest,

            average_value=
            average,

            minimum_value=
            minimum,

            maximum_value=
            maximum,

            trend_direction=
            trend,
        )


# ============================================================
# DIAGNOSTIC REGISTRY
# ============================================================

class MonitoringDiagnosticRegistry:
    """
    Stores historical monitoring
    snapshots.
    """

    # --------------------------------------------------------

    def __init__(
        self,
    ) -> None:

        self._snapshots: list[
            MonitoringDiagnosticSnapshot
        ] = []

    # --------------------------------------------------------

    def add(
        self,
        snapshot:
        MonitoringDiagnosticSnapshot,
    ) -> None:

        self._snapshots.append(
            snapshot
        )

    # --------------------------------------------------------

    def snapshots(
        self,
    ) -> list[
        MonitoringDiagnosticSnapshot
    ]:

        return list(
            self._snapshots
        )

    # --------------------------------------------------------

    def latest(
        self,
    ) -> (
        MonitoringDiagnosticSnapshot
        | None
    ):

        if (
            len(
                self._snapshots
            )
            == 0
        ):

            return None

        return (
            self._snapshots[-1]
        )


# ============================================================
# DIAGNOSTIC ENGINE
# ============================================================

class MonitoringDiagnosticsEngine(
    BaseMonitoringEngine
):
    """
    Institutional diagnostics engine.

    Combines:

        Runtime Monitoring
        Health Monitoring
        Compliance Monitoring
        Alert Monitoring

    Produces:

        MonitoringDiagnosticSnapshot
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig
        | None = None,
    ) -> None:

        super().__init__(

            metadata=
            metadata,

            config=
            config,
        )

        self.registry = (
            MonitoringDiagnosticRegistry()
        )

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    def create_snapshot(
        self,
        *,
        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,

        alert_result:
        AlertMonitoringResult,
    ) -> MonitoringDiagnosticSnapshot:

        overall_score = (

            MonitoringScoreAggregator
            .aggregate_scores(

                runtime_score=
                runtime_result.score,

                health_score=
                health_result.score,

                compliance_score=
                compliance_result.score,

                alert_score=
                alert_result.score,
            )
        )

        overall_status = (

            MonitoringScoreAggregator
            .determine_status(

                overall_score
            )
        )

        snapshot = (
            MonitoringDiagnosticSnapshot(

                timestamp=
                datetime.now(
                    UTC
                ),

                overall_score=
                overall_score,

                overall_status=
                overall_status,

                runtime_score=
                runtime_result.score,

                health_score=
                health_result.score,

                compliance_score=
                compliance_result.score,

                alert_score=
                alert_result.score,

                diagnostics={

                    "runtime":
                    runtime_result
                    .diagnostics,

                    "health":
                    health_result
                    .diagnostics,

                    "compliance":
                    compliance_result
                    .diagnostics,

                    "alerting":
                    alert_result
                    .diagnostics,
                },
            )
        )

        self.registry.add(
            snapshot
        )

        return snapshot

    # --------------------------------------------------------
    # SCORE TRENDS
    # --------------------------------------------------------

    def score_trends(
        self,
    ) -> list[
        MonitoringTrend
    ]:

        snapshots = (
            self.registry
            .snapshots()
        )

        if len(snapshots) == 0:

            return []

        overall_scores = [

            s.overall_score

            for s
            in snapshots
        ]

        runtime_scores = [

            s.runtime_score

            for s
            in snapshots
        ]

        health_scores = [

            s.health_score

            for s
            in snapshots
        ]

        compliance_scores = [

            s.compliance_score

            for s
            in snapshots
        ]

        alert_scores = [

            s.alert_score

            for s
            in snapshots
        ]

        return [

            MonitoringTrendAnalyzer
            .analyze(

                metric_name=
                "OVERALL_SCORE",

                values=
                overall_scores,
            ),

            MonitoringTrendAnalyzer
            .analyze(

                metric_name=
                "RUNTIME_SCORE",

                values=
                runtime_scores,
            ),

            MonitoringTrendAnalyzer
            .analyze(

                metric_name=
                "HEALTH_SCORE",

                values=
                health_scores,
            ),

            MonitoringTrendAnalyzer
            .analyze(

                metric_name=
                "COMPLIANCE_SCORE",

                values=
                compliance_scores,
            ),

            MonitoringTrendAnalyzer
            .analyze(

                metric_name=
                "ALERT_SCORE",

                values=
                alert_scores,
            ),
        ]

    # --------------------------------------------------------
    # SYSTEM DIAGNOSTICS
    # --------------------------------------------------------

    def system_diagnostics(
        self,
    ) -> dict[str, Any]:

        snapshots = (
            self.registry
            .snapshots()
        )

        trends = (
            self.score_trends()
        )

        return {

            "snapshot_count":
            len(
                snapshots
            ),

            "latest_snapshot":
            self.registry.latest(),

            "trends":
            trends,
        }

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,

        alert_result:
        AlertMonitoringResult,
    ) -> MonitoringDiagnosticSnapshot:

        return (
            self.create_snapshot(

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,
            )
        )
    

# ============================================================
# PART 11 — INSTITUTIONAL MONITORING REPORT
# ============================================================

from dataclasses import dataclass
from dataclasses import field

from typing import Any


# ============================================================
# REPORT SUMMARY
# ============================================================

@dataclass(slots=True)
class MonitoringReportSummary:
    """
    Executive monitoring summary.
    """

    overall_score: float

    overall_status: MonitoringStatus

    severity: MonitoringSeverity

    total_alerts: int

    compliance_violations: int

    health_score: float

    runtime_score: float

    compliance_score: float

    alert_score: float


# ============================================================
# REPORT SECTION
# ============================================================

@dataclass(slots=True)
class MonitoringReportSection:
    """
    Generic report section.
    """

    section_name: str

    status: MonitoringStatus

    score: float

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# INSTITUTIONAL REPORT
# ============================================================

@dataclass(slots=True)
class InstitutionalMonitoringReport:
    """
    Master monitoring report.

    Single source of truth for
    all monitoring outputs.
    """

    metadata: MonitoringMetadata

    summary: MonitoringReportSummary

    runtime_section: MonitoringReportSection

    health_section: MonitoringReportSection

    compliance_section: MonitoringReportSection

    alert_section: MonitoringReportSection

    diagnostic_snapshot: MonitoringDiagnosticSnapshot

    trends: list[
        MonitoringTrend
    ] = field(
        default_factory=list
    )

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# SUMMARY BUILDER
# ============================================================

class MonitoringSummaryBuilder:
    """
    Builds executive summary.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,

        alert_result:
        AlertMonitoringResult,

        snapshot:
        MonitoringDiagnosticSnapshot,
    ) -> MonitoringReportSummary:

        return (
            MonitoringReportSummary(

                overall_score=
                snapshot.overall_score,

                overall_status=
                snapshot.overall_status,

                severity=
                MonitoringSeverity(

                    max(

                        runtime_result
                        .severity
                        .value,

                        health_result
                        .severity
                        .value,

                        compliance_result
                        .severity
                        .value,

                        alert_result
                        .severity
                        .value,
                    )
                ),

                total_alerts=
                alert_result
                .total_alerts,

                compliance_violations=
                compliance_result
                .violations,

                health_score=
                health_result.score,

                runtime_score=
                runtime_result.score,

                compliance_score=
                compliance_result.score,

                alert_score=
                alert_result.score,
            )
        )


# ============================================================
# SECTION BUILDER
# ============================================================

class MonitoringSectionBuilder:
    """
    Creates report sections.
    """

    # --------------------------------------------------------

    @staticmethod
    def runtime_section(
        result:
        RuntimeMonitoringResult,
    ) -> MonitoringReportSection:

        return (
            MonitoringReportSection(

                section_name=
                "RUNTIME",

                status=
                result.status,

                score=
                result.score,

                diagnostics=
                result.diagnostics,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def health_section(
        result:
        HealthMonitoringResult,
    ) -> MonitoringReportSection:

        return (
            MonitoringReportSection(

                section_name=
                "HEALTH",

                status=
                result.status,

                score=
                result.score,

                diagnostics=
                result.diagnostics,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def compliance_section(
        result:
        ComplianceMonitoringResult,
    ) -> MonitoringReportSection:

        return (
            MonitoringReportSection(

                section_name=
                "COMPLIANCE",

                status=
                result.status,

                score=
                result.score,

                diagnostics=
                result.diagnostics,
            )
        )

    # --------------------------------------------------------

    @staticmethod
    def alert_section(
        result:
        AlertMonitoringResult,
    ) -> MonitoringReportSection:

        return (
            MonitoringReportSection(

                section_name=
                "ALERTING",

                status=
                result.status,

                score=
                result.score,

                diagnostics=
                result.diagnostics,
            )
        )


# ============================================================
# REPORT DIAGNOSTICS
# ============================================================

class MonitoringReportDiagnostics:
    """
    Extra report analytics.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        trends:
        list[
            MonitoringTrend
        ],

        snapshot:
        MonitoringDiagnosticSnapshot,
    ) -> dict[str, Any]:

        upward = 0
        downward = 0
        flat = 0

        for trend in trends:

            if (
                trend.trend_direction
                == "UP"
            ):

                upward += 1

            elif (
                trend.trend_direction
                == "DOWN"
            ):

                downward += 1

            else:

                flat += 1

        return {

            "overall_score":
            snapshot.overall_score,

            "upward_trends":
            upward,

            "downward_trends":
            downward,

            "flat_trends":
            flat,

            "trend_count":
            len(
                trends
            ),
        }


# ============================================================
# REPORT BUILDER
# ============================================================

class InstitutionalMonitoringReportBuilder:
    """
    Builds final monitoring report.
    """

    # --------------------------------------------------------

    @staticmethod
    def build(
        *,
        metadata:
        MonitoringMetadata,

        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,

        alert_result:
        AlertMonitoringResult,

        snapshot:
        MonitoringDiagnosticSnapshot,

        trends:
        list[
            MonitoringTrend
        ],
    ) -> (
        InstitutionalMonitoringReport
    ):

        summary = (
            MonitoringSummaryBuilder
            .build(

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,

                snapshot=
                snapshot,
            )
        )

        diagnostics = (
            MonitoringReportDiagnostics
            .build(

                trends=
                trends,

                snapshot=
                snapshot,
            )
        )

        return (
            InstitutionalMonitoringReport(

                metadata=
                metadata,

                summary=
                summary,

                runtime_section=
                MonitoringSectionBuilder
                .runtime_section(
                    runtime_result
                ),

                health_section=
                MonitoringSectionBuilder
                .health_section(
                    health_result
                ),

                compliance_section=
                MonitoringSectionBuilder
                .compliance_section(
                    compliance_result
                ),

                alert_section=
                MonitoringSectionBuilder
                .alert_section(
                    alert_result
                ),

                diagnostic_snapshot=
                snapshot,

                trends=
                trends,

                diagnostics=
                diagnostics,
            )
        )
    

# ============================================================
# PART 12 — MASTER MONITORING ENGINE
# ============================================================

from dataclasses import dataclass
from dataclasses import field

from typing import Any


# ============================================================
# MASTER INPUT
# ============================================================

@dataclass(slots=True)
class MonitoringInput:
    """
    Unified monitoring input.

    Every monitoring subsystem
    receives data through here.
    """

    # ---------------------------------
    # Runtime Monitoring
    # ---------------------------------

    runtime_metrics: dict[str, Any] = field(
        default_factory=dict
    )

    # ---------------------------------
    # Health Monitoring
    # ---------------------------------

    component_health: dict[str, Any] = field(
        default_factory=dict
    )

    # ---------------------------------
    # Compliance Monitoring
    # ---------------------------------

    compliance_context: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# MASTER OUTPUT
# ============================================================

@dataclass(slots=True)
class MonitoringEngineResult:
    """
    Full monitoring output.
    """

    report: InstitutionalMonitoringReport

    runtime_result: RuntimeMonitoringResult

    health_result: HealthMonitoringResult

    compliance_result: ComplianceMonitoringResult

    alert_result: AlertMonitoringResult

    diagnostic_snapshot: MonitoringDiagnosticSnapshot


# ============================================================
# MASTER ENGINE
# ============================================================

class InstitutionalMonitoringEngine:
    """
    Institutional Monitoring Platform.

    Executes:

        Runtime Monitoring
        Health Monitoring
        Compliance Monitoring
        Alerting
        Diagnostics
        Reporting

    Produces:

        InstitutionalMonitoringReport
    """

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig | None = None,
    ) -> None:

        self.metadata = metadata

        self.config = config

        # ---------------------------------
        # Engines
        # ---------------------------------

        self.runtime_engine = (
            RuntimeMonitoringEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )

        self.health_engine = (
            HealthMonitoringEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )

        self.compliance_engine = (
            ComplianceMonitoringEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )

        self.alert_engine = (
            AlertingEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )

        self.diagnostics_engine = (
            MonitoringDiagnosticsEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )

    # ========================================================
    # RUNTIME STAGE
    # ========================================================

    def run_runtime(
        self,
        inputs:
        MonitoringInput,
    ) -> RuntimeMonitoringResult:

        return (
            self.runtime_engine.run(

                metrics=
                inputs.runtime_metrics
            )
        )

    # ========================================================
    # HEALTH STAGE
    # ========================================================

    def run_health(
        self,
        inputs:
        MonitoringInput,
    ) -> HealthMonitoringResult:

        return (
            self.health_engine.run(

                component_health=
                inputs
                .component_health
            )
        )

    # ========================================================
    # COMPLIANCE STAGE
    # ========================================================

    def run_compliance(
        self,
        inputs:
        MonitoringInput,
    ) -> ComplianceMonitoringResult:

        return (
            self.compliance_engine.run(

                context=
                inputs
                .compliance_context
            )
        )

    # ========================================================
    # ALERT STAGE
    # ========================================================

    def run_alerts(
        self,
        *,
        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,
    ) -> AlertMonitoringResult:

        scores = [

            runtime_result.score,

            health_result.score,

            compliance_result.score,
        ]

        overall_score = float(
            np.mean(scores)
        )

        statuses = [

            runtime_result.status,

            health_result.status,

            compliance_result.status,
        ]

        overall_status = (
            MonitoringStatus
            .PASSED
        )

        if any(
            s ==
            MonitoringStatus
            .FAILED

            for s
            in statuses
        ):

            overall_status = (
                MonitoringStatus
                .FAILED
            )

        elif any(
            s ==
            MonitoringStatus
            .WARNING

            for s
            in statuses
        ):

            overall_status = (
                MonitoringStatus
                .WARNING
            )

        return (
            self.alert_engine.run(

                score=
                overall_score,

                status=
                overall_status,

                category=
                MonitoringCategory
                .ALERTING,

                source=
                "MASTER_ENGINE",
            )
        )

    # ========================================================
    # DIAGNOSTICS STAGE
    # ========================================================

    def run_diagnostics(
        self,
        *,
        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,

        alert_result:
        AlertMonitoringResult,
    ) -> (
        MonitoringDiagnosticSnapshot
    ):

        return (
            self.diagnostics_engine.run(

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,
            )
        )

    # ========================================================
    # REPORT STAGE
    # ========================================================

    def build_report(
        self,
        *,
        runtime_result:
        RuntimeMonitoringResult,

        health_result:
        HealthMonitoringResult,

        compliance_result:
        ComplianceMonitoringResult,

        alert_result:
        AlertMonitoringResult,

        snapshot:
        MonitoringDiagnosticSnapshot,
    ) -> (
        InstitutionalMonitoringReport
    ):

        trends = (
            self.diagnostics_engine
            .score_trends()
        )

        return (
            InstitutionalMonitoringReportBuilder
            .build(

                metadata=
                self.metadata,

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,

                snapshot=
                snapshot,

                trends=
                trends,
            )
        )

    # ========================================================
    # MASTER RUN
    # ========================================================

    def run(
        self,
        inputs:
        MonitoringInput,
    ) -> (
        MonitoringEngineResult
    ):

        runtime_result = (
            self.run_runtime(
                inputs
            )
        )

        health_result = (
            self.run_health(
                inputs
            )
        )

        compliance_result = (
            self.run_compliance(
                inputs
            )
        )

        alert_result = (
            self.run_alerts(

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,
            )
        )

        snapshot = (
            self.run_diagnostics(

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,
            )
        )

        report = (
            self.build_report(

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,

                snapshot=
                snapshot,
            )
        )

        return (
            MonitoringEngineResult(

                report=
                report,

                runtime_result=
                runtime_result,

                health_result=
                health_result,

                compliance_result=
                compliance_result,

                alert_result=
                alert_result,

                diagnostic_snapshot=
                snapshot,
            )
        )
    
# ============================================================
# PART 13 — FACTORY & CONVENIENCE APIS
# ============================================================


# ============================================================
# FACTORY
# ============================================================

class MonitoringFactory:
    """
    Central factory for monitoring.

    Institutional entry point.
    """

    # --------------------------------------------------------

    @staticmethod
    def create_engine(
        *,
        metadata:
        MonitoringMetadata,

        config:
        MonitoringConfig | None = None,
    ) -> (
        InstitutionalMonitoringEngine
    ):

        return (
            InstitutionalMonitoringEngine(

                metadata=
                metadata,

                config=
                config,
            )
        )


# ============================================================
# CONVENIENCE API
# ============================================================

def create_monitoring_engine(
    *,
    metadata:
    MonitoringMetadata,

    config:
    MonitoringConfig | None = None,
) -> (
    InstitutionalMonitoringEngine
):
    """
    Convenience wrapper.
    """

    return (
        MonitoringFactory
        .create_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )


# ============================================================
# FULL MONITORING RUN
# ============================================================

def run_monitoring(
    *,
    metadata:
    MonitoringMetadata,

    monitoring_input:
    MonitoringInput,

    config:
    MonitoringConfig | None = None,
) -> (
    MonitoringEngineResult
):
    """
    One-line institutional run.

    Example
    -------

    result = run_monitoring(
        metadata=metadata,
        monitoring_input=inputs,
    )
    """

    engine = (
        create_monitoring_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )

    return engine.run(
        monitoring_input
    )


# ============================================================
# REPORT ONLY
# ============================================================

def build_monitoring_report(
    *,
    metadata:
    MonitoringMetadata,

    monitoring_input:
    MonitoringInput,

    config:
    MonitoringConfig | None = None,
) -> (
    InstitutionalMonitoringReport
):
    """
    Return report only.
    """

    result = (
        run_monitoring(

            metadata=
            metadata,

            monitoring_input=
            monitoring_input,

            config=
            config,
        )
    )

    return result.report


# ============================================================
# QUICK RUNTIME CHECK
# ============================================================

def quick_runtime_check(
    *,
    metadata:
    MonitoringMetadata,

    runtime_metrics:
    dict[str, Any],

    config:
    MonitoringConfig | None = None,
):
    """
    Runtime-only monitoring.
    """

    engine = (
        create_monitoring_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )

    monitoring_input = (
        MonitoringInput(

            runtime_metrics=
            runtime_metrics
        )
    )

    return (
        engine.run_runtime(
            monitoring_input
        )
    )


# ============================================================
# QUICK HEALTH CHECK
# ============================================================

def quick_health_check(
    *,
    metadata:
    MonitoringMetadata,

    component_health:
    dict[str, Any],

    config:
    MonitoringConfig | None = None,
):
    """
    Health-only monitoring.
    """

    engine = (
        create_monitoring_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )

    monitoring_input = (
        MonitoringInput(

            component_health=
            component_health
        )
    )

    return (
        engine.run_health(
            monitoring_input
        )
    )


# ============================================================
# QUICK COMPLIANCE CHECK
# ============================================================

def quick_compliance_check(
    *,
    metadata:
    MonitoringMetadata,

    compliance_context:
    dict[str, Any],

    config:
    MonitoringConfig | None = None,
):
    """
    Compliance-only monitoring.
    """

    engine = (
        create_monitoring_engine(

            metadata=
            metadata,

            config=
            config,
        )
    )

    monitoring_input = (
        MonitoringInput(

            compliance_context=
            compliance_context
        )
    )

    return (
        engine.run_compliance(
            monitoring_input
        )
    )


# ============================================================
# DIAGNOSTICS ONLY
# ============================================================

def latest_monitoring_snapshot(
    engine:
    InstitutionalMonitoringEngine,
):
    """
    Latest diagnostic snapshot.
    """

    return (
        engine
        .diagnostics_engine
        .registry
        .latest()
    )


# ============================================================
# TREND ANALYSIS
# ============================================================

def monitoring_trends(
    engine:
    InstitutionalMonitoringEngine,
):
    """
    Historical score trends.
    """

    return (
        engine
        .diagnostics_engine
        .score_trends()
    )


# ============================================================
# ALERT HISTORY
# ============================================================

def alert_history(
    engine:
    InstitutionalMonitoringEngine,
):
    """
    Full alert history.
    """

    return (
        engine
        .alert_engine
        .store
        .history()
    )


# ============================================================
# SYSTEM DIAGNOSTICS
# ============================================================

def monitoring_diagnostics(
    engine:
    InstitutionalMonitoringEngine,
) -> dict[str, Any]:
    """
    Engine diagnostics.
    """

    return (
        engine
        .diagnostics_engine
        .system_diagnostics()
    )