"""Shared utilities and mock contracts for TranSmart AI."""

from transport_ai.common.mocks import (
    ALERT_CLASSES,
    DESTINATIONS,
    DRIVER_CLASSES,
    DRIVER_CLASS_LABELS,
    MAX_IMAGE_BYTES,
    MODEL_METADATA,
    ROUTES,
    USER_HISTORIES,
    classify_driver,
    forecast,
    generate_sample_driver_image,
    historical_demand,
    plan_operations,
    recommend,
)

__all__ = [
    "ALERT_CLASSES",
    "DESTINATIONS",
    "DRIVER_CLASSES",
    "DRIVER_CLASS_LABELS",
    "MAX_IMAGE_BYTES",
    "MODEL_METADATA",
    "ROUTES",
    "USER_HISTORIES",
    "classify_driver",
    "forecast",
    "generate_sample_driver_image",
    "historical_demand",
    "plan_operations",
    "recommend",
]
