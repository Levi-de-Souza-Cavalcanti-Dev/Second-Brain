"""OpenTelemetry bootstrap (optional)."""

from __future__ import annotations

from secondbrain.config import Settings


def setup_otel(settings: Settings) -> None:
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if not endpoint:
        return
    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        return

    resource = Resource.create({"service.name": "secondbrain"})
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
