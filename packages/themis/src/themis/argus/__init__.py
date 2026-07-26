"""Argus — telemetry and run diagnosis inside Themis.

Metrics as JSONL first (loss / LR / throughput / cache-stats), designed for
post-hoc attribution. Earns its own package only if it outgrows this module.
Schema: docs/argus-telemetry-schema.md (authored in the Rig Design phase).
"""
