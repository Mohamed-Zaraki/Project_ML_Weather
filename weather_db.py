"""weather_db.py - backward-compatible alias for :mod:`data`.

New code should use ``from data import DataDB``.  ``WeatherDB`` is kept so the
existing Streamlit view keeps working: it lists only the weather report(s),
while ``DataDB`` exposes every dataset stored in ``data.db``.
"""
from __future__ import annotations

from data import DEFAULT_DB, DataDB


class WeatherDB(DataDB):
    """Compatibility wrapper that filters report listings to weather runs."""

    def list_reports(self, dataset=None, dataset_like=None):
        if dataset is None and dataset_like is None:
            dataset_like = "weather%"
        return super().list_reports(dataset=dataset, dataset_like=dataset_like)

    def latest_report_id(self, dataset=None):
        rows = self.list_reports(dataset=dataset)
        return rows[0]["report_id"] if rows else None


__all__ = ["WeatherDB", "DataDB", "DEFAULT_DB"]
