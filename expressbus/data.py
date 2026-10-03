"""Lädt den aufbereiteten Datensatz (siehe scripts/build_dataset.py)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


@dataclass(frozen=True)
class Network:
    trips: pd.DataFrame  # trip_id, line, direction, headsign, start_stop, end_stop, start, end, n_stops, km
    trip_stops: pd.DataFrame  # trip_id, stop_sequence, stop_id, arr, dep, seg_km
    stops: pd.DataFrame  # stop_id, stop_name, lat, lon, station
    lines: pd.DataFrame
    meta: dict

    def stop_coords(self, stop_ids) -> tuple[np.ndarray, np.ndarray]:
        s = self.stops.set_index("stop_id")
        idx = pd.Index(stop_ids)
        return s.lat.reindex(idx).to_numpy(), s.lon.reindex(idx).to_numpy()


@lru_cache(maxsize=1)
def load() -> Network:
    return Network(
        trips=pd.read_parquet(PROCESSED / "trips.parquet"),
        trip_stops=pd.read_parquet(PROCESSED / "trip_stops.parquet"),
        stops=pd.read_parquet(PROCESSED / "stops.parquet"),
        lines=pd.read_parquet(PROCESSED / "lines.parquet"),
        meta=json.loads((PROCESSED / "meta.json").read_text()),
    )


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
