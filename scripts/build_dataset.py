"""Baut aus dem VGN-GTFS-Feed den Datensatz für die VAG-Stadtbusse an einem Stichtag.

Aufruf: uv run python scripts/build_dataset.py [YYYYMMDD]
Ergebnis in data/processed/: trips.parquet, trip_stops.parquet, stops.parquet, lines.parquet, meta.json

Daten: VGN – Verkehrsverbund Großraum Nürnberg GmbH, CC BY 3.0 DE.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "vgn_gtfs"
OUT = ROOT / "data" / "processed"

# Präfix der route_id im VGN-Feed: 13 = VAG-Stadtbus Nürnberg (11 = U-Bahn, 12 = Tram, 16 = NightLiner)
BUS_PREFIX = "13"
EXCLUDE_DESC = {"Linientaxi", "Rufbus"}


def read(name, **kw):
    return pd.read_csv(RAW / name, dtype=str, encoding="utf-8-sig", **kw)


def to_min(s: pd.Series) -> pd.Series:
    hms = s.str.split(":", expand=True).astype(int)
    return hms[0] * 60 + hms[1] + hms[2] / 60


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def active_services(day: str) -> set:
    weekday = pd.Timestamp(day).day_name().lower()
    cal = read("calendar.txt")
    cd = read("calendar_dates.txt")
    act = set(cal[(cal[weekday] == "1") & (cal.start_date <= day) & (cal.end_date >= day)].service_id)
    act |= set(cd[(cd.date == day) & (cd.exception_type == "1")].service_id)
    act -= set(cd[(cd.date == day) & (cd.exception_type == "2")].service_id)
    return act


def main(day: str = "20261005"):
    OUT.mkdir(parents=True, exist_ok=True)
    routes = read("routes.txt")
    routes = routes[
        routes.route_id.str.startswith(BUS_PREFIX + "-")
        & (routes.route_type == "3")
        & ~routes.route_desc.isin(EXCLUDE_DESC)
    ]
    trips = read("trips.txt", usecols=["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"])
    trips = trips[trips.route_id.isin(routes.route_id) & trips.service_id.isin(active_services(day))]
    trips = trips.merge(routes[["route_id", "route_short_name"]], on="route_id").rename(
        columns={"route_short_name": "line", "trip_headsign": "headsign", "direction_id": "direction"}
    )

    st = pd.read_csv(
        RAW / "stop_times.txt",
        encoding="utf-8-sig",
        usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
        dtype={"trip_id": str, "arrival_time": str, "departure_time": str, "stop_id": str, "stop_sequence": int},
        engine="pyarrow",
    )
    st = st[st.trip_id.isin(set(trips.trip_id))].copy()
    st["arr"] = to_min(st.arrival_time)
    st["dep"] = to_min(st.departure_time)
    st = st.drop(columns=["arrival_time", "departure_time"]).sort_values(["trip_id", "stop_sequence"])

    stops = read("stops.txt", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"])
    stops = stops[stops.stop_id.isin(set(st.stop_id))].copy()
    stops["lat"] = stops.stop_lat.astype(float)
    stops["lon"] = stops.stop_lon.astype(float)
    stops["station"] = stops.parent_station.fillna("").str.replace("Parent", "", regex=False)
    stops.loc[stops.station == "", "station"] = stops.stop_id
    stops = stops[["stop_id", "stop_name", "lat", "lon", "station"]]

    st = st.merge(stops[["stop_id", "lat", "lon"]], on="stop_id")
    st = st.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    same = st.trip_id.eq(st.trip_id.shift())
    seg = haversine_km(st.lat.shift(), st.lon.shift(), st.lat, st.lon)
    st["seg_km"] = np.where(same, seg, 0.0)

    g = st.groupby("trip_id")
    agg = pd.DataFrame(
        {
            "start_stop": g.stop_id.first(),
            "end_stop": g.stop_id.last(),
            "start": g.dep.first(),
            "end": g.arr.last(),
            "n_stops": g.size(),
            "km": g.seg_km.sum(),
        }
    ).reset_index()
    trips = trips.merge(agg, on="trip_id")
    trips = trips[trips.end > trips.start].sort_values("start").reset_index(drop=True)

    names = stops.set_index("stop_id").stop_name
    lines = (
        trips.assign(start_name=trips.start_stop.map(names), end_name=trips.end_stop.map(names))
        .groupby("line")
        .agg(
            trips=("trip_id", "size"),
            first=("start", "min"),
            last=("end", "max"),
            km=("km", "sum"),
            hours=("end", lambda e: (e - trips.loc[e.index, "start"]).sum() / 60),
            termini=("end_name", lambda s: " / ".join(s.value_counts().index[:2])),
        )
        .reset_index()
    )

    trips.to_parquet(OUT / "trips.parquet", index=False)
    st[st.trip_id.isin(set(trips.trip_id))][
        ["trip_id", "stop_sequence", "stop_id", "arr", "dep", "seg_km"]
    ].to_parquet(OUT / "trip_stops.parquet", index=False)
    stops.to_parquet(OUT / "stops.parquet", index=False)
    lines.to_parquet(OUT / "lines.parquet", index=False)
    meta = {
        "service_day": day,
        "source": "VGN GTFS (www.vgn.de/opendata/GTFS.zip), CC BY 3.0 DE – VGN – Verkehrsverbund Großraum Nürnberg GmbH",
        "scope": "VAG-Stadtbus Nürnberg (ohne Rufbus/Linientaxi, ohne NightLiner)",
        "trips": int(len(trips)),
        "lines": int(trips.line.nunique()),
        "stops": int(len(stops)),
        "fahrplan_km": round(float(trips.km.sum()), 1),
        "fahrplan_stunden": round(float((trips.end - trips.start).sum() / 60), 1),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
