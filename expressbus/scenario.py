"""Expresslinien-Szenarien und Hebel zur Personalersparnis.

Ein Szenario besteht aus
- einer oder mehreren Expresslinien (Haltestellenfolge, Takt, Betriebszeiten),
- Hebeln: linienübergreifende Umläufe (Interlining), geteilte Dienste,
  Ausdünnung paralleler Linien in der Nebenverkehrszeit (NVZ).

compare() rechnet eine Treppe von Schritten, damit sichtbar wird, welcher Hebel wie viele Fahrer spart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from . import data
from .scheduling import Params, Result, attach_coords, plan, profile

SKIP_SAVING_MIN = 0.6  # gesparte Zeit je ausgelassener Haltestelle (Bremsen, Halt, Anfahren)
FALLBACK_SPEED_KMH = 22.0
HEADWAY_MIN, HEADWAY_MAX = 5, 120  # erlaubter Takt in Minuten (0 oder negativ würde endlos Fahrten erzeugen)


@dataclass
class ExpressLine:
    name: str
    stops: list[str]  # Haltestellennamen (Teilstring reicht, z. B. "Nordostbahnhof")
    headway: int = 20  # Takt in Minuten
    periods: list[tuple[str, str]] = field(default_factory=lambda: [("06:00", "20:00")])
    description: str = ""


PRESETS: dict[str, ExpressLine] = {
    "X35": ExpressLine(
        "X35",
        ["Röthenbach", "Gustav-Adolf-Str", "Maximilianstr", "Nordwestring", "Nordostbahnhof"],
        headway=20,
        description="Ring-Express West/Nord: Expressvariante der Ringlinie 35, hält nur an Knoten",
    ),
    "X30": ExpressLine(
        "X30",
        ["Fürth Hauptbahnhof", "Nordwestring", "Nordostbahnhof", "Nordostpark", "Flughafen"],
        headway=20,
        description="Nordring-Express: Fürth – Nordwestring – Nordostbahnhof – Nordostpark (Fraunhofer IIS) – Flughafen",
    ),
    "X45": ExpressLine(
        "X45",
        ["Nordostbahnhof", "Mögeldorf", "Doku-Zentrum", "Langwasser Mitte"],
        headway=20,
        description="Ost-Tangente: Nordostbahnhof – Mögeldorf – Doku-Zentrum – Langwasser Mitte",
    ),
    "X65": ExpressLine(
        "X65",
        ["Röthenbach", "Frankenstraße", "Klinikum Süd", "Langwasser Mitte"],
        headway=15,
        periods=[("06:00", "09:00"), ("15:00", "19:00")],
        description="Süd-Tangente, nur Hauptverkehrszeit: Röthenbach – Frankenstraße – Klinikum Süd – Langwasser Mitte",
    ),
}

OFFPEAK_WINDOWS = [("09:00", "14:00"), ("19:00", "26:00")]


@dataclass
class Levers:
    interlining: bool = True
    split_duties: bool = True
    thin_parallel: bool = False  # parallele Linien in der NVZ ausdünnen
    thin_lines: list[str] | None = None  # None = automatisch erkennen
    params: dict = field(default_factory=dict)  # weitere Params-Overrides


# ---------------------------------------------------------------- Expresslinien bauen


def _hm(s: str) -> float:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def resolve_station(name: str) -> pd.Series:
    """Findet die Station (Elternhaltestelle) mit dem meisten Verkehr, deren Name `name` enthält."""
    net = data.load()
    st = net.stops
    exact = st[st.stop_name.str.lower() == name.lower()]
    cand = exact if len(exact) else st[st.stop_name.str.contains(name, case=False, regex=False)]
    if cand.empty:
        raise ValueError(f"Haltestelle '{name}' nicht gefunden")
    usage = net.trip_stops[net.trip_stops.stop_id.isin(cand.stop_id)].stop_id.value_counts()
    best_stop = usage.index[0] if len(usage) else cand.stop_id.iloc[0]
    station = st.loc[st.stop_id == best_stop, "station"].iloc[0]
    row = st[st.station == station].copy()
    return pd.Series(
        {
            "station": station,
            "name": row.stop_name.iloc[0],
            "stop_id": best_stop,
            "lat": row.lat.mean(),
            "lon": row.lon.mean(),
            "stop_ids": list(row.stop_id),
        }
    )


def segment_time(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    """Fahrzeit zwischen zwei Express-Halten: aus dem Bestandsfahrplan (minus gesparte Halte) oder geschätzt."""
    net = data.load()
    ts = net.trip_stops
    A = ts[ts.stop_id.isin(a.stop_ids)][["trip_id", "stop_sequence", "dep"]]
    B = ts[ts.stop_id.isin(b.stop_ids)][["trip_id", "stop_sequence", "arr"]]
    m = A.merge(B, on="trip_id", suffixes=("_a", "_b"))
    m = m[m.stop_sequence_b > m.stop_sequence_a]
    dist = float(data.haversine_km(a.lat, a.lon, b.lat, b.lon))
    floor = dist * 1.2 / 40 * 60  # physikalische Untergrenze
    if len(m):
        m = m.merge(net.trips[["trip_id", "line"]], on="trip_id")
        m["tt"] = m.arr - m.dep - (m.stop_sequence_b - m.stop_sequence_a - 1) * SKIP_SAVING_MIN
        per_line = m.groupby("line").tt.median()
        return max(float(per_line.min()), floor, 1.0), f"Fahrplan Linie {per_line.idxmin()}"
    return max(dist * 1.35 / FALLBACK_SPEED_KMH * 60 + 1.0, 1.0), "geschätzt (keine Direktverbindung)"


def build_express(x: ExpressLine) -> tuple[pd.DataFrame, dict]:
    if not HEADWAY_MIN <= x.headway <= HEADWAY_MAX:
        raise ValueError(f"{x.name}: Takt {x.headway} min ungültig, erlaubt sind {HEADWAY_MIN}–{HEADWAY_MAX} min.")
    stations = [resolve_station(s) for s in x.stops]
    segs = [segment_time(a, b) for a, b in zip(stations[:-1], stations[1:])]
    seg_t = np.array([s[0] for s in segs])
    seg_km = np.array(
        [float(data.haversine_km(a.lat, a.lon, b.lat, b.lon)) * 1.25 for a, b in zip(stations[:-1], stations[1:])]
    )
    dwell = 0.5
    run_time = float(seg_t.sum() + dwell * (len(stations) - 2))
    rows = []
    for direction, seq in ((0, stations), (1, stations[::-1])):
        for p_from, p_to in x.periods:
            t = _hm(p_from)
            while t <= _hm(p_to):
                rows.append(
                    {
                        "trip_id": f"{x.name}-{direction}-{int(t):04d}",
                        "route_id": x.name,
                        "service_id": "EXPRESS",
                        "headsign": seq[-1]["name"],
                        "direction": str(direction),
                        "line": x.name,
                        "start_stop": seq[0]["stop_id"],
                        "end_stop": seq[-1]["stop_id"],
                        "start": float(t),
                        "end": float(t + run_time),
                        "n_stops": len(seq),
                        "km": float(seg_km.sum()),
                    }
                )
                t += x.headway
    trips = pd.DataFrame(rows).drop_duplicates("trip_id")
    info = {
        "name": x.name,
        "description": x.description,
        "headway": x.headway,
        "periods": x.periods,
        "stations": [{"name": s["name"], "lat": s.lat, "lon": s.lon} for s in stations],
        "segments": [
            {"von": a["name"], "nach": b["name"], "min": round(t, 1), "quelle": q}
            for a, b, (t, q) in zip(stations[:-1], stations[1:], segs)
        ],
        "fahrzeit_min": round(run_time, 1),
        "fahrten": len(trips),
        "km_pro_fahrt": round(float(seg_km.sum()), 1),
    }
    return trips, info


def parallel_lines(x_infos: list[dict]) -> list[str]:
    """Bestandslinien, die mindestens zwei Halte einer Expresslinie bedienen."""
    net = data.load()
    ts = net.trip_stops.merge(net.trips[["trip_id", "line"]], on="trip_id")
    out = set()
    for info in x_infos:
        served = {}
        for s in info["stations"]:
            st = resolve_station(s["name"])
            for line in ts[ts.stop_id.isin(st.stop_ids)].line.unique():
                served[line] = served.get(line, 0) + 1
        out |= {l for l, c in served.items() if c >= 2}
    return sorted(out, key=lambda s: (len(s), s))


def thin_offpeak(trips: pd.DataFrame, lines: list[str]) -> tuple[pd.DataFrame, int]:
    """Halbiert den Takt der angegebenen Linien in der NVZ (jede zweite Fahrt je Richtung entfällt)."""
    in_win = np.zeros(len(trips), bool)
    for a, b in OFFPEAK_WINDOWS:
        in_win |= (trips.start >= _hm(a)) & (trips.start < _hm(b))
    cand = trips[in_win & trips.line.isin(lines)].sort_values("start")
    drop = cand.groupby(["line", "direction"]).cumcount() % 2 == 1
    drop_ids = set(cand[drop].trip_id)
    return trips[~trips.trip_id.isin(drop_ids)], len(drop_ids)


# ---------------------------------------------------------------- Rechnen mit Cache

_CACHE: dict[str, Result] = {}


def _plan_cached(trips: pd.DataFrame, p: Params) -> Result:
    key = hashlib.sha1(
        (",".join(trips.trip_id) + json.dumps(p.to_dict(), sort_keys=True)).encode()
    ).hexdigest()
    if key not in _CACHE:
        net = data.load()
        _CACHE[key] = plan(attach_coords(trips, net.stops), p)
    return _CACHE[key]


DELTA_KEYS = ("fahrzeuge", "umlaeufe", "dienste", "fahrer_bedarf", "fahrplan_km", "fahrplanstunden", "arbeitsstunden")


def _delta(k: dict, base: dict) -> dict:
    return {key: round(k[key] - base[key], 1) for key in DELTA_KEYS}


def base_params(levers: Levers) -> Params:
    return replace(Params(), **levers.params)


def compare(express: list[ExpressLine], levers: Levers | None = None) -> dict:
    """Treppe: Bestand heute → +Express getrennt → +Express integriert → +Hebel."""
    levers = levers or Levers()
    net = data.load()
    base_trips = net.trips
    p0 = base_params(levers)
    p_today = replace(p0, interlining=False, split_duties=False)

    x_trips, x_infos = [], []
    for x in express:
        t, info = build_express(x)
        x_trips.append(t)
        x_infos.append(info)
    xt = pd.concat(x_trips, ignore_index=True) if x_trips else base_trips.iloc[:0]

    steps = []

    def add(name, explanation, res: Result, extra=None):
        steps.append({"schritt": name, "erklaerung": explanation, "kpi": res.kpi, "profil": profile(res, p0), **(extra or {})})

    base = _plan_cached(base_trips, p_today)
    add("Bestand heute", "Heutiger Fahrplan, linienreine Umläufe, keine geteilten Dienste (Annahme)", base)

    if len(xt):
        x_alone = _plan_cached(xt, p_today)
        additive = ("fahrten", "fahrplan_km", "fahrplanstunden", "umlaeufe", "fahrzeuge", "leerfahrt_km",
                    "dienststuecke", "dienste", "dienste_normal", "dienste_geteilt", "dienste_kurz",
                    "fahrer_bedarf", "arbeitsstunden")
        merged_kpi = dict(base.kpi)
        for k in additive:
            merged_kpi[k] = round(base.kpi[k] + x_alone.kpi[k], 1)
        merged_kpi["linien"] = base.kpi["linien"] + x_alone.kpi["linien"]
        merged_kpi["produktivitaet"] = round(merged_kpi["fahrplanstunden"] / merged_kpi["arbeitsstunden"], 3)
        pb, px = profile(base, p0), profile(x_alone, p0)
        steps.append(
            {
                "schritt": "+ Express, getrennt geplant",
                "erklaerung": "Expresslinien bekommen eigene Fahrzeuge und eigene Dienste (klassischer Ansatz)",
                "kpi": merged_kpi,
                "profil": {"t": pb["t"], **{k: [a + b for a, b in zip(pb[k], px[k])] for k in ("fahrzeuge", "fahrer")}},
            }
        )

    all_trips = pd.concat([base_trips, xt], ignore_index=True)
    p_int = replace(p0, interlining=levers.interlining, split_duties=False)
    r = _plan_cached(all_trips, p_int)
    add(
        "+ Express, integriert geplant",
        "Gemeinsame Optimierung: Expressfahrten füllen Lücken in bestehenden Umläufen"
        + (", linienübergreifende Umläufe" if levers.interlining else ""),
        r,
    )

    if levers.split_duties:
        p_split = replace(p_int, split_duties=True)
        r = _plan_cached(all_trips, p_split)
        add("+ geteilte Dienste", "Dienste mit langer Mittagspause decken Früh- und Spätspitze mit einem Fahrer ab", r)
    else:
        p_split = p_int

    thinned = []
    if levers.thin_parallel and x_infos:
        thinned = levers.thin_lines or parallel_lines(x_infos)
        t2, n_drop = thin_offpeak(all_trips, thinned)
        r = _plan_cached(t2, p_split)
        add(
            "+ Parallellinien in NVZ ausgedünnt",
            f"Linien {', '.join(thinned)}: in der Nebenverkehrszeit jede zweite Fahrt, die Expresslinie übernimmt",
            r,
            {"entfallene_fahrten": n_drop},
        )

    b = steps[0]["kpi"]
    for s in steps:
        s["delta"] = _delta(s["kpi"], b)

    # Grenzkosten: Was kostet die Expresslinie zusätzlich, wenn der Bestand schon genauso optimiert geplant wird?
    grenz = {}
    if len(xt):
        base_opt = _plan_cached(base_trips, p_split)
        grenz["optimiert"] = {
            "erklaerung": "Bestand und Bestand+Express mit denselben Hebeln geplant",
            "delta": _delta(_plan_cached(all_trips, p_split).kpi, base_opt.kpi),
        }
        if thinned:
            grenz["optimiert_mit_ausduennung"] = {
                "erklaerung": f"wie oben, zusätzlich Linien {', '.join(thinned)} in der NVZ ausgedünnt",
                "delta": _delta(steps[-1]["kpi"], base_opt.kpi),
            }
        grenz["getrennt"] = {
            "erklaerung": "Expresslinie mit eigenen Fahrzeugen und Diensten",
            "delta": _delta(steps[1]["kpi"], steps[0]["kpi"]),
        }
    return {
        "express": x_infos,
        "parallel_ausgeduennt": thinned,
        "steps": steps,
        "grenzkosten": grenz,
        "params": asdict(p_split),
        "hinweis": "Bestand heute ist eine Modellrechnung aus dem GTFS-Soll-Fahrplan, nicht die echte VAG-Dienstplanung.",
    }


def warmup() -> None:
    """Rechnet den Bestand mit den Standard-Hebeln vor (füllt den Cache)."""
    net = data.load()
    p0 = base_params(Levers())
    _plan_cached(net.trips, replace(p0, interlining=False, split_duties=False))
    _plan_cached(net.trips, p0)
