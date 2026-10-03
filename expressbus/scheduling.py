"""Umlauf- und Dienstplanung für einen Betriebstag.

1. Umlaufplanung (Vehicle Scheduling): Fahrten werden per Min-Cost-Flow zu Fahrzeugumläufen verkettet.
   Ziel: möglichst wenige Umläufe, danach möglichst wenig Standzeit und Leerfahrt.
2. Dienstplanung (Crew Scheduling): Umläufe werden an Ablösepunkten in Dienststücke geschnitten
   (dynamische Programmierung, mehrere alternative Schnittmuster), danach paart CP-SAT die Stücke zu Diensten
   mit Pause. Jedes Muster wird parallel exakt gepaart, das beste gewinnt (Multistart); optional verbessert ein
   gemeinsames Modell Schnitt und Paarung zusammen (crew_mode="joint").
   Ziel: möglichst wenige Dienste, also möglichst wenige Fahrer.

Vereinfachungen (bewusst, für den Hackathon): Leerfahrzeiten aus Luftlinie × Umwegfaktor,
ein virtuelles Depot mit fester Aus- und Einrückzeit, höchstens zwei Stücke pro Dienst,
Regeln (Arbeitszeit, Pausen, Lenkzeit) als Parameter, nicht als vollständiges Tarifwerk.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from ortools.graph.python import min_cost_flow
from ortools.sat.python import cp_model

from .data import haversine_km

DEPOT = "DEPOT"


@dataclass
class Params:
    # Umlaufplanung
    min_layover: float = 4.0  # Mindestwendezeit zwischen zwei Fahrten (min)
    max_wait: float = 90.0  # längere Standzeit → Fahrzeug fährt ins Depot (neuer Umlauf)
    deadhead_speed_kmh: float = 22.0
    detour_factor: float = 1.3
    deadhead_fixed: float = 2.0
    max_deadhead_km: float = 6.0
    interlining: bool = True  # Umläufe dürfen die Linie wechseln
    pull_out: float = 15.0  # Ausrücken aus dem Depot (min)
    pull_in: float = 15.0  # Einrücken ins Depot (min)
    # Dienstplanung
    max_piece: float = 270.0  # max. Länge eines Dienststücks ohne Pause (4,5 h Lenkzeit)
    target_piece: float = 240.0
    min_break: float = 30.0
    max_work: float = 540.0  # max. Arbeitszeit ohne Pause (9 h)
    max_spread: float = 600.0  # max. Schichtlänge normaler Dienst (10 h)
    split_duties: bool = True  # geteilte Dienste mit langer Pause zulassen
    split_min_break: float = 120.0
    max_spread_split: float = 750.0  # 12,5 h
    sign_on: float = 10.0  # Dienstbeginn (Fahrzeugübernahme, Kontrolle)
    sign_off: float = 5.0
    relief_radius_km: float = 0.3  # Ablösung nur, wenn Ankunft und nächste Abfahrt so nah beieinander liegen
    depot_travel: float = 20.0  # Wegezeit Depot ↔ Ablösepunkt
    fte_per_duty: float = 1.7  # Fahrer (Köpfe) je täglichem Dienst (7-Tage-Betrieb, Urlaub, Krankheit)
    time_limit_s: float = 12.0  # je Schnittmuster im Multistart
    crew_mode: str = "multistart"  # "joint" = zusätzlich gemeinsames Modell (gründlicher, langsamer)
    joint_time_limit_s: float = 60.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Result:
    trips: pd.DataFrame
    blocks: list[list[int]]
    pieces: pd.DataFrame
    duties: pd.DataFrame
    kpi: dict = field(default_factory=dict)


# ---------------------------------------------------------------- Hilfsfunktionen


def _deadhead_min(d_km: np.ndarray, p: Params) -> np.ndarray:
    return np.where(d_km < 0.05, 0.0, p.deadhead_fixed + d_km * p.detour_factor / p.deadhead_speed_kmh * 60)


def attach_coords(trips: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    s = stops.set_index("stop_id")
    t = trips.copy()
    t["slat"] = t.start_stop.map(s.lat)
    t["slon"] = t.start_stop.map(s.lon)
    t["elat"] = t.end_stop.map(s.lat)
    t["elon"] = t.end_stop.map(s.lon)
    return t


# ---------------------------------------------------------------- Umlaufplanung


def schedule_vehicles(trips: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, list[list[int]], dict]:
    t = trips.sort_values("start").reset_index(drop=True)
    n = len(t)
    start, end = t.start.to_numpy(), t.end.to_numpy()
    slat, slon, elat, elon = (t[c].to_numpy() for c in ("slat", "slon", "elat", "elon"))
    line = t.line.to_numpy()

    tails, heads, costs, dists = [], [], [], []
    for i in range(n):
        lo = np.searchsorted(start, end[i] + p.min_layover, "left")
        hi = np.searchsorted(start, end[i] + p.max_wait, "right")
        if lo >= hi:
            continue
        j = np.arange(lo, hi)
        if not p.interlining:
            j = j[line[j] == line[i]]
            if not len(j):
                continue
        d = haversine_km(elat[i], elon[i], slat[j], slon[j])
        dh = _deadhead_min(d, p)
        ok = (d <= p.max_deadhead_km) & (end[i] + dh + p.min_layover <= start[j])
        j, d, dh = j[ok], d[ok], dh[ok]
        wait = start[j] - end[i]
        c = wait + 2 * dh + np.where(line[j] == line[i], 0, 3)
        tails.append(np.full(len(j), i))
        heads.append(j)
        costs.append(c)
        dists.append(d)

    tails = np.concatenate(tails) if tails else np.array([], int)
    heads = np.concatenate(heads) if heads else np.array([], int)
    costs = np.concatenate(costs) if costs else np.array([])
    dists = np.concatenate(dists) if dists else np.array([])

    # Knoten: Ende(i) = i, Start(j) = n + j, Depot = 2n
    K = 100_000  # Kosten je Umlauf → Umlaufzahl wird zuerst minimiert
    depot = 2 * n
    mcf = min_cost_flow.SimpleMinCostFlow()
    all_t = np.concatenate([tails, np.arange(n), np.full(n, depot)])
    all_h = np.concatenate([heads + n, np.full(n, depot), np.arange(n) + n])
    all_c = np.concatenate([np.round(costs * 10), np.full(n, K // 2), np.full(n, K // 2)]).astype(np.int64)
    mcf.add_arcs_with_capacity_and_unit_cost(all_t, all_h, np.ones(len(all_t), np.int64), all_c)
    supplies = np.concatenate([np.ones(n), -np.ones(n), [0]]).astype(np.int64)
    mcf.set_nodes_supplies(np.arange(2 * n + 1), supplies)
    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError(f"Umlaufplanung nicht lösbar (Status {status})")

    flows = mcf.flows(np.arange(len(tails)))
    used = flows > 0
    succ = np.full(n, -1)
    succ[tails[used]] = heads[used]
    has_pred = np.zeros(n, bool)
    has_pred[heads[used]] = True
    deadhead_km = float((dists[used] * p.detour_factor).sum())

    blocks = []
    for s in np.where(~has_pred)[0]:
        chain, k = [], s
        while k != -1:
            chain.append(int(k))
            k = succ[k]
        blocks.append(chain)
    blocks.sort(key=lambda b: start[b[0]])

    t["block"] = -1
    for b_id, b in enumerate(blocks):
        t.loc[b, "block"] = b_id

    # Fahrzeugbedarf = maximale Zahl gleichzeitig laufender Umläufe (inkl. Aus-/Einrücken)
    ev = []
    for b in blocks:
        ev.append((start[b[0]] - p.pull_out, 1))
        ev.append((end[b[-1]] + p.pull_in, -1))
    ev.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    peak_time = 0.0
    for tm, d in ev:
        cur += d
        if cur > peak:
            peak, peak_time = cur, tm

    stats = {
        "umlaeufe": len(blocks),
        "fahrzeuge": int(peak),
        "spitzenzeit": _hhmm(peak_time),
        "leerfahrt_km": round(deadhead_km, 1),
        "linienwechsel": int((line[tails[used]] != line[heads[used]]).sum()),
    }
    return t, blocks, stats


# ---------------------------------------------------------------- Dienstplanung


def _cut_patterns(p: Params) -> list[tuple[float, float]]:
    """(max. Stücklänge, Ziel-Stücklänge) für alternative Schnittmuster je Umlauf."""
    M = p.max_piece
    return [(M, M - 30), (M, M), (M - 30, M - 45)]


def cut_pieces(t: pd.DataFrame, blocks: list[list[int]], p: Params) -> tuple[pd.DataFrame, list[list[list[int]]]]:
    """Schneidet jeden Umlauf an Ablösepunkten in Dienststücke.

    Pro Umlauf entstehen mehrere alternative Schnittmuster (DP mit unterschiedlicher Ziel-Stücklänge).
    Welches Muster genommen wird, entscheidet später CP-SAT zusammen mit der Paarung zu Diensten.
    Rückgabe: Stücke (dedupliziert) und je Umlauf ein Muster pro Schnittvariante (Listen von Stück-Indizes).
    """
    start, end = t.start.to_numpy(), t.end.to_numpy()
    slat, slon, elat, elon = (t[c].to_numpy() for c in ("slat", "slon", "elat", "elon"))
    e_stop = t.end_stop.to_numpy()
    line = t.line.to_numpy()
    rows, key_to_idx, patterns = [], {}, []
    for b_id, b in enumerate(blocks):
        k = len(b)
        # Positionen 0..k: 0 = Ausrücken, k = Einrücken, m = nach Fahrt b[m-1]
        T = np.empty(k + 1)
        loc = [DEPOT] * (k + 1)
        lat = np.full(k + 1, np.nan)
        lon = np.full(k + 1, np.nan)
        valid = np.zeros(k + 1, bool)
        T[0], T[k] = start[b[0]] - p.pull_out, end[b[-1]] + p.pull_in
        valid[0] = valid[k] = True
        for m in range(1, k):
            a, nxt = b[m - 1], b[m]
            T[m] = end[a]
            loc[m] = e_stop[a]
            lat[m], lon[m] = elat[a], elon[a]
            valid[m] = haversine_km(elat[a], elon[a], slat[nxt], slon[nxt]) <= p.relief_radius_km
        vpos = np.where(valid)[0]
        block_patterns = []
        for cap, target in _cut_patterns(p):
            best = {0: (0.0, -1)}
            for bi in vpos[1:]:
                cand = []
                for ai in vpos[vpos < bi]:
                    L = T[bi] - T[ai]
                    if L <= cap:
                        cand.append((best[ai][0] + 1000 + max(0.0, target - L) * 2, ai))
                prev = vpos[vpos < bi][-1]
                if not cand:  # kein zulässiger Schnitt → Stück wird zu lang (als Verstoß markiert)
                    cand.append((best[prev][0] + 1e6, prev))
                best[bi] = min(cand)
            cuts, cur = [], k
            while cur != 0:
                cuts.append(cur)
                cur = best[cur][1]
            cuts = [0] + cuts[::-1]
            idxs = []
            for a_pos, b_pos in zip(cuts[:-1], cuts[1:]):
                key = (b_id, int(a_pos), int(b_pos))
                if key not in key_to_idx:
                    trip_idx = b[a_pos:b_pos]
                    key_to_idx[key] = len(rows)
                    rows.append(
                        {
                            "block": b_id,
                            "start": T[a_pos],
                            "end": T[b_pos],
                            "start_loc": loc[a_pos],
                            "end_loc": loc[b_pos],
                            "slat": lat[a_pos],
                            "slon": lon[a_pos],
                            "elat": lat[b_pos],
                            "elon": lon[b_pos],
                            "drive": float((end[trip_idx] - start[trip_idx]).sum()),
                            "n_trips": len(trip_idx),
                            "lines": ",".join(sorted(set(line[trip_idx]), key=lambda s: (len(s), s))),
                            "too_long": (T[b_pos] - T[a_pos]) > p.max_piece + 1e-6,
                        }
                    )
                idxs.append(key_to_idx[key])
            block_patterns.append(idxs)
        patterns.append(block_patterns)
    pieces = pd.DataFrame(rows)
    pieces["length"] = pieces.end - pieces.start
    pieces["piece"] = np.arange(len(pieces))
    return pieces, patterns


def _travel(loc_a, lat_a, lon_a, loc_b, lat_b, lon_b, p: Params) -> np.ndarray:
    """Wegezeit eines Fahrers zwischen Ende eines Stücks und Beginn des nächsten."""
    d = haversine_km(lat_a, lon_a, lat_b, lon_b)
    tt = np.where(d <= p.relief_radius_km, 2.0, 5.0 + d * 1.4 / 15 * 60)
    a_dep = loc_a == DEPOT
    b_dep = loc_b == DEPOT
    tt = np.where(a_dep & b_dep, 0.0, tt)
    tt = np.where(a_dep ^ b_dep, p.depot_travel, tt)
    return tt


def _pair_candidates(P: pd.DataFrame, p: Params, active: np.ndarray) -> dict:
    """Zulässige Paare (Stück a, Pause, Stück b) unter den aktiven Stücken, mit Arbeitszeit, Pause, Schichtlänge."""
    n = len(P)
    s, e, L = P.start.to_numpy(), P.end.to_numpy(), P.length.to_numpy()
    sloc, eloc = P.start_loc.to_numpy(), P.end_loc.to_numpy()
    slat, slon, elat, elon = (P[c].to_numpy() for c in ("slat", "slon", "elat", "elon"))
    wege_start = np.where(sloc == DEPOT, 0.0, p.depot_travel)  # Anreise zum Ablösepunkt
    wege_end = np.where(eloc == DEPOT, 0.0, p.depot_travel)
    max_spread = p.max_spread_split if p.split_duties else p.max_spread

    pa, pb, pcost, pinfo = [], [], [], []
    for a in np.where(active)[0]:
        j = np.where(active & (s >= e[a] + p.min_break) & (s <= s[a] + max_spread))[0]
        if not len(j):
            continue
        tt = _travel(eloc[a], elat[a], elon[a], sloc[j], slat[j], slon[j], p)
        brk = s[j] - e[a] - tt
        work = p.sign_on + wege_start[a] + L[a] + tt + L[j] + wege_end[j] + p.sign_off
        spread = e[j] + wege_end[j] + p.sign_off - (s[a] - wege_start[a] - p.sign_on)
        ok = (brk >= p.min_break) & (work <= p.max_work)
        normal = spread <= p.max_spread
        split = p.split_duties & (brk >= p.split_min_break) & (spread <= p.max_spread_split)
        ok &= normal | split
        j = j[ok]
        if not len(j):
            continue
        sp, wk, br, is_split = spread[ok], work[ok], brk[ok], ~normal[ok]
        # Modellgröße begrenzen: je Stück die besten normalen und die besten geteilten Partner behalten
        idx_n = np.where(~is_split)[0]
        idx_s = np.where(is_split)[0]
        keep = np.concatenate([idx_n[np.argsort(sp[idx_n])[:40]], idx_s[np.argsort(sp[idx_s])[:40]]])
        pa.append(np.full(len(keep), a))
        pb.append(j[keep])
        pcost.append(sp[keep])
        pinfo.append(np.stack([wk[keep], br[keep], is_split[keep]], axis=1))
    cat = lambda xs, empty: np.concatenate(xs) if xs else empty
    return {
        "a": cat(pa, np.array([], int)),
        "b": cat(pb, np.array([], int)),
        "cost": cat(pcost, np.array([])),
        "info": cat(pinfo, np.empty((0, 3))),
        "single_work": p.sign_on + wege_start + L + wege_end + p.sign_off,
        "wege_start": wege_start,
        "wege_end": wege_end,
    }


W_DUTY = 10_000  # Kosten je Dienst → Dienstzahl wird zuerst minimiert, danach Schichtlänge


def _solve(n: int, C: dict, patterns, active: np.ndarray | None, p: Params, time_limit: float,
           workers: int, hint: dict | None = None):
    """CP-SAT: Stücke zu Diensten paaren.

    active=None → gemeinsames Modell: je Umlauf wird eines seiner Schnittmuster gewählt.
    active=Maske → festes Schnittmuster: genau die aktiven Stücke müssen abgedeckt werden.
    """
    pa, pb = C["a"], C["b"]
    sel = np.arange(len(pa)) if active is None else np.where(active[pa] & active[pb])[0]
    nodes = np.arange(n) if active is None else np.where(active)[0]
    m = cp_model.CpModel()
    x = {k: m.new_bool_var(f"x{k}") for k in sel}
    y = {i: m.new_bool_var(f"y{i}") for i in nodes}
    cover = {i: [y[i]] for i in nodes}
    for k in sel:
        cover[pa[k]].append(x[k])
        cover[pb[k]].append(x[k])
    z = []
    if active is None:
        z_of_piece = {i: [] for i in nodes}
        for b_id, pats in enumerate(patterns):
            uniq = {tuple(pt) for pt in pats}
            zs = []
            for pt in sorted(uniq):
                zv = m.new_bool_var(f"z{b_id}_{len(zs)}")
                zs.append(zv)
                z.append((zv, pt))
                for i in pt:
                    z_of_piece[i].append(zv)
            m.add_exactly_one(zs)
        for i in nodes:
            m.add(sum(cover[i]) == sum(z_of_piece[i]))
    else:
        for i in nodes:
            m.add_exactly_one(cover[i])
    m.minimize(
        sum(int(W_DUTY + C["cost"][k]) * x[k] for k in sel)
        + sum(int(W_DUTY + C["single_work"][i] + 60) * y[i] for i in nodes)  # Kurzdienste leicht bestrafen
    )
    if hint:
        for k, v in x.items():
            m.add_hint(v, k in hint["x"])
        for i, v in y.items():
            m.add_hint(v, i in hint["y"])
        for zv, pt in z:
            m.add_hint(zv, set(pt) <= hint["pieces"])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    xs = {k for k, v in x.items() if solver.value(v)}
    ys = {i for i, v in y.items() if solver.value(v)}
    pieces_used = {int(pa[k]) for k in xs} | {int(pb[k]) for k in xs} | {int(i) for i in ys}
    return {
        "x": xs,
        "y": ys,
        "pieces": pieces_used,
        "duties": len(xs) + len(ys),
        "objective": solver.objective_value,
        "status": solver.status_name(status),
    }


def pair_pieces(pieces: pd.DataFrame, patterns: list[list[list[int]]], p: Params) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Multistart: jedes Schnittmuster einzeln exakt paaren (parallel), das beste gewinnt.
    Mit p.crew_mode == "joint" verbessert danach ein gemeinsames Modell (Startlösung = bestes Muster)."""
    P = pieces
    n = len(P)
    t0 = time.time()
    n_cfg = len(patterns[0]) if patterns else 0

    def run_cfg(c):
        active = np.zeros(n, bool)
        for pats in patterns:
            active[pats[c]] = True
        C = _pair_candidates(P, p, active)
        r = _solve(n, C, patterns, active, p, p.time_limit_s, workers=4)
        return (r, C) if r else None

    with ThreadPoolExecutor(max_workers=n_cfg) as ex:
        results = [r for r in ex.map(run_cfg, range(n_cfg)) if r]
    if not results:
        raise RuntimeError("Dienstplanung ohne Lösung")
    best, C = min(results, key=lambda rc: rc[0]["objective"])
    multistart = [r["duties"] for r, _ in results]
    if p.crew_mode == "joint":
        CJ = _pair_candidates(P, p, np.ones(n, bool))
        idx = {(a, b): k for k, (a, b) in enumerate(zip(CJ["a"], CJ["b"]))}
        hint_x = {idx.get((C["a"][k], C["b"][k])) for k in best["x"]}
        if None not in hint_x:  # Startlösung ist im gemeinsamen Modell darstellbar
            r = _solve(n, CJ, patterns, None, p, p.joint_time_limit_s, workers=8, hint={**best, "x": hint_x})
            if r and r["objective"] <= best["objective"]:
                best, C = r, CJ

    s, e = P.start.to_numpy(), P.end.to_numpy()
    ws, we = C["wege_start"], C["wege_end"]
    rows = []
    for k in best["x"]:
        a, b = int(C["a"][k]), int(C["b"][k])
        wk, br, is_split = C["info"][k]
        rows.append(
            {
                "pieces": [a, b],
                "start": s[a] - ws[a] - p.sign_on,
                "end": e[b] + we[b] + p.sign_off,
                "work": wk,
                "break": br,
                "type": "geteilt" if is_split else "normal",
                "lines": ",".join(sorted(set(P.lines[a].split(",")) | set(P.lines[b].split(",")), key=lambda q: (len(q), q))),
            }
        )
    for i in best["y"]:
        rows.append(
            {
                "pieces": [int(i)],
                "start": s[i] - ws[i] - p.sign_on,
                "end": e[i] + we[i] + p.sign_off,
                "work": C["single_work"][i],
                "break": 0.0,
                "type": "kurz",
                "lines": P.lines[i],
            }
        )
    duties = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    duties["spread"] = duties.end - duties.start
    stats = {
        "solver_status": best["status"],
        "solver_s": round(time.time() - t0, 1),
        "paar_kandidaten": int(len(C["a"])),
        "multistart_dienste": multistart,
    }
    return duties, P.loc[sorted(best["pieces"])], stats


# ---------------------------------------------------------------- Gesamtlauf


def plan(trips: pd.DataFrame, p: Params | None = None) -> Result:
    p = p or Params()
    t, blocks, vstats = schedule_vehicles(trips, p)
    cand_pieces, patterns = cut_pieces(t, blocks, p)
    duties, pieces, cstats = pair_pieces(cand_pieces, patterns, p)
    fahrstunden = float((t.end - t.start).sum() / 60)
    work_h = float(duties.work.sum() / 60)
    kpi = {
        "fahrten": int(len(t)),
        "linien": int(t.line.nunique()),
        "fahrplan_km": round(float(t.km.sum()), 1),
        "fahrplanstunden": round(fahrstunden, 1),
        **vstats,
        "dienststuecke": int(len(pieces)),
        "stuecke_zu_lang": int(pieces.too_long.sum()),
        "dienste": int(len(duties)),
        "dienste_normal": int((duties.type == "normal").sum()),
        "dienste_geteilt": int((duties.type == "geteilt").sum()),
        "dienste_kurz": int((duties.type == "kurz").sum()),
        "fahrer_bedarf": round(len(duties) * p.fte_per_duty, 1),
        "arbeitsstunden": round(work_h, 1),
        "produktivitaet": round(fahrstunden / work_h, 3) if work_h else None,
        **cstats,
    }
    return Result(trips=t, blocks=blocks, pieces=pieces, duties=duties, kpi=kpi)


PROFILE_START, PROFILE_END, PROFILE_STEP = 240, 1560, 15  # 04:00 bis 02:00 in 15-Minuten-Schritten


def profile(r: Result, p: Params | None = None) -> dict:
    """Fahrzeuge im Einsatz und Fahrer am Steuer über den Tag (für das Tagesprofil-Diagramm)."""
    p = p or Params()
    grid = np.arange(PROFILE_START, PROFILE_END, PROFILE_STEP)
    st, en = r.trips.start.to_numpy(), r.trips.end.to_numpy()
    b_start = np.array([st[b[0]] - p.pull_out for b in r.blocks])
    b_end = np.array([en[b[-1]] + p.pull_in for b in r.blocks])
    veh = ((b_start[None, :] <= grid[:, None]) & (b_end[None, :] > grid[:, None])).sum(1)
    ps, pe = r.pieces.start.to_numpy(), r.pieces.end.to_numpy()
    drv = ((ps[None, :] <= grid[:, None]) & (pe[None, :] > grid[:, None])).sum(1)
    return {"t": [_hhmm(g) for g in grid], "fahrzeuge": veh.tolist(), "fahrer": drv.tolist()}


def _hhmm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}"
