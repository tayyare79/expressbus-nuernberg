"""Web-Backend für die Demo. Start: uv run uvicorn expressbus.api:app --port 8000"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data, explainer, scenario
from .data import ROOT

app = FastAPI(title="Expressbus-Planer Nürnberg")
WEB = ROOT / "web"


# ---------------------------------------------------------------- Netz und Stammdaten


def _network_geojson() -> dict:
    """Je Linie der häufigste Laufweg (Richtung 0) als Linienzug."""
    net = data.load()
    t = net.trips[net.trips.direction == "0"]
    ts = net.trip_stops[net.trip_stops.trip_id.isin(t.trip_id)]
    seq = ts.groupby("trip_id").stop_id.agg(tuple)
    seq = seq.to_frame("seq").join(t.set_index("trip_id").line)
    coords = net.stops.set_index("stop_id")[["lon", "lat"]]
    feats = []
    for line, g in seq.groupby("line"):
        pattern = g.seq.value_counts().index[0]
        feats.append(
            {
                "type": "Feature",
                "properties": {"line": line},
                "geometry": {"type": "LineString", "coordinates": coords.loc[list(pattern)].values.round(5).tolist()},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


_NETWORK = None


@app.get("/api/meta")
def meta():
    net = data.load()
    presets = []
    for x in scenario.PRESETS.values():
        _, info = scenario.build_express(x)
        presets.append({**asdict(x), "info": info})
    stop_names = sorted(net.stops.stop_name.unique())
    return {"meta": net.meta, "presets": presets, "stop_names": stop_names, "claude": _claude_available()}


@app.get("/api/network")
def network():
    global _NETWORK
    if _NETWORK is None:
        _NETWORK = _network_geojson()
    return _NETWORK


# ---------------------------------------------------------------- Szenarien


class ExpressIn(BaseModel):
    name: str
    stops: list[str] = Field(min_length=2)
    headway: int = Field(20, ge=scenario.HEADWAY_MIN, le=scenario.HEADWAY_MAX)
    periods: list[tuple[str, str]] = [("06:00", "20:00")]


class ScenarioIn(BaseModel):
    presets: list[str] = []
    express_lines: list[ExpressIn] = []
    interlining: bool = True
    split_duties: bool = True
    thin_parallel: bool = False
    thin_lines: list[str] | None = None
    fte_per_duty: float = 1.7


_LAST_SCENARIO: dict = {}
_LOCK = threading.Lock()  # Optimierer nutzt alle Kerne – Szenarien nacheinander rechnen


def _run(req: ScenarioIn) -> dict:
    lines = [scenario.PRESETS[p] for p in req.presets if p in scenario.PRESETS]
    lines += [scenario.ExpressLine(x.name, x.stops, x.headway, [tuple(p) for p in x.periods]) for x in req.express_lines]
    if not lines:
        raise HTTPException(400, "Bitte mindestens eine Expresslinie wählen.")
    levers = scenario.Levers(
        interlining=req.interlining,
        split_duties=req.split_duties,
        thin_parallel=req.thin_parallel,
        thin_lines=req.thin_lines,
        params={"fte_per_duty": req.fte_per_duty},
    )
    with _LOCK:
        try:
            return scenario.compare(lines, levers)
        except ValueError as e:
            raise HTTPException(400, str(e))


@app.post("/api/scenario")
def run_scenario(req: ScenarioIn):
    res = _run(req)
    _LAST_SCENARIO["result"] = res
    return res


# ---------------------------------------------------------------- Claude-Chat


def _claude_available() -> bool:
    """Gibt es Zugangsdaten (API-Key, Token oder `ant auth login`-Profil)? Ohne Anfrage an die API."""
    try:
        c = anthropic.Anthropic()
    except Exception:
        return False
    return any((c.api_key, c.auth_token, c.credentials, c.custom_auth))


_SESSIONS: dict[str, dict] = {}


class ChatIn(BaseModel):
    session_id: str | None = None
    message: str


@app.post("/api/chat")
def chat(req: ChatIn):
    sid = req.session_id or uuid.uuid4().hex
    sess = _SESSIONS.get(sid)
    if sess is None:
        sess = {"last": None}

        def on_tool(name, inp, out):
            if name == "run_scenario":
                sess["last_input"] = inp

        try:
            sess["explainer"] = explainer.Explainer(on_tool=on_tool)
        except Exception as e:
            raise HTTPException(503, f"Claude nicht verfügbar: {e}. ANTHROPIC_API_KEY setzen.")
        _SESSIONS[sid] = sess
    sess.pop("last_input", None)
    try:
        answer = sess["explainer"].ask(req.message)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "Claude-Zugang ungültig. ANTHROPIC_API_KEY prüfen.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate-Limit erreicht, bitte kurz warten.")
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Keine Verbindung zur Claude API.")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API Fehler {e.status_code}: {e.message}")
    out = {"session_id": sid, "answer": answer}
    # Hat Claude ein Szenario gerechnet, bekommt die Oberfläche das volle Ergebnis (aus dem Cache, ohne Neuberechnung)
    if "last_input" in sess:
        inp = sess["last_input"]
        req2 = ScenarioIn(
            presets=inp.get("presets") or [],
            express_lines=[ExpressIn(**x) for x in inp.get("express_lines") or []],
            interlining=inp.get("interlining", True),
            split_duties=inp.get("split_duties", True),
            thin_parallel=inp.get("thin_parallel", False),
            thin_lines=inp.get("thin_lines") or None,
            fte_per_duty=inp.get("fte_per_duty") or 1.7,
        )
        out["scenario"] = _run(req2)
        out["scenario_request"] = req2.model_dump()
    return out


# ---------------------------------------------------------------- Start


@app.on_event("startup")
def warmup():
    """Bestand und Netz im Hintergrund vorrechnen, damit das erste Szenario schneller ist."""

    def work():
        network()
        with _LOCK:
            scenario.warmup()

    threading.Thread(target=work, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/web", StaticFiles(directory=WEB), name="web")
