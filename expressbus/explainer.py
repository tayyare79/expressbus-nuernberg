"""Claude als Szenario-Erklärer: beantwortet Was-wäre-wenn-Fragen auf Deutsch und ruft dafür den Optimierer auf.

Braucht Anthropic-Zugangsdaten (ANTHROPIC_API_KEY o. ä.). Auf dem Event gibt es 100 $ API-Guthaben pro Person.
Aufruf zum Testen: uv run python -m expressbus.explainer "Was kostet ein Express Fürth–Flughafen im 20-Minuten-Takt?"
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import anthropic

from . import data, scenario

MODEL = "claude-opus-5-5"

SYSTEM = """Du bist Planungsassistent für den Busbetrieb der VAG Nürnberg im Hackathon "Expressbusse trotz Fahrermangel".
Nürnberg hat genug Busse für zusätzliche Expresslinien, aber zu wenige Fahrer. Deine Aufgabe: Szenarien rechnen
und erklären, wie neue Expresslinien mit möglichst wenig zusätzlichem Personal fahren können.

Arbeitsweise:
- Rechne immer mit den Werkzeugen, erfinde keine Zahlen. Haltestellen vorher mit find_stops prüfen.
- Ein Dienst ist eine Fahrerschicht an einem Werktag. Fahrerbedarf = Dienste × Faktor (Standard 1,7 wegen 7-Tage-Betrieb,
  Urlaub, Krankheit).
- Die Kennzahl, auf die es ankommt, sind die Grenzkosten der Expresslinie: zusätzliche Dienste bzw. Fahrer gegenüber dem
  gleich optimierten Bestand. Unterscheide klar zwischen Einsparungen im Bestand (Hebel) und den Kosten der Expresslinie.
- Nenne Annahmen und Grenzen: Der Bestand ist eine Modellrechnung aus dem GTFS-Soll-Fahrplan (VGN, Stichtag Montag
  5.10.2026), nicht die echte VAG-Dienstplanung; Fahrzeiten der Expresslinie sind aus dem Fahrplan abgeleitet oder geschätzt.
- Antworte auf Deutsch, knapp und konkret: erst das Ergebnis in einem Satz, dann die wichtigsten Zahlen, dann Hebel
  und Abwägungen (z. B. dünnerer Takt auf Parallellinien in der Nebenverkehrszeit).
"""

TOOLS = [
    {
        "name": "find_stops",
        "description": "Sucht Haltestellen im VAG-Busnetz nach Namensteil und zeigt, welche Buslinien sie bedienen. "
        "Vor run_scenario nutzen, um Haltestellennamen zu prüfen.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Namensteil, z. B. 'Nordostbahnhof'"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_lines",
        "description": "Listet alle VAG-Stadtbuslinien am Stichtag mit Fahrten, Betriebszeit und Endhaltestellen.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_presets",
        "description": "Zeigt die vordefinierten Beispiel-Expresslinien (X30, X35, X45, X65).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_scenario",
        "description": "Rechnet ein Szenario mit Umlauf- und Dienstplanung (dauert 10–60 s). Gibt eine Treppe zurück "
        "(Bestand heute → Express getrennt → integriert → Hebel) und die Grenzkosten der Expresslinien.",
        "input_schema": {
            "type": "object",
            "properties": {
                "presets": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(scenario.PRESETS)},
                    "description": "Namen vordefinierter Expresslinien",
                },
                "express_lines": {
                    "type": "array",
                    "description": "Eigene Expresslinien",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "stops": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                            "headway": {
                                "type": "integer",
                                "minimum": scenario.HEADWAY_MIN,
                                "maximum": scenario.HEADWAY_MAX,
                                "description": "Takt in Minuten",
                            },
                            "periods": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                                "description": "Betriebszeiten als [[\"06:00\",\"09:00\"], ...]",
                            },
                        },
                        "required": ["name", "stops", "headway"],
                    },
                },
                "interlining": {"type": "boolean", "description": "Linienübergreifende Umläufe zulassen (Standard ja)"},
                "split_duties": {"type": "boolean", "description": "Geteilte Dienste zulassen (Standard ja)"},
                "thin_parallel": {
                    "type": "boolean",
                    "description": "Parallele Linien in der Nebenverkehrszeit ausdünnen (Standard nein)",
                },
                "thin_lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Welche Linien ausdünnen; leer = automatisch erkannte Parallellinien",
                },
                "fte_per_duty": {"type": "number", "description": "Fahrer je täglichem Dienst (Standard 1,7)"},
            },
        },
    },
]


# ---------------------------------------------------------------- Werkzeuge


def find_stops(query: str) -> dict:
    net = data.load()
    st = net.stops[net.stops.stop_name.str.contains(query, case=False, regex=False)]
    ts = net.trip_stops[net.trip_stops.stop_id.isin(st.stop_id)].merge(net.trips[["trip_id", "line"]], on="trip_id")
    ts = ts.merge(st[["stop_id", "stop_name"]], on="stop_id")
    out = (
        ts.groupby("stop_name")
        .agg(linien=("line", lambda s: sorted(set(s), key=lambda q: (len(q), q))), abfahrten=("trip_id", "size"))
        .sort_values("abfahrten", ascending=False)
        .head(15)
        .reset_index()
    )
    return {"treffer": out.to_dict("records")}


def list_lines() -> dict:
    lines = data.load().lines.copy()
    lines["betrieb"] = lines["first"].map(_hhmm) + "–" + lines["last"].map(_hhmm)
    lines = lines.sort_values("line", key=lambda s: s.str.zfill(4))
    return {
        "linien": lines[["line", "trips", "betrieb", "termini"]]
        .rename(columns={"line": "linie", "trips": "fahrten"})
        .to_dict("records")
    }


def list_presets() -> dict:
    return {"presets": [asdict(x) for x in scenario.PRESETS.values()]}


def run_scenario(
    presets: list[str] | None = None,
    express_lines: list[dict] | None = None,
    interlining: bool = True,
    split_duties: bool = True,
    thin_parallel: bool = False,
    thin_lines: list[str] | None = None,
    fte_per_duty: float | None = None,
) -> dict:
    lines = [scenario.PRESETS[p] for p in presets or []]
    for x in express_lines or []:
        lines.append(
            scenario.ExpressLine(
                name=x["name"],
                stops=x["stops"],
                headway=int(x["headway"]),
                periods=[tuple(p) for p in x.get("periods") or [("06:00", "20:00")]],
            )
        )
    if not lines:
        raise ValueError("Mindestens eine Expresslinie angeben (presets oder express_lines).")
    params = {"fte_per_duty": fte_per_duty} if fte_per_duty else {}
    levers = scenario.Levers(
        interlining=interlining,
        split_duties=split_duties,
        thin_parallel=thin_parallel,
        thin_lines=thin_lines or None,
        params=params,
    )
    return compact(scenario.compare(lines, levers))


def compact(res: dict) -> dict:
    """Kürzt das Ergebnis auf das, was Claude zum Erklären braucht."""
    keep = ("fahrten", "fahrplan_km", "fahrplanstunden", "fahrzeuge", "umlaeufe", "dienste", "dienste_geteilt",
            "fahrer_bedarf", "arbeitsstunden", "produktivitaet", "leerfahrt_km")
    return {
        "expresslinien": [
            {k: x[k] for k in ("name", "description", "headway", "periods", "fahrzeit_min", "fahrten", "segments")}
            | {"halte": [s["name"] for s in x["stations"]]}
            for x in res["express"]
        ],
        "ausgeduennte_parallellinien": res["parallel_ausgeduennt"],
        "treppe": [
            {"schritt": s["schritt"], "erklaerung": s["erklaerung"], "kpi": {k: s["kpi"][k] for k in keep},
             "delta_zum_bestand": s["delta"], **({"entfallene_fahrten": s["entfallene_fahrten"]} if "entfallene_fahrten" in s else {})}
            for s in res["steps"]
        ],
        "grenzkosten_expresslinien": res["grenzkosten"],
        "fahrer_je_dienst": res["params"]["fte_per_duty"],
        "hinweis": res["hinweis"],
    }


HANDLERS = {"find_stops": find_stops, "list_lines": list_lines, "list_presets": list_presets, "run_scenario": run_scenario}


def _hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------- Agentenschleife


class Explainer:
    """Mehrstufiges Gespräch. Der Verlauf enthält alle Blöcke (auch thinking/tool_use) unverändert."""

    def __init__(self, client: anthropic.Anthropic | None = None, on_tool=None):
        self.client = client or anthropic.Anthropic()
        self.messages: list[dict] = []
        self.on_tool = on_tool  # Callback(name, input, result) – z. B. damit die Web-UI das letzte Szenario zeigt

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        for _ in range(12):
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=self.messages,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                cache_control={"type": "ephemeral"},
            )
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "refusal":
                return "Die Anfrage wurde abgelehnt. Bitte anders formulieren."
            if response.stop_reason == "max_tokens":
                return _text(response) + "\n\n(Antwort abgeschnitten)"
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                return _text(response)
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    out = HANDLERS[block.name](**(block.input or {}))
                    if self.on_tool:
                        self.on_tool(block.name, block.input, out)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out, ensure_ascii=False, default=str)}
                    )
                except Exception as e:  # Fehler an Claude zurückgeben, damit es korrigieren kann
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Fehler: {e}", "is_error": True})
            self.messages.append({"role": "user", "content": results})
        return "Abbruch: zu viele Werkzeugaufrufe."


def _text(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Was kostet die Expresslinie X30 an Fahrern, und wie kriegen wir das mit dem vorhandenen Personal hin?"
    print(Explainer().ask(q))
