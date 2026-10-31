"""Command-line demo. Run ``python -m drone_voice --help``.

Everything here is simulated: inputs are typed or come from fixture files (not
from the speech model), and the adapter is ``SimulatedController``.
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

from .clock import ManualClock, SystemClock
from .contract import Confidence, ContractError, format_timestamp, parse_timestamp
from .controller import SimulatedController
from .decision_log import DecisionLog
from .dedup import DedupStoreError, InMemoryDedupStore, JsonlDedupStore
from .dispatcher import Dispatcher
from .jsonio import JsonFormatError, dumps, loads_strict, write_json_file
from .pipeline import DEFAULT_SETTINGS_PATH, DEFAULT_VOCABULARY_PATH, REPO_ROOT, build_pipeline
from .settings import load_settings
from .transcription import MANUAL_SOURCE, SIMULATED_SOURCE, TranscriptionEvent
from .vocabulary import load_vocabulary

DEFAULT_STATE_PATH = REPO_ROOT / "state" / "processed_command_ids.jsonl"
_BATCH_KEYS = {"utterance_id", "text", "is_final", "t", "source", "confidence", "comment"}
SIMULATION_NOTICE = "SIMULATION ONLY: no drone, no autopilot, no flight physics were involved."


class UsageError(Exception):
    pass


def _parse_now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        return parse_timestamp(value, "--now")
    except ContractError as exc:
        raise UsageError(exc.detail) from None


def _section(out: TextIO, title: str, payload: Any) -> None:
    out.write(f"== {title} ==\n")
    out.write(payload if isinstance(payload, str) else dumps(payload))
    out.write("\n")


def _load_config(args: argparse.Namespace):
    return load_vocabulary(args.vocabulary), load_settings(args.settings)


def cmd_demo(args: argparse.Namespace, out: TextIO) -> int:
    vocabulary, settings = _load_config(args)
    clock = ManualClock(_parse_now(args.now))
    log = DecisionLog(args.decision_log, include_text=not args.no_log_text)
    pipeline = build_pipeline(clock, vocabulary=vocabulary, settings=settings,
                              dedup_store=InMemoryDedupStore(), log=log)
    event = TranscriptionEvent(
        utterance_id=args.utterance_id or f"utt-{uuid.uuid4()}",
        text=args.text,
        is_final=not args.partial,
        source=MANUAL_SOURCE,
    )
    history = pipeline.dispatcher.history
    message, _ = pipeline.handle_event(event)
    if message is None:
        _section(out, "Message JSON", "(partial transcription: no message produced, nothing dispatched)")
    else:
        _section(out, "Message JSON", message.to_dict())
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            try:
                write_json_file(args.save_json, message.to_dict())
            except FileExistsError:
                raise UsageError(f"{args.save_json} already exists; refusing to overwrite") from None
            out.write(f"(saved to {args.save_json})\n")
    _section(out, "Dispatch results", [r.to_dict() for r in history])
    active = pipeline.dispatcher.active_movement
    if active is not None:
        before = len(history)
        clock.advance_to(active.ends_at)
        pipeline.dispatcher.poll()
        _section(out, f"After simulated time advanced to {format_timestamp(clock.now())}",
                 [r.to_dict() for r in history[before:]])
    _section(out, "Simulated adapter calls", pipeline.controller.calls)
    out.write(SIMULATION_NOTICE + "\n")
    return 0


def _read_batch(path: Path) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    previous_t = 0.0
    for line_no, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            item = loads_strict(raw)
        except JsonFormatError as exc:
            raise UsageError(f"{path}:{line_no}: {exc}") from None
        if not isinstance(item, dict):
            raise UsageError(f"{path}:{line_no}: each line must be a JSON object")
        unknown = set(item) - _BATCH_KEYS
        missing = {"utterance_id", "text", "is_final"} - set(item)
        if unknown or missing:
            raise UsageError(f"{path}:{line_no}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        t = item.get("t", previous_t)
        if type(t) not in (int, float) or not math.isfinite(t) or t < previous_t:
            raise UsageError(f"{path}:{line_no}: 't' must be a finite number, non-decreasing across lines")
        previous_t = float(t)
        confidence = item.get("confidence")
        try:
            if confidence is not None:
                if not isinstance(confidence, dict) or set(confidence) != {"value", "provider"}:
                    raise ValueError("confidence must be an object with exactly 'value' and 'provider'")
                confidence = Confidence(confidence["value"], confidence["provider"])
            event = TranscriptionEvent(
                utterance_id=item["utterance_id"],
                text=item["text"],
                is_final=item["is_final"],
                source=item.get("source", SIMULATED_SOURCE),
                confidence=confidence,
            )
        except ValueError as exc:
            raise UsageError(f"{path}:{line_no}: {exc}") from None
        lines.append({"line": line_no, "t": previous_t, "event": event})
    return lines


def cmd_batch(args: argparse.Namespace, out: TextIO) -> int:
    vocabulary, settings = _load_config(args)
    items = _read_batch(Path(args.input))  # validate the whole file before acting
    start = _parse_now(args.now)
    store = JsonlDedupStore(args.state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        out_fh = open(args.out, "x", encoding="utf-8", newline="\n")
    except FileExistsError:
        raise UsageError(f"{args.out} already exists; refusing to overwrite") from None
    with out_fh:
        clock = ManualClock(start)
        log = DecisionLog(args.decision_log, include_text=not args.no_log_text)
        pipeline = build_pipeline(clock, vocabulary=vocabulary, settings=settings, dedup_store=store, log=log)
        history = pipeline.dispatcher.history
        counts: dict[str, int] = {}

        def emit(record: dict[str, Any]) -> None:
            out_fh.write(dumps(record, indent=None) + "\n")

        def emit_results(line: int | None, before: int) -> None:
            for result in history[before:]:
                emit({"line": line, "dispatch": result.to_dict()})
                counts[result.outcome.value] = counts.get(result.outcome.value, 0) + 1

        def run_until(target: datetime, line: int | None) -> None:
            """Advance simulated time to ``target``, stopping each movement exactly when it is due."""
            while True:
                active = pipeline.dispatcher.active_movement
                if active is None or active.ends_at > target:
                    break
                clock.advance_to(active.ends_at)
                before = len(history)
                pipeline.dispatcher.poll()
                emit_results(line, before)
                if pipeline.dispatcher.active_movement is active:
                    break  # the stop failed; poll() retries it at the next step
            clock.advance_to(target)

        for item in items:
            run_until(start + timedelta(seconds=item["t"]), item["line"])
            event: TranscriptionEvent = item["event"]
            before = len(history)
            message, _ = pipeline.handle_event(event)
            emit({
                "line": item["line"],
                "utterance_id": event.utterance_id,
                "is_final": event.is_final,
                "message": message.to_dict() if message else None,
                "note": None if message else "partial transcription: not interpreted, not dispatched",
            })
            if message is None:
                counts["partial_ignored"] = counts.get("partial_ignored", 0) + 1
            emit_results(item["line"], before)
        active = pipeline.dispatcher.active_movement
        if active is not None:
            run_until(active.ends_at, None)
        emit({"simulated_adapter_calls": pipeline.controller.calls, "notice": SIMULATION_NOTICE})

    out.write(f"Processed {len(items)} input lines from {args.input}\n")
    out.write(f"Results written to {args.out}\n")
    out.write(f"Processed-id store: {args.state}\n")
    out.write("Outcomes: " + dumps(dict(sorted(counts.items())), indent=None) + "\n")
    out.write(f"Simulated adapter calls: {len(pipeline.controller.calls)}\n")
    out.write(SIMULATION_NOTICE + "\n")
    return 0


def cmd_dispatch(args: argparse.Namespace, out: TextIO) -> int:
    vocabulary, settings = _load_config(args)
    clock = SystemClock()
    controller = SimulatedController(clock)
    log = DecisionLog(args.decision_log, include_text=not args.no_log_text)
    dispatcher = Dispatcher(controller, settings, clock, dedup_store=JsonlDedupStore(args.state),
                            vocabulary=vocabulary, log=log)
    result = dispatcher.dispatch_json(Path(args.message).read_bytes())
    _section(out, "Dispatch result", result.to_dict())
    _section(out, "Simulated adapter calls", controller.calls)
    if dispatcher.active_movement is not None:
        out.write(f"(simulated movement would end at {format_timestamp(dispatcher.active_movement.ends_at)}; "
                  "this one-shot command exits before that)\n")
    out.write(SIMULATION_NOTICE + "\n")
    return 0


def cmd_check_config(args: argparse.Namespace, out: TextIO) -> int:
    vocabulary, settings = _load_config(args)
    _section(out, "Vocabulary (PROVISIONAL, to validate with users and real model outputs)", [
        {"expression": e.expression, "normalized": e.normalized, "intent": e.intent.value}
        for e in vocabulary.entries
    ])
    _section(out, "Settings (simulation only, not flight parameters)", {
        "ttl_s": settings.ttl_s,
        "max_ttl_s": settings.max_ttl_s,
        "max_future_skew_s": settings.max_future_skew_s,
        "max_speed_m_s": settings.max_speed_m_s,
        "max_duration_s": settings.max_duration_s,
        "motion": {k.value: {"speed_m_s": v.speed_m_s, "duration_s": v.duration_s}
                   for k, v in settings.motion.items()},
    })
    out.write("Configuration OK\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m drone_voice", description=SIMULATION_NOTICE)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--vocabulary", default=str(DEFAULT_VOCABULARY_PATH), help="vocabulary JSON file")
    common.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH), help="settings JSON file")
    logging = argparse.ArgumentParser(add_help=False)
    logging.add_argument("--decision-log", help="append structured decisions (JSON Lines) to this file")
    logging.add_argument("--no-log-text", action="store_true", help="do not write transcription text in the log")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", parents=[common, logging], help="interpret one transcription and simulate it")
    demo.add_argument("text", help="transcription text (treated as FINAL unless --partial)")
    demo.add_argument("--partial", action="store_true", help="mark the transcription as partial (never executed)")
    demo.add_argument("--utterance-id", help="utterance identifier (default: a new random one)")
    demo.add_argument("--now", help="simulated start time, e.g. 2026-01-01T12:00:00Z (default: current UTC time)")
    demo.add_argument("--save-json", help="also write the message to this file (never overwrites)")
    demo.set_defaults(func=cmd_demo)

    batch = sub.add_parser("batch", parents=[common, logging], help="process a JSON Lines file of transcriptions")
    batch.add_argument("input", help="JSONL: utterance_id, text, is_final, optional t/source/confidence/comment")
    batch.add_argument("--out", required=True, help="JSONL results file (must not exist)")
    batch.add_argument("--state", default=str(DEFAULT_STATE_PATH), help="persistent processed-id store")
    batch.add_argument("--now", help="simulated start time (default: current UTC time)")
    batch.set_defaults(func=cmd_batch)

    dispatch = sub.add_parser("dispatch", parents=[common, logging],
                              help="validate and dispatch a saved JSON message with the real clock")
    dispatch.add_argument("message", help="message JSON file")
    dispatch.add_argument("--state", default=str(DEFAULT_STATE_PATH), help="persistent processed-id store")
    dispatch.set_defaults(func=cmd_dispatch)

    check = sub.add_parser("check-config", parents=[common], help="validate vocabulary and settings")
    check.set_defaults(func=cmd_check_config)
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    if out is None:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        out = sys.stdout
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, out)
    except (UsageError, ValueError, DedupStoreError, OSError) as exc:  # ValueError covers ConfigError, ContractError
        print(f"error: {exc}", file=sys.stderr)
        return 2
