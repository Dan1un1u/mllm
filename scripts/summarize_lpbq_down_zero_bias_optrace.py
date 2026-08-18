#!/usr/bin/env python3
"""Compare raw HTP phases for the two down-projection bias variants."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGET = "model.layers.14.mlp.down_proj"


def category(htp_type: str, flags: tuple[str, ...]) -> str:
    low = htp_type.lower()
    if "expand_block_quant" in low:
        return "weight_expand_dequant"
    if "weights_to_vtcm" in low or ("weight" in low and ("dma" in low or "wait" in low)):
        return "weight_dma_wait"
    if "bias_to_vtcm" in low or ("bias" in low and ("dma" in low or "wait" in low)):
        return "bias_dma_wait"
    if "convlayer_s1.opt" in low or "uses_hmx" in flags:
        return "hmx_mac"
    if any(token in low for token in ("checkpoint", "dma_set", "sync", "wait")):
        return "synchronization"
    return "other_fused"


def interval_union(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    events = document if isinstance(document, list) else document["traceEvents"]
    process_names = {
        event.get("pid"): event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    groups = defaultdict(lambda: {"event_count": 0, "work_cycles": 0,
                                  "hardware_active_cycles": 0, "dominant_path_cycles": 0,
                                  "intervals": [], "htp_types": Counter()})
    seen = set()
    signatures = Counter()
    for event in events:
        if event.get("ph") != "X" or not process_names.get(event.get("pid"), "").startswith("QNN::"):
            continue
        args = event.get("args", {})
        if args.get("QNN Op Name") != TARGET:
            continue
        duration = int(event.get("dur", 0) or 0)
        start = int(event.get("ts", 0) or 0)
        htp_type = args.get("HTP Op Type", event.get("name", ""))
        flags = tuple(args.get("Flags", []))
        identity = (args.get("ID"), event.get("tid"), start, duration)
        if duration <= 0 or identity in seen:
            continue
        seen.add(identity)
        key = category(htp_type, flags)
        group = groups[key]
        group["event_count"] += 1
        group["work_cycles"] += duration
        group["hardware_active_cycles"] += int(args.get("Duration (cycles)", 0) or 0)
        group["dominant_path_cycles"] += int(args.get("Dominant Path Cycles", 0) or 0)
        group["intervals"].append((start, start + duration))
        group["htp_types"][htp_type] += 1
        signatures[(key, htp_type, flags)] += 1

    result = {}
    all_intervals = []
    for key, group in groups.items():
        intervals = group.pop("intervals")
        all_intervals.extend(intervals)
        group["active_union_cycles"] = interval_union(intervals)
        group["htp_types"] = dict(sorted(group["htp_types"].items()))
        result[key] = group
    return {
        "target": TARGET,
        "event_signature": [
            {"category": key[0], "htp_type": key[1], "flags": list(key[2]), "count": count}
            for key, count in sorted(signatures.items())
        ],
        "signature_event_count": sum(signatures.values()),
        "target_work_cycles": sum(group["work_cycles"] for group in result.values()),
        "target_active_union_cycles": interval_union(all_intervals),
        "phases": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("omitted", type=Path)
    parser.add_argument("explicit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    omitted = summarize(args.omitted)
    explicit = summarize(args.explicit)
    report = {
        "omitted": omitted,
        "explicit_u8_zero": explicit,
        "same_physical_event_signature": omitted["event_signature"] == explicit["event_signature"],
        "explicit_over_omitted_target_work": (
            explicit["target_work_cycles"] / omitted["target_work_cycles"]
        ),
        "explicit_over_omitted_target_active_union": (
            explicit["target_active_union_cycles"] / omitted["target_active_union_cycles"]
        ),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "same_physical_event_signature": report["same_physical_event_signature"],
        "omitted_target_work_cycles": omitted["target_work_cycles"],
        "explicit_target_work_cycles": explicit["target_work_cycles"],
        "explicit_over_omitted_target_work": report["explicit_over_omitted_target_work"],
        "omitted_target_active_union_cycles": omitted["target_active_union_cycles"],
        "explicit_target_active_union_cycles": explicit["target_active_union_cycles"],
        "explicit_over_omitted_target_active_union": report["explicit_over_omitted_target_active_union"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
