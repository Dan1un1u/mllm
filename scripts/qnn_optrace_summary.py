#!/usr/bin/env python3
"""Summarize a QNN HTP Optrace Chrome Trace at QNN-operator granularity."""

import argparse
import csv
import gc
import json
import re
from collections import defaultdict
from pathlib import Path


def scalar_string(value):
    if not isinstance(value, dict):
        return None
    for item in value.values():
        if isinstance(item, str):
            return item
    return None


def load_htp_flags(path):
    if path is None:
        return {}
    with path.open() as stream:
        document = json.load(stream)
    flags = defaultdict(set)
    for node in document.get("graph", {}).get("nodes", {}).values():
        params = node.get("scalar_params", {})
        qnn_name = scalar_string(params.get("qnn_op_name"))
        op_flag = scalar_string(params.get("op_flags"))
        if qnn_name and op_flag:
            resource = op_flag.removeprefix("uses_").upper()
            if resource != "NULL_EXEC":
                flags[qnn_name].add(resource)
    del document
    gc.collect()
    return flags


def lane_type(name):
    match = re.search(r"Type:\s*([^ ]+)", name)
    return match.group(1) if match else "UNKNOWN"


def logical_op_type(qnn_name, qnn_type):
    if qnn_type != "Conv2d_w_blk_exp_scale":
        return qnn_type
    projection_names = (
        ".q_proj", ".k_proj", ".v_proj", ".o_proj",
        ".up_proj", ".gate_proj", ".down_proj",
    )
    if qnn_name == "lm_head" or any(name in qnn_name for name in projection_names):
        return "LinearProjection_via_LPBQ_Conv2D"
    return "LPBQ_Conv2D_BlockScale"


def interval_stats(intervals):
    if not intervals:
        return 0, 0, 0, 0, 0, 0
    ordered = sorted(intervals)
    first = ordered[0][0]
    last = max(end for _, end in ordered)
    work = sum(end - start for start, end in ordered)
    active = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            active += current_end - current_start
            current_start, current_end = start, end
    active += current_end - current_start

    points = []
    for start, end in ordered:
        points.append((start, 1))
        points.append((end, -1))
    parallel = maximum = 0
    for _, delta in sorted(points, key=lambda point: (point[0], point[1])):
        parallel += delta
        maximum = max(maximum, parallel)
    return first, last, last - first, active, work, maximum


def main():
    parser = argparse.ArgumentParser(
        description="Combine an Optrace Chrome Trace and its generated _htp.json into an operator CSV."
    )
    parser.add_argument("chrometrace", type=Path)
    parser.add_argument("--htp-json", type=Path, help="qnn-profile-viewer's *_htp.json output")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--type-summary-output", type=Path,
                        help="Optional logical operator-type aggregation CSV")
    args = parser.parse_args()

    topology_flags = load_htp_flags(args.htp_json)
    with args.chrometrace.open() as stream:
        document = json.load(stream)
    events = document if isinstance(document, list) else document.get("traceEvents", [])

    process_names = {
        event.get("pid"): event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    thread_names = {
        (event.get("pid"), event.get("tid")): event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }

    intervals = defaultdict(list)
    lanes = defaultdict(set)
    qnn_types = defaultdict(set)
    seen = set()
    for event in events:
        if event.get("ph") != "X" or not process_names.get(event.get("pid"), "").startswith("QNN::"):
            continue
        info = event.get("args", {})
        qnn_name = info.get("QNN Op Name")
        duration = event.get("dur", 0)
        if not qnn_name or duration <= 0:
            continue
        start = event.get("ts", 0)
        identity = (info.get("ID"), qnn_name, event.get("tid"), start, duration)
        if identity in seen:
            continue
        seen.add(identity)
        intervals[qnn_name].append((start, start + duration))
        lanes[qnn_name].add(lane_type(thread_names.get((event.get("pid"), event.get("tid")), "")))
        if info.get("QNN Op Type"):
            qnn_types[qnn_name].add(info["QNN Op Type"])

    columns = [
        "qnn_op_name",
        "qnn_op_type",
        "logical_op_type",
        "execution_domain",
        "kernel_resources",
        "trace_lanes",
        "start_cycle",
        "end_cycle",
        "wall_span_cycles",
        "active_union_cycles",
        "total_work_cycles",
        "overlapped_work_cycles",
        "max_parallelism",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    logical_intervals = defaultdict(list)
    logical_resources = defaultdict(set)
    logical_lanes = defaultdict(set)
    logical_instances = defaultdict(int)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for qnn_name in sorted(intervals):
            first, last, wall, active, work, parallel = interval_stats(intervals[qnn_name])
            qnn_type = "+".join(sorted(qnn_types[qnn_name]))
            logical_type = logical_op_type(qnn_name, qnn_type)
            row = {
                    "qnn_op_name": qnn_name,
                    "qnn_op_type": qnn_type,
                    "logical_op_type": logical_type,
                    "execution_domain": "HTP",
                    "kernel_resources": "+".join(sorted(topology_flags.get(qnn_name, []))) or "UNKNOWN",
                    "trace_lanes": "+".join(sorted(lanes[qnn_name])),
                    "start_cycle": first,
                    "end_cycle": last,
                    "wall_span_cycles": wall,
                    "active_union_cycles": active,
                    "total_work_cycles": work,
                    "overlapped_work_cycles": max(0, work - active),
                    "max_parallelism": parallel,
                }
            writer.writerow(row)
            output_rows.append(row)
            logical_intervals[logical_type].extend(intervals[qnn_name])
            logical_resources[logical_type].update(topology_flags.get(qnn_name, []))
            logical_lanes[logical_type].update(lanes[qnn_name])
            logical_instances[logical_type] += 1

    if args.type_summary_output is not None:
        summary_columns = [
            "logical_op_type", "qnn_op_instances", "kernel_resources", "trace_lanes",
            "wall_span_cycles", "active_union_cycles", "total_work_cycles",
            "overlapped_work_cycles", "max_parallelism",
        ]
        args.type_summary_output.parent.mkdir(parents=True, exist_ok=True)
        with args.type_summary_output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary_columns)
            writer.writeheader()
            for logical_type in sorted(logical_intervals):
                _, _, wall, active, work, parallel = interval_stats(logical_intervals[logical_type])
                writer.writerow({
                    "logical_op_type": logical_type,
                    "qnn_op_instances": logical_instances[logical_type],
                    "kernel_resources": "+".join(sorted(logical_resources[logical_type])) or "UNKNOWN",
                    "trace_lanes": "+".join(sorted(logical_lanes[logical_type])),
                    "wall_span_cycles": wall,
                    "active_union_cycles": active,
                    "total_work_cycles": work,
                    "overlapped_work_cycles": max(0, work - active),
                    "max_parallelism": parallel,
                })

    all_intervals = [interval for values in intervals.values() for interval in values]
    _, _, wall, active, work, parallel = interval_stats(all_intervals)
    print(f"operators={len(intervals)} wall_cycles={wall} active_union_cycles={active} "
          f"total_work_cycles={work} max_parallelism={parallel}")
    print("execution_domain=HTP means the operator was observed inside the HTP graph; host-side CPU work is outside Optrace.")


if __name__ == "__main__":
    main()
