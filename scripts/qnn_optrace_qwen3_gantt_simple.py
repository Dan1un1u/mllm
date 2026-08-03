#!/usr/bin/env python3
"""Generate a compact three-layer Qwen3 structure Gantt from archived Optrace.

The main stage bars are colored by HTP execution resource (HMX, HVX, or a
mixed/fused HMX+HVX stage), while DMA is encoded as a wave overlay.  Optional
conversion lanes use raw Chrome Trace intervals so their placement remains
overlap-aware instead of treating summed work cycles as serialized latency.
"""

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

from qnn_optrace_quantization import CATEGORY_INFO, classify_quant_kernel
from qnn_optrace_qwen3_structure import classification, infer_head_counts, load_qhas
from qnn_optrace_summary import lane_type


LAYERS = (0, 5, 27)
CONVERSION_CATEGORIES = (
    "lpbq_weight_expand_dequant",
    "explicit_convert_requant",
    "fused_linearclip_requant",
    "matmul_signed_conversion",
)
CONVERSION_SHORT_LABELS = {
    "lpbq_weight_expand_dequant": "W4 expand/dequant",
    "explicit_convert_requant": "Convert / CastType",
    "fused_linearclip_requant": "Fused requant",
    "matmul_signed_conversion": "Signed conversion",
}
RESOURCE_INFO = {
    "hmx": ("HMX compute", "Matrix / tensor arithmetic reported on an HMX lane"),
    "hvx": ("HVX compute", "Vector, reduction, format conversion and weight expansion on HVX"),
    "dma_transfer": ("DMA transfer", "Explicit DMA activity such as weights/activations/bias to VTCM"),
    "dma_wait": ("DMA wait", "Explicit wait for a DMA dependency to complete"),
    "sync": ("Checkpoint / sync", "DMA set, checkpoint and scheduling synchronization"),
}
RESOURCE_ORDER = tuple(RESOURCE_INFO)
FOCUS_GROUPS = [
    ("input_norm", "Input RMSNorm", {"input_rmsnorm"}),
    ("q_projection", "Q projection", {"q_projection"}),
    ("k_projection", "K projection", {"k_projection"}),
    ("v_projection", "V projection", {"v_projection"}),
    ("qk_norm", "Q/K head RMSNorm", {"q_head_rmsnorm", "k_head_rmsnorm"}),
    ("qk_rope", "Q/K RoPE", {"q_rope", "k_rope"}),
    ("kv_cache", "KV-cache update", {"kv_cache_update"}),
    ("qk_similarity", "QKᵀ similarity", {"qk_similarity"}),
    ("softmax", "Scale + mask + Softmax", {"scale_mask_softmax"}),
    ("attention_value", "Attention × V + merge", {"attention_value", "head_merge"}),
    ("o_projection", "O projection", {"o_projection"}),
    ("attention_residual", "Attention residual", {"attention_residual"}),
    ("post_norm", "Post-attention RMSNorm", {"post_attention_rmsnorm"}),
    ("mlp_gate_projection", "MLP gate projection", {"mlp_gate_projection"}),
    ("mlp_up_projection", "MLP up projection", {"mlp_up_projection"}),
    ("mlp_silu", "MLP SiLU", {"mlp_silu"}),
    ("mlp_gate_product", "MLP gate × up", {"mlp_gate_product"}),
    ("mlp_down_projection", "MLP down projection", {"mlp_down_projection"}),
    ("mlp_residual", "MLP residual", {"mlp_residual"}),
]
FOCUS_BY_STAGE = {stage: key for key, _, stages in FOCUS_GROUPS for stage in stages}
FOCUS_LABELS = {key: label for key, label, _ in FOCUS_GROUPS}
GROUPS = [
    ("norm", "Input RMSNorm", {"input_rmsnorm"}),
    ("qkv", "Q / K / V projection", {"q_projection", "k_projection", "v_projection"}),
    ("rope", "Q/K Norm + RoPE", {"q_head_rmsnorm", "k_head_rmsnorm", "q_rope", "k_rope"}),
    ("cache", "KV cache", {"kv_cache_update"}),
    ("attention", "QKᵀ + Softmax + Attn×V", {"qk_similarity", "scale_mask_softmax", "attention_value", "head_merge"}),
    ("output", "O projection + residual", {"o_projection", "attention_residual"}),
    ("postnorm", "Post-attention RMSNorm", {"post_attention_rmsnorm"}),
    ("mlp", "MLP", {"mlp_gate_projection", "mlp_up_projection", "mlp_silu", "mlp_gate_product", "mlp_down_projection", "mlp_residual"}),
]
GROUP_BY_STAGE = {stage: key for key, _, stages in GROUPS for stage in stages}
GROUP_LABELS = {key: label for key, label, _ in GROUPS}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def n(row, key, integer=False):
    value = row.get(key, 0) or 0
    return int(float(value)) if integer else float(value)


def infer_graph_execute_us(rows):
    values = [
        n(row, "critical_path_us_estimate") * 100 / n(row, "critical_path_percent")
        for row in rows if n(row, "critical_path_percent") > 0
    ]
    values.sort()
    return values[len(values) // 2]


def resource_style(resources):
    tokens = set((resources or "").split("+"))
    if {"HMX", "HVX"} <= tokens:
        execution = "fusion"
    elif "HMX" in tokens:
        execution = "hmx"
    elif "HVX" in tokens:
        execution = "hvx"
    else:
        execution = "other"
    return execution, "DMA" in tokens


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersection_length(left, right):
    left = merge_intervals(left)
    right = merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def intersection_intervals(left, right):
    left = merge_intervals(left)
    right = merge_intervals(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return merge_intervals(result)


def merge_annotated(events):
    """Merge intervals while retaining whether any constituent event is critical."""
    merged = []
    for start, end, critical in sorted(events):
        if not merged or start > merged[-1][1]:
            merged.append([start, end, bool(critical)])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] |= bool(critical)
    return merged


def classify_resource(flags, htp_type, thread_name):
    low_flags = {str(flag).lower() for flag in flags}
    low_type = (htp_type or "").lower()
    if "dma_wait" in low_flags or "dmawait" in low_type:
        return "dma_wait"
    if low_flags & {"dma_set", "sync"} or any(token in low_type for token in ("checkpoint", "dma_set")):
        return "sync"
    if "dma" in low_flags or "to_vtcm" in low_type:
        return "dma_transfer"
    lane = lane_type(thread_name).upper()
    if "uses_hmx" in low_flags or lane == "HMX":
        return "hmx"
    if "uses_hvx" in low_flags or lane == "HVX":
        return "hvx"
    return None


def cook(row, graph_execute_us, total_dominant, layer=None):
    dominant = n(row, "num_dominant_path_cycles_htp_0", True)
    critical_us = (
        n(row, "critical_path_us_estimate")
        if row.get("critical_path_us_estimate") not in (None, "")
        else graph_execute_us * dominant / total_dominant if total_dominant else 0
    )
    stage = row["stage"]
    resources = row.get("kernel_resources", "")
    execution, dma = resource_style(resources)
    return {
        "layer": layer,
        "order": n(row, "stage_order", True),
        "stage": stage,
        "label": row.get("stage_label", stage),
        "group": GROUP_BY_STAGE.get(stage, "global"),
        "resources": resources,
        "execution": execution,
        "dma": dma,
        "start": n(row, "start_cycle", True),
        "end": n(row, "end_cycle", True),
        "wall": n(row, "wall_span_cycles", True),
        "active": n(row, "active_union_cycles", True),
        "dominant": dominant,
        "critical_us": critical_us,
    }


def cook_runtime_tracks(result_dir, graph):
    prefix = result_dir / f"qwen3-sm8750-v79-{graph}"
    qhas_path = Path(f"{prefix}-chrometrace_qnn_htp_analysis_summary.json")
    with qhas_path.open() as stream:
        qhas_document = json.load(stream)
    qhas_data = qhas_document["data"]
    qhas_rows = qhas_data["qnn_op_instances_nodes"]["data"]
    htp_instance_rows = qhas_data["htp_op_instances"]["data"]
    qhas_overall = qhas_data["htp_overall_summary"]["data"][0]
    dominant_path_rows = qhas_data["dominant_path_htp_0"]["data"]
    num_q_heads, num_kv_heads = infer_head_counts(qhas_rows)
    qnn_types = {row["qnn_op"]: row.get("qnn_op_type", "") for row in qhas_rows}

    # QHAS uses an absolute HTP clock while Chrome Trace rebases the same clock
    # to zero.  Dominant Path HTP0 is gap-free, so subtracting the first path
    # cycle gives intervals directly comparable with Chrome Trace timestamps.
    path_base = int(dominant_path_rows[0]["start_cycle"])
    critical_path = []
    for row in dominant_path_rows:
        stage, layer, _, _ = classification(
            row["qnn_op"], qnn_types.get(row["qnn_op"], ""), num_q_heads, num_kv_heads
        )
        critical_path.append((
            int(row["start_cycle"]) - path_base,
            int(row["end_cycle"]) - path_base,
            layer,
            FOCUS_BY_STAGE.get(stage),
        ))

    focus_owner_dominant = defaultdict(int)
    focus_official_resources = defaultdict(lambda: defaultdict(int))
    focus_official_quant = defaultdict(lambda: defaultdict(int))
    full_official_resources = defaultdict(int)
    full_official_quant = defaultdict(int)
    ownership_totals = defaultdict(int)
    qnn_direct_total = 0
    for row in qhas_rows:
        direct = n(row, "num_dominant_path_cycles_htp_0", True)
        qnn_direct_total += direct
        stage, layer, _, _ = classification(
            row["qnn_op"], row.get("qnn_op_type", ""), num_q_heads, num_kv_heads
        )
        focus_key = FOCUS_BY_STAGE.get(stage)
        if layer in LAYERS and focus_key:
            ownership_totals[f"layer_{layer}"] += direct
            focus_owner_dominant[(layer, focus_key)] += direct
        elif layer in LAYERS:
            ownership_totals["selected_layer_other"] += direct
        elif layer is not None:
            ownership_totals["other_layers"] += direct
        else:
            ownership_totals["global_runtime"] += direct

    for row in htp_instance_rows:
        direct = n(row, "num_dominant_path_cycles", True)
        if not direct:
            continue
        if row.get("hmx"):
            resource = "hmx"
        elif row.get("hvx"):
            resource = "hvx"
        elif row.get("dma_wait"):
            resource = "dma_wait"
        elif row.get("dma"):
            resource = "dma_transfer"
        elif row.get("dma_set") or row.get("sync"):
            resource = "sync"
        else:
            resource = "unresolved"
        full_official_resources[resource] += direct
        flags = tuple(
            name for field, name in (
                ("hmx", "uses_hmx"), ("hvx", "uses_hvx"), ("dma", "dma"),
                ("dma_wait", "dma_wait"), ("dma_set", "dma_set"), ("sync", "sync"),
            ) if row.get(field)
        )
        category = classify_quant_kernel(
            row.get("htp_op", ""), row.get("qnn_op_type", ""), flags
        )
        if category in CONVERSION_CATEGORIES:
            full_official_quant[category] += direct
        stage, layer, _, _ = classification(
            row["qnn_op"], row.get("qnn_op_type", ""), num_q_heads, num_kv_heads
        )
        focus_key = FOCUS_BY_STAGE.get(stage)
        if layer not in LAYERS or not focus_key:
            continue
        focus_official_resources[(layer, focus_key)][resource] += direct
        if category in CONVERSION_CATEGORIES:
            focus_official_quant[(layer, focus_key)][category] += direct

    path_cycles = sum(int(row.get("dp_cycles", 0) or 0) for row in dominant_path_rows)
    closure = {
        "timeline_cycles": int(qhas_overall["timeline_cycles"]),
        "qhas_time_us": int(qhas_overall.get("time_us", 0) or 0),
        "graph_execute_us": int(qhas_overall.get("graph_execute_us", 0) or 0),
        "dominant_path_cycles": path_cycles,
        "qnn_direct_cycles": qnn_direct_total,
        "unmapped_cycles": max(0, path_cycles - qnn_direct_total),
        "ownership": dict(ownership_totals),
        "resource_totals": dict(full_official_resources),
        "quant_by_category": dict(full_official_quant),
        "quant_total": sum(full_official_quant.values()),
    }
    del qhas_document, qhas_data, dominant_path_rows, htp_instance_rows
    gc.collect()

    with Path(f"{prefix}-chrometrace.json").open() as stream:
        document = json.load(stream)
    events = document if isinstance(document, list) else document.get("traceEvents", [])
    process_names = {
        event.get("pid"): event.get("args", {}).get("name", "")
        for event in events if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    thread_names = {
        (event.get("pid"), event.get("tid")): event.get("args", {}).get("name", "")
        for event in events if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    resource_groups = defaultdict(lambda: {
        "intervals": [], "work": 0, "hardware_active": 0, "dominant": 0,
        "events": 0, "qnn_ops": set(), "htp_types": Counter(), "stages": set(),
    })
    focus_resource_groups = defaultdict(lambda: {
        "intervals": [], "work": 0, "hardware_active": 0, "dominant": 0,
        "events": 0, "qnn_ops": set(), "htp_types": Counter(), "stages": set(),
    })
    conversion_groups = defaultdict(lambda: {
        "intervals": [], "work": 0, "hardware_active": 0, "dominant": 0,
        "events": 0, "qnn_ops": set(), "resources": set(), "dma": False,
    })
    focus_quant_groups = defaultdict(lambda: {
        "intervals": [], "work": 0, "dominant": 0, "events": 0,
    })
    focus_activity_groups = defaultdict(lambda: {
        "critical": [], "offcritical": [],
    })
    seen = set()
    for event in events:
        if event.get("ph") != "X" or not process_names.get(event.get("pid"), "").startswith("QNN::"):
            continue
        args = event.get("args", {})
        qnn_name = args.get("QNN Op Name")
        qnn_type = args.get("QNN Op Type", "")
        duration = int(event.get("dur", 0) or 0)
        start = int(event.get("ts", 0) or 0)
        if not qnn_name or duration <= 0:
            continue
        identity = (args.get("ID"), qnn_name, event.get("tid"), start, duration)
        if identity in seen:
            continue
        seen.add(identity)
        stage, layer, _, _ = classification(qnn_name, qnn_type, num_q_heads, num_kv_heads)
        if layer not in LAYERS:
            continue
        end = start + duration
        flags = tuple(args.get("Flags", []))
        htp_type = args.get("HTP Op Type", event.get("name", ""))
        dominant = int(args.get("Dominant Path Cycles", 0) or 0)
        hardware_active = int(args.get("Duration (cycles)", 0) or 0)
        focus_key = FOCUS_BY_STAGE.get(stage)
        if focus_key:
            activity = focus_activity_groups[(layer, focus_key)]
            activity["critical" if dominant > 0 else "offcritical"].append((start, end))
        resource = classify_resource(
            flags, htp_type, thread_names.get((event.get("pid"), event.get("tid")), "")
        )
        if resource:
            group = resource_groups[(layer, resource)]
            group["intervals"].append((start, end, dominant > 0))
            group["work"] += duration
            group["hardware_active"] += hardware_active
            group["dominant"] += dominant
            group["events"] += 1
            group["qnn_ops"].add(qnn_name)
            group["htp_types"][htp_type] += duration
            group["stages"].add(stage)
            if focus_key:
                focus_group = focus_resource_groups[(layer, focus_key, resource)]
                focus_group["intervals"].append((start, end, dominant > 0))
                focus_group["work"] += duration
                focus_group["hardware_active"] += hardware_active
                focus_group["dominant"] += dominant
                focus_group["events"] += 1
                focus_group["qnn_ops"].add(qnn_name)
                focus_group["htp_types"][htp_type] += duration
                focus_group["stages"].add(stage)
        category = classify_quant_kernel(htp_type, qnn_type, flags)
        if category in CONVERSION_CATEGORIES:
            group = conversion_groups[(layer, category)]
            group["intervals"].append((start, end))
            group["work"] += duration
            group["hardware_active"] += hardware_active
            group["dominant"] += dominant
            group["events"] += 1
            group["qnn_ops"].add(qnn_name)
            group["resources"].add("HMX" if "uses_hmx" in flags else "HVX")
            low_flags = {str(flag).lower() for flag in flags}
            group["dma"] |= "dma" in low_flags or "dma" in htp_type.lower() or "to_vtcm" in htp_type.lower()
            if focus_key:
                quant_group = focus_quant_groups[(layer, focus_key, category)]
                quant_group["intervals"].append((start, end))
                quant_group["work"] += duration
                quant_group["dominant"] += dominant
                quant_group["events"] += 1
    del document, events
    gc.collect()

    resource_layers = {str(layer): [] for layer in LAYERS}
    raw_resource_intervals = defaultdict(list)
    for (layer, resource), group in resource_groups.items():
        segments = merge_annotated(group["intervals"])
        raw_resource_intervals[(layer, resource)] = [(start, end) for start, end, _ in segments]
        label, description = RESOURCE_INFO[resource]
        resource_layers[str(layer)].append({
            "resource": resource,
            "label": label,
            "description": description,
            "segments": segments,
            "start": segments[0][0],
            "end": segments[-1][1],
            "work": group["work"],
            "active": sum(end - start for start, end, _ in segments),
            "wall": segments[-1][1] - segments[0][0],
            "hardware_active": group["hardware_active"],
            "dominant": group["dominant"],
            "events": group["events"],
            "qnn_ops": len(group["qnn_ops"]),
            "stages": len(group["stages"]),
            "top_htp_types": "; ".join(f"{name}: {cycles:,}" for name, cycles in group["htp_types"].most_common(4)),
        })
    overlap = {}
    for layer in LAYERS:
        hmx = raw_resource_intervals[(layer, "hmx")]
        hvx = raw_resource_intervals[(layer, "hvx")]
        dma = raw_resource_intervals[(layer, "dma_transfer")]
        wait = raw_resource_intervals[(layer, "dma_wait")]
        compute = hmx + hvx
        covered = compute + dma
        wait_active = sum(end - start for start, end in merge_intervals(wait))
        wait_group = resource_groups.get((layer, "dma_wait"), {})
        overlap[str(layer)] = {
            "hmx_hvx": intersection_length(hmx, hvx),
            "hmx_dma": intersection_length(hmx, dma),
            "hvx_dma": intersection_length(hvx, dma),
            "wait_compute": intersection_length(wait, compute),
            "wait_uncovered": max(0, wait_active - intersection_length(wait, covered)),
            "wait_dominant": wait_group.get("dominant", 0),
        }
        hmx_hvx_intervals = intersection_intervals(hmx, hvx)
        if hmx_hvx_intervals:
            resource_layers[str(layer)].append({
                "resource": "hmx_hvx_overlap",
                "label": "HMX ∩ HVX overlap",
                "description": "Time windows in which at least one HMX and one HVX event are both active",
                "segments": [[start, end, False] for start, end in hmx_hvx_intervals],
                "start": hmx_hvx_intervals[0][0],
                "end": hmx_hvx_intervals[-1][1],
                "work": sum(end - start for start, end in hmx_hvx_intervals),
                "active": sum(end - start for start, end in hmx_hvx_intervals),
                "wall": hmx_hvx_intervals[-1][1] - hmx_hvx_intervals[0][0],
                "hardware_active": 0, "dominant": 0, "events": 0,
                "qnn_ops": 0, "stages": 0, "top_htp_types": "derived interval intersection",
            })
    resource_order = ("hmx", "hvx", "hmx_hvx_overlap", "dma_transfer", "dma_wait", "sync")
    for values in resource_layers.values():
        values.sort(key=lambda item: resource_order.index(item["resource"]))

    focus_layers = {str(layer): [] for layer in LAYERS}
    for layer in LAYERS:
        for focus_key, focus_label, _ in FOCUS_GROUPS:
            groups = {
                resource: focus_resource_groups.get((layer, focus_key, resource))
                for resource in RESOURCE_ORDER
            }
            groups = {resource: group for resource, group in groups.items() if group}
            if not groups:
                continue
            intervals = {
                resource: [(start, end) for start, end, _ in group["intervals"]]
                for resource, group in groups.items()
            }
            all_intervals = [interval for values in intervals.values() for interval in values]
            all_merged = merge_intervals(all_intervals)
            active_by_resource = {
                resource: sum(end - start for start, end in merge_intervals(values))
                for resource, values in intervals.items()
            }
            dominant_by_resource = {
                resource: group["dominant"] for resource, group in groups.items()
            }
            hmx = intervals.get("hmx", [])
            hvx = intervals.get("hvx", [])
            dma = intervals.get("dma_transfer", [])
            wait = intervals.get("dma_wait", [])
            wait_active = active_by_resource.get("dma_wait", 0)
            wait_uncovered = max(0, wait_active - intersection_length(wait, hmx + hvx + dma))
            total_dominant = sum(dominant_by_resource.values())
            focus_wall = all_merged[-1][1] - all_merged[0][0]
            focus_start, focus_end = all_merged[0][0], all_merged[-1][1]

            # Every point in the envelope belongs to exactly one official QHAS
            # dominant-path segment.  Split that ownership into this operator,
            # another operator in the same layer, another layer, or global/runtime.
            path_ownership = defaultdict(int)
            own_path_intervals = []
            for path_start, path_end, owner_layer, owner_focus in critical_path:
                hit_start = max(focus_start, path_start)
                hit_end = min(focus_end, path_end)
                if hit_start >= hit_end:
                    continue
                hit = hit_end - hit_start
                if owner_layer == layer and owner_focus == focus_key:
                    path_ownership["own"] += hit
                    own_path_intervals.append((hit_start, hit_end))
                elif owner_layer == layer:
                    path_ownership["same_layer_other"] += hit
                elif owner_layer is not None:
                    path_ownership["other_layer"] += hit
                else:
                    path_ownership["global_runtime"] += hit

            owner_direct = focus_owner_dominant[(layer, focus_key)]
            raw_other = sum(path_ownership.values()) - path_ownership["own"]
            calibrated_other = max(0, focus_wall - owner_direct)
            other_scale = calibrated_other / raw_other if raw_other else 0
            owner_scope = {
                key: path_ownership[key] * other_scale
                for key in ("same_layer_other", "other_layer", "global_runtime")
            }

            # Use QHAS HTP-instance flags for the authoritative resource split.
            # Unlike event-level Chrome Trace aggregation, these rows conserve
            # exactly to the QNN-node direct contribution.
            official_by_resource = dict(focus_official_resources[(layer, focus_key)])
            official_quant = dict(focus_official_quant[(layer, focus_key)])
            activity = focus_activity_groups[(layer, focus_key)]
            offcritical_intervals = merge_intervals(activity["offcritical"])
            critical_event_intervals = merge_intervals(activity["critical"])
            offcritical_active = sum(end - start for start, end in offcritical_intervals)
            offcritical_own_overlap = intersection_length(
                offcritical_intervals, own_path_intervals
            )
            offcritical_critical_event_overlap = intersection_length(
                offcritical_intervals, critical_event_intervals
            )
            quant_groups = {
                category: focus_quant_groups.get((layer, focus_key, category))
                for category in CONVERSION_CATEGORIES
            }
            quant_groups = {category: group for category, group in quant_groups.items() if group}
            quant_intervals = [
                interval for group in quant_groups.values() for interval in group["intervals"]
            ]
            quant_by_category = {
                category: group["dominant"] for category, group in quant_groups.items()
            }
            if total_dominant < focus_wall * 0.05:
                signal = "low critical attribution"
            elif dominant_by_resource.get("dma_wait", 0) >= max(
                dominant_by_resource.get("hmx", 0), dominant_by_resource.get("hvx", 0)
            ) and dominant_by_resource.get("dma_wait", 0) > 0:
                signal = "DMA wait / dependency"
            elif dominant_by_resource.get("hvx", 0) > dominant_by_resource.get("hmx", 0) * 1.2:
                signal = "HVX / vector-conversion"
            elif dominant_by_resource.get("hmx", 0) > dominant_by_resource.get("hvx", 0) * 1.2:
                signal = "HMX compute"
            elif intersection_length(hmx, hvx) > 0:
                signal = "HMX–HVX pipelined"
            else:
                signal = "mixed / synchronization"
            focus_layers[str(layer)].append({
                "key": focus_key,
                "label": focus_label,
                "start": all_merged[0][0],
                "end": all_merged[-1][1],
                "active": sum(end - start for start, end in all_merged),
                "wall": focus_wall,
                "dominant": total_dominant,
                "owner_direct": owner_direct,
                "owner_path_intersection": path_ownership["own"],
                "owner_by_resource": official_by_resource,
                "owner_scope": owner_scope,
                "resource_calibration": 1.0,
                "offcritical_active": offcritical_active,
                "offcritical_own_overlap": offcritical_own_overlap,
                "offcritical_critical_event_overlap": offcritical_critical_event_overlap,
                "work": sum(group["work"] for group in groups.values()),
                "active_by_resource": active_by_resource,
                "dominant_by_resource": dominant_by_resource,
                "hmx_hvx_overlap": intersection_length(hmx, hvx),
                "wait_uncovered": wait_uncovered,
                "quant_dominant": min(
                    official_by_resource.get("hvx", 0), sum(official_quant.values())
                ),
                "quant_dominant_raw": sum(group["dominant"] for group in quant_groups.values()),
                "quant_work": sum(group["work"] for group in quant_groups.values()),
                "quant_active": sum(
                    end - start for start, end in merge_intervals(quant_intervals)
                ),
                "quant_by_category": official_quant,
                "signal": signal,
            })

    conversion_layers = {str(layer): [] for layer in LAYERS}
    conversion_all_intervals = []
    for (layer, category), group in conversion_groups.items():
        segments = merge_intervals(group["intervals"])
        resource_text = "+".join(sorted(group["resources"])) or "UNKNOWN"
        execution, _ = resource_style(resource_text)
        conversion_layers[str(layer)].append({
            "category": category,
            "label": CONVERSION_SHORT_LABELS[category],
            "description": CATEGORY_INFO[category][2],
            "segments": segments,
            "start": segments[0][0],
            "end": segments[-1][1],
            "work": group["work"],
            "active": sum(end - start for start, end in segments),
            "wall": segments[-1][1] - segments[0][0],
            "hardware_active": group["hardware_active"],
            "dominant": group["dominant"],
            "events": group["events"],
            "qnn_ops": len(group["qnn_ops"]),
            "resources": resource_text,
            "execution": execution,
            "dma": group["dma"],
        })
        conversion_all_intervals.extend(group["intervals"])
    for values in conversion_layers.values():
        values.sort(key=lambda item: CONVERSION_CATEGORIES.index(item["category"]))
    all_merged = merge_intervals(conversion_all_intervals)
    return {
        "closure": closure,
        "resources": resource_layers,
        "overlap": overlap,
        "focus": focus_layers,
        "conversion": conversion_layers,
        "events": sum(group["events"] for group in conversion_groups.values()),
        "work": sum(group["work"] for group in conversion_groups.values()),
        "active": sum(end - start for start, end in all_merged),
        "dominant": sum(group["dominant"] for group in conversion_groups.values()),
    }


def build_graph(result_dir, graph):
    prefix = result_dir / f"qwen3-sm8750-v79-{graph}"
    all_layer_rows = read_csv(Path(f"{prefix}-qwen3-layer-stage.csv"))
    operator_rows = read_csv(Path(f"{prefix}-qwen3-operator-structure.csv"))
    graph_execute_us = infer_graph_execute_us(all_layer_rows)
    total_dominant = sum(n(row, "num_dominant_path_cycles_htp_0", True) for row in operator_rows)

    selected = [
        cook(row, graph_execute_us, total_dominant, int(row["layer"]))
        for row in all_layer_rows if int(row["layer"]) in LAYERS
    ]
    runtime = cook_runtime_tracks(result_dir, graph)
    start = min(item["start"] for item in selected)
    end = max(item["end"] for item in selected)
    return {
        "id": graph,
        "title": "single-token decode" if graph == "s1" else "32-token chunk",
        "graph_execute_us": graph_execute_us,
        "start": start,
        "end": end,
        "span": end - start,
        "layers": {str(layer): [item for item in selected if item["layer"] == layer] for layer in LAYERS},
        "runtime": runtime,
    }


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3 SM8750 V79 — Bottleneck-focused Structure Gantt</title><style>
:root{--ink:#18263a;--muted:#66768a;--line:#d8e1eb;--paper:#fff;--bg:#eef2f6;--hmx:#2563a6;--hvx:#2b9a80;--fusion:#7655b5;--dma:#d8871e;--wait:#c84d55;--sync:#697586;--critical:#ffd34d;--other:#7c8795}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,"Noto Sans SC",system-ui,sans-serif}header{padding:30px 40px;background:linear-gradient(125deg,#17375e,#23728a);color:#fff}header h1{margin:0 0 7px;font-size:28px}header p{margin:0;color:#d8e8f3}main{max-width:1450px;margin:18px auto;padding:0 18px 40px}.panel{margin:14px 0;padding:18px;background:var(--paper);border:1px solid var(--line);border-radius:12px}.toolbar{display:flex;flex-wrap:wrap;gap:9px;align-items:center}.toolbar .spacer{width:12px}.tab,.toggle{border:1px solid #b8c8d8;background:#fff;color:#264b70;border-radius:7px;padding:7px 12px;font:inherit}.tab.active{background:#226c9c;color:#fff;border-color:#226c9c}.toggle.active{background:#fff3d6;color:#7b4b00;border-color:#e5ae42}.cards{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px;margin-top:14px}.card{padding:12px 14px;border:1px solid var(--line);border-radius:9px}.card small{display:block;color:var(--muted)}.card b{font-size:21px;color:#24557f}.note{padding:10px 13px;border-left:4px solid #e19a27;background:#fff7e9}.legend{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0;color:#42556b}.sw{position:relative;display:inline-block;width:22px;height:11px;margin-right:6px;border-radius:3px;vertical-align:-1px}.scroll{overflow-x:auto}.chart{min-width:1120px}.axis{position:relative;height:34px;margin-left:170px;border-bottom:1px solid #9dafc3}.tick{position:absolute;bottom:0;height:8px;border-left:1px solid #9dafc3}.tick span{position:absolute;bottom:9px;transform:translateX(-50%);white-space:nowrap;color:#53677e;font-size:11px}.phase-row{display:grid;grid-template-columns:162px 1fr;gap:8px;border-bottom:1px solid #edf1f5;min-height:32px}.track{position:relative;background-image:linear-gradient(to right,#edf1f5 1px,transparent 1px);background-size:10% 100%}.bar{position:absolute;min-width:2px;border:1px solid #fff9;border-radius:4px;opacity:.92;cursor:pointer;overflow:hidden}.bar:hover{z-index:20;filter:saturate(1.35);box-shadow:0 0 0 2px #132b4633}.bar span{position:relative;z-index:2;display:block;padding:0 4px;color:#fff;font-size:9px;white-space:nowrap;overflow:hidden}.r-hmx{background-color:var(--hmx)}.r-hvx{background-color:var(--hvx)}.r-fusion{background-color:var(--fusion)}.r-dma{background-color:var(--dma)}.r-wait{background-color:var(--wait)}.r-sync{background-color:var(--sync)}.r-other{background-color:var(--other)}.dma::after{content:"";position:absolute;z-index:1;inset:0;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='7' viewBox='0 0 18 7'%3E%3Cpath d='M0 4 Q3 0 6 4 T12 4 T18 4' fill='none' stroke='white' stroke-width='1.2' opacity='.9'/%3E%3C/svg%3E");background-size:18px 7px;pointer-events:none}.sw.dma::after{border-radius:3px}.bar.critical{border:1px solid #fff9;box-shadow:inset 0 -2px 0 var(--critical)}.sw.critical{background:linear-gradient(to bottom,#27374b 0 70%,var(--critical) 70% 100%)}.local-grid{display:grid;grid-template-columns:1fr;gap:18px}.layer-panel{border:1px solid var(--line);border-radius:10px;overflow:hidden}.layer-panel h3{margin:0;padding:12px 16px;background:#edf3f8;color:#244c72}.phase-label{padding:7px 5px;text-align:right;color:#42566c}.resource-row{background:#f5f9fc;min-height:23px}.resource-row .phase-label{padding:3px 5px;font-size:12px;font-weight:650}.resource-head{display:grid;grid-template-columns:162px 1fr;gap:8px;border-top:2px solid #77a8ce;background:#eef6fb}.resource-head b{padding:8px 5px;text-align:right;color:#25577c}.overlap{padding:6px 8px;display:flex;flex-wrap:wrap;gap:6px}.metric{padding:2px 7px;border:1px solid #cbdce9;border-radius:10px;background:#fff;color:#3e5f78;font-size:11px}.metric.warn{border-color:#e3a8ac;background:#fff3f3;color:#963d43}.quant-row{background:#fffbf2}.quant-row .phase-label{font-size:12px;color:#8a5b16}.quant-head{display:grid;grid-template-columns:162px 1fr;gap:8px;padding-top:7px;border-top:2px solid #f0c76c;background:#fffbf2}.quant-head b{grid-column:1;padding:2px 5px 5px;text-align:right;color:#8a5b16}.bar.quant{border-color:#ffd274;box-shadow:inset 0 0 0 1px #6d430033}.tooltip{position:fixed;z-index:100;display:none;max-width:460px;padding:11px 13px;border-radius:8px;background:#10243d;color:#f5f8fc;white-space:pre-line;pointer-events:none;box-shadow:0 8px 25px #0005;font-size:12px}.boundary{margin-top:12px;color:var(--muted);font-size:12px}
@media(max-width:800px){.cards{grid-template-columns:1fr}header{padding:22px}main{padding:0 8px}}</style></head><body>
<header><h1>Qwen3-1.7B × SM8750 V79：关键路径瓶颈视角</h1><p>Layer 0 / Layer 5 / Layer 27 · binned resource occupancy + critical-path ranking</p></header><main>
<section class="panel"><div class="toolbar"><b>Graph</b><button class="tab active" data-g="s1">s1 · decode</button><button class="tab" data-g="s32">s32 · 32-token</button><span class="spacer"></span><button class="toggle active" id="resource-toggle">显示资源占用热力条</button><button class="toggle" id="quant-toggle">量化/反量化轨道已隐藏</button></div><div class="cards" id="cards"></div><div class="legend"><span><i class="sw r-hmx"></i>HMX occupancy</span><span><i class="sw r-hvx"></i>HVX occupancy</span><span><i class="sw r-fusion"></i>HMX∩HVX overlap</span><span><i class="sw r-dma dma"></i>DMA transfer</span><span><i class="sw r-wait dma"></i>DMA wait</span><span>颜色越深 = 时间窗内占用率越高</span></div><p class="boundary">资源事件被聚合到固定时间窗，避免数千个微事件淹没趋势。热力条用于定位持续占用、并行和等待区域；精确 cycles 与关键路径归因以阶段排名表为准。</p></section>
<section class="panel"><h2>三个 Layer 的结构与资源占用趋势</h2><p class="note">同一横坐标下比较 HMX、HVX、HMX∩HVX、DMA transfer 和 DMA wait。这里展示的是时间窗占用率而非单个 kernel；红色持续且关键路径排名较高的阶段，才是优先检查的等待瓶颈。</p><div class="scroll"><div class="chart"><div class="local-grid" id="locals"></div></div></div></section>
<section class="panel"><h2>关键路径瓶颈排名</h2><div id="diagnosis"></div></section>
<section class="panel"><h2>指标口径</h2><ul><li><b>Resource active union</b>：同一资源所有事件区间取并集；用于看资源实际活跃窗口。</li><li><b>Timeline work</b>：原始事件时长之和，同资源内部也可能重叠，不能作为墙钟延迟。</li><li><b>Uncovered DMA wait</b>：DMA wait 中未与已识别的 HMX、HVX 或 DMA transfer 重叠的区间，是“可能真的在等”的诊断量，不自动等于可优化延迟。</li><li><b>Direct critical</b>：QHAS 的 dominant-path 直接归因。金色边框只表示合并区间包含有关键路径贡献的事件，并不表示整段都在关键路径上。</li></ul></section>
</main><div class="tooltip" id="tip"></div><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),ORDER=['norm','qkv','rope','cache','attention','output','postnorm','mlp'],LABELS=__LABELS__,QCATS=['lpbq_weight_expand_dequant','explicit_convert_requant','fused_linearclip_requant','matmul_signed_conversion'],RORDER=['hmx','hvx','hmx_hvx_overlap','dma_transfer','dma_wait'];let graph='s1',showResources=true,showQuant=false;
function fmt(n,d=2){return Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}function pos(v,s,e){return {l:100*(s-v.start)/v.span,w:Math.max(.15,100*(e-s)/v.span)}}function axis(v){let x='<div class="axis">';for(let i=0;i<=10;i++){const cyc=v.span*i/10,ms=v.graph_execute_us*i/10/1000;x+=`<i class="tick" style="left:${i*10}%"><span>${fmt(cyc/1e6,2)}M cyc<br>${fmt(ms,2)} ms</span></i>`}return x+'</div>'}
function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;')}function execName(s){return s.execution==='fusion'?'HMX + HVX mixed/fused':s.execution.toUpperCase()}function tipText(s){return `${s.label} [${s.stage}]\nEnvelope: ${fmt(s.wall)} cycles\nActive union: ${fmt(s.active)} cycles\nDirect critical: ${fmt(s.dominant)} cycles ≈ ${fmt(s.critical_us,2)} μs\nExecution: ${execName(s)}${s.dma?' + DMA':''}\nResource flags: ${s.resources}`}function quantTip(q){return `${q.label}\n${q.description}\nTimeline work: ${fmt(q.work)} cycles (可重叠，不是 latency)\nActive union: ${fmt(q.active)} cycles\nEnvelope: ${fmt(q.wall)} cycles\nDirect critical: ${fmt(q.dominant)} cycles\nEvents / QNN ops: ${fmt(q.events)} / ${fmt(q.qnn_ops)}\nExecution: ${execName(q)}${q.dma?' + DMA':''}`}function resourceTip(r){return `${r.label}\n${r.description}\nTimeline work: ${fmt(r.work)} cycles (可重叠)\nActive union: ${fmt(r.active)} cycles\nEnvelope: ${fmt(r.wall)} cycles\nHardware active payload: ${fmt(r.hardware_active)} cycles\nDirect critical: ${fmt(r.dominant)} cycles\nEvents / QNN ops / stages: ${fmt(r.events)} / ${fmt(r.qnn_ops)} / ${fmt(r.stages)}\nTop kernels by work:\n${r.top_htp_types}`}
function bind(root){root.querySelectorAll('[data-tip]').forEach(e=>{e.onmouseenter=x=>{const t=document.getElementById('tip');t.textContent=e.dataset.tip;t.style.display='block';move(x)};e.onmousemove=move;e.onmouseleave=()=>document.getElementById('tip').style.display='none'})}function move(e){const t=document.getElementById('tip');t.style.left=Math.min(innerWidth-t.offsetWidth-10,e.clientX+13)+'px';t.style.top=Math.min(innerHeight-t.offsetHeight-10,e.clientY+13)+'px'}
function bar(v,s,top,h=7){const p=pos(v,s.start,s.end),dma=s.dma?' dma':'';return `<i class="bar r-${s.execution}${dma}" style="left:${p.l}%;width:${p.w}%;top:${top}px;height:${h}px" data-tip="${esc(tipText(s))}"></i>`}function qbars(v,q){const dma=q.dma?' dma':'',tip=esc(quantTip(q));return q.segments.map(seg=>{const p=pos(v,seg[0],seg[1]);return `<i class="bar quant r-${q.execution}${dma}" style="left:${p.l}%;width:${p.w}%;top:6px;height:8px" data-tip="${tip}"></i>`}).join('')}function resourceClass(r){return r.resource==='hmx'?'r-hmx':r.resource==='hvx'?'r-hvx':r.resource==='hmx_hvx_overlap'?'r-fusion':r.resource==='dma_transfer'?'r-dma dma':r.resource==='dma_wait'?'r-wait dma':'r-sync'}function heatCells(v,r,bins=80){const width=v.span/bins,base=resourceTip(r);let out='';for(let i=0;i<bins;i++){const start=v.start+i*width,end=start+width;let covered=0,critical=false;for(const seg of r.segments){const hit=Math.max(0,Math.min(end,seg[1])-Math.max(start,seg[0]));covered+=hit;if(hit>0&&seg[2])critical=true}if(covered<=0)continue;const occupancy=Math.min(1,covered/width),opacity=.16+.84*Math.sqrt(occupancy),tip=esc(`${base}\nWindow occupancy: ${fmt(occupancy*100,1)}%\nWindow active: ${fmt(covered)} cycles`);out+=`<i class="bar ${resourceClass(r)}${critical?' critical':''}" style="left:${i*100/bins}%;width:${100/bins+.02}%;top:5px;height:11px;border-radius:1px;opacity:${opacity}" data-tip="${tip}"></i>`}return out}
function cards(g){const all=Object.values(g.runtime.focus).flat(),agg={};for(const x of all){const a=agg[x.key]||(agg[x.key]={label:x.label,dominant:0});a.dominant+=x.dominant}const top=Object.values(agg).sort((a,b)=>b.dominant-a.dominant)[0],wu=Object.values(g.runtime.overlap).reduce((a,x)=>a+x.wait_uncovered,0);document.getElementById('cards').innerHTML=`<div class="card"><small>Graph execute</small><b>${fmt(g.graph_execute_us/1000,3)} ms</b></div><div class="card"><small>三层累计 #1 direct-critical</small><b style="font-size:17px">${top.label}</b><small>${fmt(top.dominant/1e6,3)}M cycles</small></div><div class="card"><small>三层 uncovered DMA wait</small><b>${fmt(wu/1e3,2)}K cycles</b></div>`}
function overlapMetrics(o){return `<span class="metric">HMX∩HVX ${fmt(o.hmx_hvx)}</span><span class="metric warn">uncovered wait ${fmt(o.wait_uncovered)}</span><span class="metric warn">wait direct-critical ${fmt(o.wait_dominant)}</span>`}
function localPanel(g,layer){const items=g.layers[String(layer)],rs=g.runtime.resources[String(layer)],qs=g.runtime.conversion[String(layer)],starts=[...items.map(x=>x.start),...rs.map(x=>x.start),...qs.map(x=>x.start)],ends=[...items.map(x=>x.end),...rs.map(x=>x.end),...qs.map(x=>x.end)],raw0=Math.min(...starts),raw1=Math.max(...ends),pad=(raw1-raw0)*.035,v={...g,start:raw0-pad,end:raw1+pad,span:(raw1-raw0)*1.07,graph_execute_us:g.graph_execute_us*((raw1-raw0)*1.07)/g.span};let x=`<div class="layer-panel"><h3>Layer ${layer}</h3>${axis(v)}`;for(const key of ORDER){const subset=items.filter(s=>s.group===key).sort((a,b)=>a.order-b.order),height=Math.max(32,12+subset.length*7);x+=`<div class="phase-row" style="min-height:${height}px"><div class="phase-label">${LABELS[key]}</div><div class="track">`;subset.forEach((s,i)=>x+=bar(v,s,5+i*7,7));x+='</div></div>'}if(showResources){x+=`<div class="resource-head"><b>资源占用趋势</b><div class="overlap">${overlapMetrics(g.runtime.overlap[String(layer)])}</div></div>`;for(const resource of RORDER){const r=rs.find(x=>x.resource===resource);if(!r)continue;x+=`<div class="phase-row resource-row"><div class="phase-label">${r.label}</div><div class="track">${heatCells(v,r)}</div></div>`}}if(showQuant){x+='<div class="quant-head"><b>量化 / 反量化活动</b><span></span></div>';for(const category of QCATS){const q=qs.find(x=>x.category===category);if(!q)continue;x+=`<div class="phase-row quant-row" style="min-height:21px"><div class="phase-label">↳ ${q.label}</div><div class="track">${qbars(v,q)}</div></div>`}}return x+'</div>'}
function diagnosis(g){const layers=['0','5','27'],rows=layers.map(l=>{const rs=g.runtime.resources[l],get=k=>(rs.find(x=>x.resource===k)||{active:0}).active,o=g.runtime.overlap[l];return `<tr><td>Layer ${l}</td><td>${fmt(get('hmx'))}</td><td>${fmt(get('hvx'))}</td><td>${fmt(get('dma_transfer'))}</td><td>${fmt(get('dma_wait'))}</td><td>${fmt(o.hmx_hvx)}</td><td>${fmt(o.hmx_dma)}</td><td>${fmt(o.hvx_dma)}</td><td>${fmt(o.wait_uncovered)}</td><td>${fmt(o.wait_dominant)}</td></tr>`}).join(''),noHmxDma=layers.every(l=>g.runtime.overlap[l].hmx_dma===0),signal=noHmxDma?'三个采样 Layer 均未观察到 HMX 与 DMA transfer 同时活跃；DMA 与 HVX 存在重叠。结合非零 wait direct-critical，当前更值得检查的是 DMA/HVX 准备阶段到 HMX 消费之间的串行依赖。':'已观察到 HMX 与 DMA transfer 重叠，需按具体 Layer 检查是否有效隐藏搬运。';document.getElementById('diagnosis').innerHTML=`<p class="note">${signal} Optrace 能定位等待窗口，但仅凭现有依赖字段还不能把每个 wait 唯一归因到某一个 HMX/HVX consumer。</p><div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px"><thead><tr>${['Layer','HMX active','HVX active','DMA active','DMA wait active','HMX∩HVX','HMX∩DMA','HVX∩DMA','Uncovered wait','Wait direct-critical'].map(x=>`<th style="padding:6px;border:1px solid #d8e1eb;background:#edf3f8;white-space:nowrap">${x}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`;document.querySelectorAll('#diagnosis td').forEach(x=>{x.style.padding='6px';x.style.border='1px solid #d8e1eb';x.style.textAlign='right';x.style.whiteSpace='nowrap'})}
function bottlenecks(g){const layers=['0','5','27'],sections=layers.map(layer=>{const phases=[...g.runtime.focus[layer]].sort((a,b)=>b.dominant-a.dominant),total=phases.reduce((a,x)=>a+x.dominant,0),rows=phases.slice(0,5).map((x,i)=>{const d=x.dominant_by_resource,a=x.active_by_resource,breakdown=`HMX ${fmt(d.hmx||0)} / HVX ${fmt(d.hvx||0)} / wait ${fmt(d.dma_wait||0)}`;return `<tr><td>${i+1}</td><td style="text-align:left">${x.label}</td><td>${fmt(x.dominant)}</td><td>${fmt(100*x.dominant/Math.max(total,1),1)}%</td><td style="text-align:left">${breakdown}</td><td>${fmt(x.active)}</td><td>${fmt(x.hmx_hvx_overlap)}</td><td>${fmt(x.wait_uncovered)}</td><td style="text-align:left">${x.signal}</td></tr>`}).join('');return `<h3>Layer ${layer}</h3><div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px"><thead><tr>${['#','Qwen3 phase','Direct critical','Layer share','Critical breakdown','Active union','HMX∩HVX','Uncovered wait','Primary signal'].map(x=>`<th style="padding:6px;border:1px solid #d8e1eb;background:#edf3f8;white-space:nowrap">${x}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`}).join(''),all=layers.flatMap(l=>g.runtime.focus[l]),aggregate={};for(const x of all){const a=aggregate[x.key]||(aggregate[x.key]={label:x.label,dominant:0,wait:0});a.dominant+=x.dominant;a.wait+=x.wait_uncovered}const top=Object.values(aggregate).sort((a,b)=>b.dominant-a.dominant)[0];document.getElementById('diagnosis').innerHTML=`<p class="note">三个采样 Layer 累计 direct-critical 最高的是 <b>${top.label}</b>（${fmt(top.dominant)} cycles）。排名按事件级 dominant-path 直接归因，而不是 work cycles；优先检查排名高且 uncovered wait 也高的阶段。</p>${sections}`;document.querySelectorAll('#diagnosis td').forEach(x=>{x.style.padding='6px';x.style.border='1px solid #d8e1eb';x.style.textAlign=x.style.textAlign||'right';x.style.whiteSpace='nowrap'})}
function locals(g){const el=document.getElementById('locals');el.innerHTML=[0,5,27].map(l=>localPanel(g,l)).join('');bind(el)}function render(){const g=D[graph];cards(g);locals(g);bottlenecks(g)}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{graph=b.dataset.g;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));render()});document.getElementById('resource-toggle').onclick=e=>{showResources=!showResources;e.currentTarget.classList.toggle('active',showResources);e.currentTarget.textContent=showResources?'显示资源占用热力条':'资源占用热力条已隐藏';locals(D[graph])};document.getElementById('quant-toggle').onclick=e=>{showQuant=!showQuant;e.currentTarget.classList.toggle('active',showQuant);e.currentTarget.textContent=showQuant?'显示量化/反量化轨道':'量化/反量化轨道已隐藏';locals(D[graph])};render();
</script></body></html>'''


COMPOSITION_TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3 SM8750 V79 — Critical-path Ownership</title><style>
:root{--ink:#18263a;--muted:#68788c;--line:#d8e1eb;--paper:#fff;--bg:#eef2f6;--hmx:#2868aa;--hvx:#2a9b80;--dma:#d98a22;--other:#c9d1db;--quant:#f4d35e;--l0:#2868aa;--l5:#2a9b80;--l27:#7655b5;--global:#d98a22;--missing:#cf5b62}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,"Noto Sans SC",system-ui,sans-serif}header{padding:28px 40px;background:linear-gradient(125deg,#17375e,#23728a);color:#fff}h1{margin:0 0 6px;font-size:27px}header p{margin:0;color:#d9e9f3}main{max-width:1540px;margin:16px auto;padding:0 16px 40px}.panel{margin:14px 0;padding:17px;background:var(--paper);border:1px solid var(--line);border-radius:12px}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px}.tab,.sort{border:1px solid #b9c8d8;background:#fff;color:#285174;border-radius:7px;padding:7px 12px;font:inherit}.tab.active,.sort.active{background:#246f9e;color:#fff;border-color:#246f9e}.spacer{width:14px}.cards{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px;margin-top:13px}.card{padding:11px 13px;border:1px solid var(--line);border-radius:9px}.card small{display:block;color:var(--muted)}.card b{display:block;color:#24557f;font-size:19px}.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:12px;color:#44586e}.sw{display:inline-block;width:24px;height:11px;margin-right:5px;border-radius:3px;vertical-align:-1px}.hmx{background:var(--hmx)}.hvx{background:var(--hvx)}.dma{background:var(--dma)}.other{background:var(--other)}.quant-demo{position:relative;background:var(--hvx)}.quant-demo::after,.quant-overlay{content:"";position:absolute;inset:0;background:repeating-linear-gradient(135deg,transparent 0 3px,var(--quant) 3px 5px)}.note{padding:10px 13px;border-left:4px solid #e19a27;background:#fff7e9}.closure{margin-top:13px;padding:12px;border:1px solid var(--line);border-radius:9px;background:#f8fafc}.closure-bar{display:flex;height:18px;overflow:hidden;border-radius:4px;box-shadow:0 0 0 1px #8292a333}.closure-labels{display:flex;flex-wrap:wrap;gap:12px;margin-top:7px;font-size:11px;color:#52667c}.closure-labels i{display:inline-block;width:9px;height:9px;margin-right:4px;border-radius:2px}.layer{margin:14px 0;border:1px solid var(--line);border-radius:10px;overflow:hidden}.layer-head{display:flex;flex-wrap:wrap;gap:18px;align-items:baseline;padding:10px 14px;background:#eaf2f8}.layer-head h3{margin:0;color:#244f76}.layer-head span{color:#5f7184;font-size:12px}.op-head,.op-row{display:grid;grid-template-columns:38px 185px 110px minmax(390px,1fr) 82px 86px 105px 120px 140px;align-items:center}.op-head{padding:7px 9px;background:#f5f7fa;border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px;font-weight:700;color:#53677d}.op-head>span,.op-row>span{padding:0 5px}.op-row{min-height:38px;padding:4px 9px;border-bottom:1px solid #edf1f5}.op-row:last-child{border-bottom:0}.rank{text-align:center;color:#708196}.op-name{font-weight:650}.number{text-align:right;font-variant-numeric:tabular-nums}.sub{display:block;color:var(--muted);font-size:10px}.scale{position:relative;height:18px;background-image:linear-gradient(to right,#e8edf2 1px,transparent 1px);background-size:10% 100%}.latency{position:relative;display:flex;height:16px;min-width:2px;border-radius:3px;overflow:hidden;box-shadow:0 0 0 1px #8796a633}.seg{position:relative;height:100%}.seg-hmx{background:var(--hmx)}.seg-hvx{background:var(--hvx)}.seg-dma{background:var(--dma)}.seg-other{background:var(--other)}.quant-overlay{right:auto}.signal{font-size:11px;color:#4d6175}.signal.wait{color:#a33e43;font-weight:700}.tip{position:fixed;z-index:100;display:none;max-width:490px;padding:11px 13px;background:#10243d;color:#f6f8fb;border-radius:8px;white-space:pre-line;pointer-events:none;box-shadow:0 8px 24px #0005;font-size:12px}.foot{color:var(--muted);font-size:12px}.pie-grid{display:grid;grid-template-columns:repeat(2,minmax(420px,1fr));gap:14px;margin-top:13px}.pie-card{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center;padding:14px;border:1px solid var(--line);border-radius:10px;background:#fafcfe}.pie-card h3{grid-column:1/-1;margin:0;color:#244f76}.pie-card svg{display:block;width:220px;height:220px}.pie-legend{display:grid;gap:8px}.pie-legend .item{display:grid;grid-template-columns:13px 1fr auto;gap:7px;align-items:center}.pie-legend .dot{width:12px;height:12px;border-radius:3px}.pie-legend small{grid-column:2/-1;color:var(--muted)}@media(max-width:1100px){.op-head,.op-row{min-width:1280px}.cards{grid-template-columns:1fr}.layer{overflow-x:auto}.pie-grid{grid-template-columns:1fr}}@media(max-width:620px){.pie-card{grid-template-columns:1fr}.pie-card svg{margin:auto}}</style></head><body>
<header><h1>Qwen3-1.7B × SM8750 V79：关键路径所有权 × 算子 E2E</h1><p>mutually-exclusive QHAS direct contribution · overlap-aware envelope · quantization subset</p></header><main>
<section class="panel"><div class="toolbar"><b>Graph</b><button class="tab active" data-g="s1">s1 · decode</button><button class="tab" data-g="s32">s32 · 32-token</button><span class="spacer"></span><b>排序</b><button class="sort active" data-sort="structure">Qwen3 顺序</button><button class="sort" data-sort="e2e">E2E envelope</button><button class="sort" data-sort="critical">Own critical</button></div><div class="cards" id="cards"></div><div class="closure" id="closure"></div><div class="legend"><span><i class="sw hmx"></i>Own HMX critical</span><span><i class="sw hvx"></i>Own HVX critical</span><span><i class="sw dma"></i>Own DMA critical（transfer + wait + sync）</span><span><i class="sw other"></i>Other-owner critical inside envelope</span><span><i class="sw quant-demo"></i>斜纹 = own HVX 中可见的量化/转换子集</span></div></section>
<section class="panel"><p class="note">每条 bar 的总宽度仍是语义算子的 E2E envelope；内部前三段是该算子互斥拥有的 QHAS direct-critical contribution，灰色是同一时间窗内由同 Layer 其他算子、其他 Layer 或 global/runtime 持有的关键路径。四段严格闭合到该 envelope，但不同算子的 envelope 仍会重叠，不能相加。</p><div id="layers"></div></section>
<section class="panel"><h2>三个 Layer 的局部甘特图</h2><p class="note">横向位置使用真实 event 起止时间；条内颜色只表达 critical ownership 构成，不表示 HMX/HVX/DMA 在条内的实际先后位置。非关键活动使用 <code>Dominant Path Cycles=0</code> 的事件区间 union 单独统计，不再用灰色 residual 反推。</p><div id="gantts"></div></section>
<section class="panel"><h2>口径边界</h2><ul><li>全图 closure 直接比较官方 <code>Dominant Path HTP0</code> 与 QNN-node direct contribution；当前覆盖约 99.4%，小额差值保留为 unmapped/system。</li><li>每个 QNN node 只映射到一个 Qwen3 stage，因此 own critical 可以跨算子相加；E2E envelope、active union 和 work cycles 不能相加为 latency。</li><li>HMX/HVX/DMA 拆分直接来自 QHAS <code>htp_op_instances</code> 的官方资源标志，其 direct cycles 与 QNN-node contribution 严格守恒；Chrome Trace event direct 仅作为交叉检查。</li><li>Off-critical overlap 是非关键事件区间与该算子 own-DP 区间的真实交集；它是并行活动量，不额外增加墙钟时间。</li><li>量化斜纹是 own HVX critical 的子集，包含 W4 expand/dequant、Convert/CastType、fused requant 和 signed conversion。</li></ul></section>
<section class="panel"><h2>关键路径资源总构成</h2><p class="note">四张饼图均只统计互斥的 QHAS QNN-node direct-critical cycles。全图包括 28 Layers 与 embedding/final norm/lm_head/runtime 等 global nodes；Layer 图汇总该 Layer 的 19 个 Qwen3 语义算子。量化斜纹是 HVX 扇区中的可见转换子集，不是额外开销。</p><div class="legend"><span><i class="sw hmx"></i>Own HMX critical</span><span><i class="sw hvx"></i>Own HVX critical</span><span><i class="sw dma"></i>Own DMA critical（transfer + wait + sync）</span><span><i class="sw quant-demo"></i>斜纹 = own HVX 中可见的量化/转换子集</span></div><div class="pie-grid" id="pies"></div></section>
</main><div class="tip" id="tip"></div><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),QNAME={lpbq_weight_expand_dequant:'W4 expand/dequant',explicit_convert_requant:'Convert/CastType',fused_linearclip_requant:'Fused requant',matmul_signed_conversion:'Signed conversion'};let graph='s1',sortMode='structure';
function fmt(n,d=2){return Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;')}function pct(n,d){return 100*n/Math.max(d,1)}function approxUs(g,cycles){const c=g.runtime.closure;return cycles*c.graph_execute_us/Math.max(c.dominant_path_cycles,1)}
function parts(x){const d=x.owner_by_resource||{},hmx=d.hmx||0,hvx=d.hvx||0,dma=(d.dma_transfer||0)+(d.dma_wait||0)+(d.sync||0),other=Math.max(0,x.wall-hmx-hvx-dma);return {hmx,hvx,dma,other}}
function quantText(x){const rows=Object.entries(x.quant_by_category||{}).sort((a,b)=>b[1]-a[1]);return rows.length?rows.map(([k,v])=>`${QNAME[k]}: ${fmt(v)} QHAS critical cycles`).join('\n'):'No standalone visible quant/conversion critical event'}
function tooltip(g,x,p){const s=x.owner_scope||{};return `${x.label}\nE2E envelope: ${fmt(x.wall)} cycles ≈ ${fmt(approxUs(g,x.wall),2)} μs\nActive union: ${fmt(x.active)} cycles\n\nMUTUALLY-EXCLUSIVE OWN CRITICAL\nQNN-node direct: ${fmt(x.owner_direct)} cycles (${fmt(pct(x.owner_direct,x.wall),1)}% envelope; ${fmt(pct(x.owner_direct,g.runtime.closure.dominant_path_cycles),3)}% full graph)\nHMX: ${fmt(p.hmx)}\nHVX: ${fmt(p.hvx)}\nDMA: ${fmt(p.dma)}\nQHAS HTP-instance resource sum: ${fmt(p.hmx+p.hvx+p.dma)}\nChrome Trace event direct (cross-check only): ${fmt(x.dominant)}\n\nOTHER CRITICAL OWNER INSIDE ENVELOPE\nSame-layer other op: ${fmt(s.same_layer_other||0)}\nOther layer: ${fmt(s.other_layer||0)}\nGlobal/runtime: ${fmt(s.global_runtime||0)}\n\nOFF-CRITICAL ACTIVITY (not added to latency)\nActive union: ${fmt(x.offcritical_active)}\nOverlap with own DP: ${fmt(x.offcritical_own_overlap)}\nOverlap with this op's critical-event intervals: ${fmt(x.offcritical_critical_event_overlap)}\n\nVisible quant critical: ${fmt(x.quant_dominant)} (subset of official HVX)\n${quantText(x)}`}
function bind(root){root.querySelectorAll('[data-tip]').forEach(e=>{e.onmouseenter=x=>{const t=document.getElementById('tip');t.textContent=e.dataset.tip;t.style.display='block';move(x)};e.onmousemove=move;e.onmouseleave=()=>document.getElementById('tip').style.display='none'})}function move(e){const t=document.getElementById('tip');t.style.left=Math.min(innerWidth-t.offsetWidth-10,e.clientX+13)+'px';t.style.top=Math.min(innerHeight-t.offsetHeight-10,e.clientY+13)+'px'}
function fill(g,x){const p=parts(x),hmx=pct(p.hmx,x.wall),hvx=pct(p.hvx,x.wall),dma=pct(p.dma,x.wall),other=pct(p.other,x.wall),qratio=Math.min(100,pct(x.quant_dominant,Math.max(p.hvx,1))),tip=esc(tooltip(g,x,p));return {p,tip,html:`<i class="seg seg-hmx" style="width:${hmx}%"></i><i class="seg seg-hvx" style="width:${hvx}%"><i class="quant-overlay" style="width:${qratio}%"></i></i><i class="seg seg-dma" style="width:${dma}%"></i><i class="seg seg-other" style="width:${other}%"></i>`}}
function bar(g,x,maxWall){const f=fill(g,x),outer=100*x.wall/maxWall;return `<div class="scale"><div class="latency" style="width:${outer}%" data-tip="${f.tip}">${f.html}</div></div>`}
function ganttBar(g,x,v){const f=fill(g,x),left=pct(x.start-v.start,v.span),width=Math.max(.18,pct(x.wall,v.span));return `<div class="latency" style="position:absolute;left:${left}%;width:${width}%;top:5px" data-tip="${f.tip}">${f.html}</div>`}
function ganttAxis(g,v){let ticks='';for(let i=0;i<=8;i++){const cycles=v.span*i/8,us=approxUs(g,cycles);ticks+=`<i style="position:absolute;left:${i*12.5}%;bottom:0;height:8px;border-left:1px solid #9dafc3"><span style="position:absolute;bottom:9px;transform:translateX(-50%);white-space:nowrap;color:#5d7085;font-size:10px">${fmt(cycles/1e6,2)}M cyc<br>${fmt(us,1)} μs</span></i>`}return `<div style="display:grid;grid-template-columns:250px 1fr;gap:8px"><span></span><div style="position:relative;height:38px;border-bottom:1px solid #9dafc3">${ticks}</div></div>`}
function ganttLayer(g,id){const rows=g.runtime.focus[id],raw0=Math.min(...rows.map(x=>x.start)),raw1=Math.max(...rows.map(x=>x.end)),pad=(raw1-raw0)*.025,v={start:raw0-pad,end:raw1+pad,span:(raw1-raw0)*1.05};let body=ganttAxis(g,v);rows.forEach((x,i)=>{body+=`<div style="display:grid;grid-template-columns:250px 1fr;gap:8px;min-height:31px;border-bottom:1px solid #edf1f5"><span style="padding:4px 6px;text-align:right;font-weight:650;color:#40576e">${i+1}. ${x.label}<small class="sub">own ${fmt(x.owner_direct)} · offcrit∩own ${fmt(x.offcritical_own_overlap)}</small></span><div class="scale" style="height:31px">${ganttBar(g,x,v)}</div></div>`});const layerDirect=rows.reduce((a,x)=>a+x.owner_direct,0);return `<section class="layer"><div class="layer-head"><h3>Layer ${id}</h3><span>Local envelope ${fmt(raw1-raw0)} cycles</span><span>Mutually-exclusive layer contribution <b>${fmt(layerDirect)}</b> cycles · ${fmt(pct(layerDirect,g.runtime.closure.dominant_path_cycles),3)}% full graph</span></div><div style="min-width:1180px;padding:0 9px 9px">${body}</div></section>`}
function ordered(rows){const copy=[...rows];if(sortMode==='e2e')copy.sort((a,b)=>b.wall-a.wall);else if(sortMode==='critical')copy.sort((a,b)=>b.owner_direct-a.owner_direct);return copy}
function layer(g,id,maxWall){const original=g.runtime.focus[id],rows=ordered(original),topE=[...original].sort((a,b)=>b.wall-a.wall)[0],topC=[...original].sort((a,b)=>b.owner_direct-a.owner_direct)[0],topQ=[...original].sort((a,b)=>b.quant_dominant-a.quant_dominant)[0],layerDirect=original.reduce((a,x)=>a+x.owner_direct,0);let html=`<section class="layer"><div class="layer-head"><h3>Layer ${id}</h3><span>Layer direct <b>${fmt(layerDirect)}</b> cycles · ${fmt(pct(layerDirect,g.runtime.closure.dominant_path_cycles),3)}% full graph</span><span>Top own critical: <b>${topC.label}</b> ${fmt(topC.owner_direct)}</span><span>Top E2E: <b>${topE.label}</b> ${fmt(topE.wall)}</span><span>Top quant: <b>${topQ.label}</b> ${fmt(topQ.quant_dominant)}</span></div><div class="op-head"><span>#</span><span>Qwen3 operator</span><span style="text-align:right">E2E</span><span>Critical ownership inside envelope（shared scale）</span><span style="text-align:right">Own DP</span><span style="text-align:right">Graph DP</span><span style="text-align:right">Quant</span><span style="text-align:right">Offcrit ∩ own</span><span>Bottleneck signal</span></div>`;rows.forEach((x,i)=>{const signalClass=x.signal.includes('wait')?'signal wait':'signal';html+=`<div class="op-row"><span class="rank">${i+1}</span><span class="op-name">${x.label}<small class="sub">direct ${fmt(x.owner_direct)} cycles</small></span><span class="number">${fmt(x.wall)}<small class="sub">≈ ${fmt(approxUs(g,x.wall),2)} μs</small></span>${bar(g,x,maxWall)}<span class="number">${fmt(pct(x.owner_direct,x.wall),1)}%</span><span class="number">${fmt(pct(x.owner_direct,g.runtime.closure.dominant_path_cycles),3)}%</span><span class="number">${fmt(x.quant_dominant)}<small class="sub">${fmt(pct(x.quant_dominant,Math.max((x.owner_by_resource||{}).hvx||0,1)),1)}% HVX</small></span><span class="number">${fmt(x.offcritical_own_overlap)}<small class="sub">${fmt(pct(x.offcritical_own_overlap,x.owner_direct),1)}% own DP</small></span><span class="${signalClass}">${x.signal}</span></div>`});return html+'</section>'}
function closurePanel(g){const c=g.runtime.closure,o=c.ownership,total=c.dominant_path_cycles,items=[['Layer 0',o.layer_0||0,'var(--l0)'],['Layer 5',o.layer_5||0,'var(--l5)'],['Layer 27',o.layer_27||0,'var(--l27)'],['Other layers',o.other_layers||0,'#8998a9'],['Global/runtime',(o.global_runtime||0)+(o.selected_layer_other||0),'var(--global)'],['Unmapped',c.unmapped_cycles,'var(--missing)']];document.getElementById('closure').innerHTML=`<b>Full-graph critical-path ownership closure</b><div class="closure-bar">${items.map(x=>`<i style="width:${pct(x[1],total)}%;background:${x[2]}" title="${x[0]} ${fmt(x[1])} cycles"></i>`).join('')}</div><div class="closure-labels">${items.map(x=>`<span><i style="background:${x[2]}"></i>${x[0]} ${fmt(x[1]/1e6,3)}M (${fmt(pct(x[1],total),3)}%)</span>`).join('')}</div>`}
function cards(g,all){const c=g.runtime.closure,coverage=pct(c.qnn_direct_cycles,c.dominant_path_cycles),selected=all.reduce((a,x)=>a+x.owner_direct,0),top=[...all].sort((a,b)=>b.owner_direct-a.owner_direct)[0];document.getElementById('cards').innerHTML=`<div class="card"><small>Graph execute / QHAS timeline</small><b>${fmt(c.graph_execute_us/1000,3)} / ${fmt(c.qhas_time_us/1000,3)} ms</b><small>${fmt(c.dominant_path_cycles)} DP cycles</small></div><div class="card"><small>QNN-node direct closure</small><b>${fmt(coverage,3)}%</b><small>${fmt(c.qnn_direct_cycles)} mapped + ${fmt(c.unmapped_cycles)} unmapped</small></div><div class="card"><small>Selected 3 layers / top operator</small><b>${fmt(pct(selected,c.dominant_path_cycles),3)}% full DP</b><small>${top.label}: ${fmt(top.owner_direct)} cycles</small></div>`;closurePanel(g)}
function piePoint(p,r=47){const a=p*Math.PI*2/100-Math.PI/2;return [50+r*Math.cos(a),50+r*Math.sin(a)]}function pieSector(a,b,color,extra=''){if(b<=a)return '';if(b-a>=99.999)return `<circle cx="50" cy="50" r="47" fill="${color}" ${extra}/>`;const p0=piePoint(a),p1=piePoint(b),large=b-a>50?1:0;return `<path d="M50 50 L${p0[0]} ${p0[1]} A47 47 0 ${large} 1 ${p1[0]} ${p1[1]} Z" fill="${color}" ${extra}/>`}
function pieValues(g,id){if(id==='all'){const d=g.runtime.closure.resource_totals||{},hmx=d.hmx||0,hvx=d.hvx||0,dma=(d.dma_transfer||0)+(d.dma_wait||0)+(d.sync||0),unresolved=d.unresolved||0;return {title:'Full graph · 28 Layers + global',hmx,hvx,dma,quant:Math.min(hvx,g.runtime.closure.quant_total||0),owner:g.runtime.closure.qnn_direct_cycles,unresolved}}const rows=g.runtime.focus[id],sum=k=>rows.reduce((a,x)=>a+((x.owner_by_resource||{})[k]||0),0),hmx=sum('hmx'),hvx=sum('hvx'),dma=sum('dma_transfer')+sum('dma_wait')+sum('sync'),unresolved=sum('unresolved');return {title:`Layer ${id}`,hmx,hvx,dma,quant:Math.min(hvx,rows.reduce((a,x)=>a+x.quant_dominant,0)),owner:rows.reduce((a,x)=>a+x.owner_direct,0),unresolved}}
function pieCard(g,id,index){const x=pieValues(g,id),total=x.hmx+x.hvx+x.dma,h=pct(x.hmx,total),v=pct(x.hvx,total),d=pct(x.dma,total),q=pct(x.quant,total),pattern=`quant-hatch-${g.id}-${index}`,segments=pieSector(0,h,'var(--hmx)')+pieSector(h,h+v,'var(--hvx)')+pieSector(h+v,100,'var(--dma)')+pieSector(h,h+q,`url(#${pattern})`,'opacity=".95"'),items=[['Own HMX critical',x.hmx,'var(--hmx)'],['Own HVX critical',x.hvx,'var(--hvx)'],['Own DMA critical',x.dma,'var(--dma)']];return `<article class="pie-card"><h3>${x.title}</h3><svg viewBox="0 0 100 100" role="img" aria-label="${x.title} critical resource pie"><defs><pattern id="${pattern}" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(35)"><line x1="0" y1="0" x2="0" y2="5" stroke="var(--quant)" stroke-width="2.2"/></pattern></defs>${segments}<circle cx="50" cy="50" r="47" fill="none" stroke="#fff" stroke-width="1"/></svg><div class="pie-legend">${items.map(y=>`<div class="item"><i class="dot" style="background:${y[2]}"></i><b>${y[0]}</b><span>${fmt(y[1])} · ${fmt(pct(y[1],total),2)}%</span></div>`).join('')}<div class="item"><i class="dot quant-demo"></i><b>Visible quant/conversion</b><span>${fmt(x.quant)} · ${fmt(pct(x.quant,x.hvx),2)}% HVX</span><small>HVX 子集；不额外加入总量</small></div><small>Pie denominator: ${fmt(total)} resource-classified QNN direct cycles · owner coverage ${fmt(pct(total,x.owner),4)}%${x.unresolved?` · unresolved ${fmt(x.unresolved)}`:''}</small></div></article>`}
function pies(g){document.getElementById('pies').innerHTML=['all','0','5','27'].map((id,i)=>pieCard(g,id,i)).join('')}
function render(){const g=D[graph],ids=['0','5','27'],all=ids.flatMap(x=>g.runtime.focus[x]),maxWall=Math.max(...all.map(x=>x.wall));cards(g,all);const el=document.getElementById('layers');el.innerHTML=ids.map(x=>layer(g,x,maxWall)).join('');const ge=document.getElementById('gantts');ge.innerHTML=ids.map(x=>ganttLayer(g,x)).join('');pies(g);bind(el);bind(ge)}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{graph=b.dataset.g;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));render()});document.querySelectorAll('.sort').forEach(b=>b.onclick=()=>{sortMode=b.dataset.sort;document.querySelectorAll('.sort').forEach(x=>x.classList.toggle('active',x===b));render()});render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {graph: build_graph(args.results_dir, graph) for graph in ("s1", "s32")}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    labels = json.dumps(GROUP_LABELS, ensure_ascii=False, separators=(",", ":"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        COMPOSITION_TEMPLATE.replace("__DATA__", payload).replace("__LABELS__", labels),
        encoding="utf-8",
    )
    for graph, value in data.items():
        print(f"{graph}: graph_execute_us={value['graph_execute_us']:.0f} span={value['span']} displayed_layers={list(value['layers'])}")
    print(f"generated: {args.output}")


if __name__ == "__main__":
    main()
