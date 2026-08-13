#!/usr/bin/env python3
"""Add a quantization-oriented view to QNN HTP Optrace reports.

The report deliberately separates three layers which are easy to conflate:

* compile-time logical recipes (for example LPBQ W4, A16, O16, group 16),
* pre-finalize QNN tensor encodings (scale/zero-point/block metadata), and
* post-lowering HTP kernels and their physical output data types.

Kernel work cycles are additive work, not latency.  Dominant-path cycles and
interval unions are emitted alongside them so overlapping HVX/DMA/HMX work is
not accidentally summed as wall-clock latency.
"""

import argparse
import csv
import gc
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from qnn_optrace_qwen3_structure import STAGE_INFO, classification, infer_head_counts, load_qhas
from qnn_optrace_summary import interval_stats


CATEGORY_INFO = {
    "lpbq_weight_expand_dequant": (
        10, "LPBQ block-weight expand/dequant", "HVX expands logical W4 block-scaled weights to physical per-channel QInt8"
    ),
    "lpbq_weight_dma_wait": (20, "LPBQ weight DMA / wait", "Weight transfer/staging to VTCM; can overlap expansion and MAC"),
    "lpbq_bias_dma_wait": (30, "LPBQ bias DMA / wait", "Bias transfer/staging to VTCM"),
    "lpbq_hmx_mac": (40, "LPBQ HMX MAC", "Matrix/conv arithmetic after weight preparation"),
    "lpbq_sync_checkpoint": (50, "LPBQ synchronization", "DMA sets, checkpoints, waits and scheduling glue"),
    "lpbq_other_fused": (60, "LPBQ other fused work", "Other lowered work owned by Conv2d_w_blk_exp_scale"),
    "explicit_convert_requant": (70, "Explicit Convert/CastType", "Graph-visible type conversion, quantize/dequantize or requantize"),
    "fused_linearclip_requant": (80, "Fused linear clip/requant", "Consumer-fused clamp and output requantization"),
    "matmul_signed_conversion": (90, "MatMul signed conversion", "HTP-internal conversion/shuffle into signed operands"),
    "other_quant_kernel": (100, "Other quantization-related kernel", "Quantization-related HTP name not covered above"),
}

CONTROL_FLAGS = {"sync", "dma_wait", "dma_set"}


def physical_output_dtype(args):
    dtype = args.get("Data Type")
    if dtype:
        return dtype
    flags = set(args.get("Flags", []))
    if args.get("Rank") == 0 or flags & CONTROL_FLAGS:
        return "CONTROL(no tensor)"
    return "UNSPECIFIED"


def classify_quant_kernel(htp_type, qnn_type, flags):
    low = htp_type.lower()
    qnn_low = (qnn_type or "").lower()
    if qnn_type == "Conv2d_w_blk_exp_scale":
        if "expand_block_quant" in low:
            return "lpbq_weight_expand_dequant"
        if "weights_to_vtcm" in low or ("weight" in low and ("dma" in low or "wait" in low)):
            return "lpbq_weight_dma_wait"
        if "bias_to_vtcm" in low or ("bias" in low and ("dma" in low or "wait" in low)):
            return "lpbq_bias_dma_wait"
        if "convlayer_s1.opt" in low or "uses_hmx" in flags:
            return "lpbq_hmx_mac"
        if any(token in low for token in ("checkpoint", "dma_set", "sync", "wait")):
            return "lpbq_sync_checkpoint"
        return "lpbq_other_fused"
    if "convert_weights_to_signed" in low or ("signed" in low and "shuff" in low):
        return "matmul_signed_conversion"
    if qnn_type in {"Convert", "CastType"}:
        return "explicit_convert_requant"
    if "linearclip" in low:
        return "fused_linearclip_requant"
    if any(token in low or token in qnn_low for token in ("quant", "dequant", "requant")):
        return "other_quant_kernel"
    return None


def load_runtime_events(trace_path, qhas_path):
    _, _, qhas_rows = load_qhas(qhas_path)
    num_q_heads, num_kv_heads = infer_head_counts(qhas_rows)
    with trace_path.open() as stream:
        document = json.load(stream)
    events = document if isinstance(document, list) else document.get("traceEvents", [])
    process_names = {
        event.get("pid"): event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    physical_dtypes_by_id = defaultdict(set)
    for event in events:
        if event.get("ph") != "X" or not process_names.get(event.get("pid"), "").startswith("QNN::"):
            continue
        event_args = event.get("args", {})
        event_id = event_args.get("ID")
        dtype = event_args.get("Data Type")
        if event_id and dtype:
            physical_dtypes_by_id[event_id].add(dtype)
    records = []
    seen = set()
    for event in events:
        if event.get("ph") != "X" or not process_names.get(event.get("pid"), "").startswith("QNN::"):
            continue
        args = event.get("args", {})
        qnn_name = args.get("QNN Op Name")
        qnn_type = args.get("QNN Op Type", "")
        htp_type = args.get("HTP Op Type", event.get("name", ""))
        flags = tuple(args.get("Flags", []))
        output_dtype = physical_output_dtype(args)
        input_dtypes = set()
        for input_id in args.get("Inputs Ops", []):
            input_dtypes.update(physical_dtypes_by_id.get(input_id, set()))
        if input_dtypes:
            input_dtype_signature = " + ".join(sorted(input_dtypes))
        elif output_dtype == "CONTROL(no tensor)":
            input_dtype_signature = "CONTROL(dependency only)"
        elif "dma" in flags and "to_vtcm" in htp_type.lower():
            input_dtype_signature = f"{output_dtype} (inferred: dtype-preserving DMA)"
        elif args.get("Inputs Ops"):
            input_dtype_signature = "UNRESOLVED(input ID not typed by reader)"
        else:
            input_dtype_signature = "NONE/IMPLICIT"
        category = classify_quant_kernel(htp_type, qnn_type, flags)
        trace_duration = int(event.get("dur", 0) or 0)
        trace_start = int(event.get("ts", 0) or 0)
        hardware_active_cycles = int(args.get("Duration (cycles)", 0) or 0)
        if not qnn_name or not category or trace_duration <= 0:
            continue
        identity = (args.get("ID"), qnn_name, event.get("tid"), trace_start, trace_duration)
        if identity in seen:
            continue
        seen.add(identity)
        stage, layer, head, detail = classification(qnn_name, qnn_type, num_q_heads, num_kv_heads)
        records.append({
            "category": category,
            "qnn_name": qnn_name,
            "qnn_type": qnn_type,
            "htp_type": htp_type,
            "stage": stage,
            "stage_label": STAGE_INFO[stage]["label"],
            "layer": layer,
            "head": head,
            "semantic_detail": detail,
            "data_type": output_dtype,
            "input_data_types": input_dtype_signature,
            "step_size": args.get("Step Size"),
            "zero_offset": args.get("Zero Offset"),
            "flags": "+".join(flags) or "NONE",
            "start_cycle": trace_start,
            "duration_cycles": trace_duration,
            "hardware_active_cycles": hardware_active_cycles,
            "dominant_path_cycles": int(args.get("Dominant Path Cycles", 0) or 0),
        })
    del document, events
    gc.collect()
    return records


def aggregate_runtime(records, keys):
    groups = {}
    for record in records:
        key = tuple(record[name] for name in keys)
        group = groups.setdefault(key, {
            "event_count": 0, "qnn_ops": set(), "work_cycles": 0, "hardware_active_cycles": 0,
            "dominant_path_cycles": 0,
            "intervals": [], "physical_input_dtypes": Counter(), "physical_dtypes": Counter(),
            "htp_types": Counter(), "flags": set(), "qnn_input_types": set(), "qnn_output_types": set(),
            "step_sizes": set(), "zero_offsets": set(),
        })
        group["event_count"] += 1
        group["qnn_ops"].add(record["qnn_name"])
        group["work_cycles"] += record["duration_cycles"]
        group["hardware_active_cycles"] += record["hardware_active_cycles"]
        group["dominant_path_cycles"] += record["dominant_path_cycles"]
        group["intervals"].append((record["start_cycle"], record["start_cycle"] + record["duration_cycles"]))
        group["physical_input_dtypes"][record["input_data_types"]] += record["duration_cycles"]
        group["physical_dtypes"][record["data_type"]] += record["duration_cycles"]
        group["htp_types"][record["htp_type"]] += record["duration_cycles"]
        group["flags"].add(record["flags"])
        if record.get("qnn_input_types"):
            group["qnn_input_types"].add(record["qnn_input_types"])
        if record.get("qnn_output_types"):
            group["qnn_output_types"].add(record["qnn_output_types"])
        if record["step_size"] is not None:
            group["step_sizes"].add(record["step_size"])
        if record["zero_offset"] is not None:
            group["zero_offsets"].add(record["zero_offset"])
    rows = []
    for key, group in groups.items():
        _, _, wall, active, _, parallel = interval_stats(group.pop("intervals"))
        row = dict(zip(keys, key))
        row.update(group)
        row.update({
            "qnn_op_instances": len(group["qnn_ops"]),
            "active_union_cycles": active,
            "wall_span_cycles": wall,
            "max_parallelism": parallel,
            "physical_input_dtypes_text": "; ".join(
                f"{k}: {v:,} cycles" for k, v in group["physical_input_dtypes"].most_common()
            ),
            "physical_dtypes_text": "; ".join(
                f"{k}: {v:,} cycles" for k, v in group["physical_dtypes"].most_common()
            ),
            "qnn_input_types_text": " | ".join(sorted(group["qnn_input_types"])) or "not in manifest",
            "qnn_output_types_text": " | ".join(sorted(group["qnn_output_types"])) or "not in manifest",
            "top_htp_types": "; ".join(f"{k}:{v}" for k, v in group["htp_types"].most_common(4)),
            "flags_text": "; ".join(sorted(group["flags"])),
            "step_sizes_text": "; ".join(str(value) for value in sorted(group["step_sizes"])),
            "zero_offsets_text": "; ".join(str(value) for value in sorted(group["zero_offsets"])),
        })
        rows.append(row)
    return rows


def load_manifests(paths):
    tensors = {}
    operations = {}
    graphs = []
    for path in paths:
        with path.open() as stream:
            manifest = json.load(stream)
        graphs.append(manifest.get("graph", path.stem))
        for tensor in manifest.get("tensors", []):
            tensors[f"{manifest.get('graph')}::{tensor['name']}"] = tensor | {"graph": manifest.get("graph")}
        for operation in manifest.get("operations", []):
            operations[f"{manifest.get('graph')}::{operation['name']}"] = operation | {"graph": manifest.get("graph")}
    return graphs, tensors, operations


def manifest_tensor_type(tensor):
    logical = tensor.get("logical_quant_dtype", tensor.get("ir_storage_dtype", "unknown"))
    qnn_dtype = tensor.get("qnn_dtype", "unknown")
    encoding = tensor.get("qnn_quantization", {}).get("encoding", "undefined")
    recipe = tensor.get("quant_recipe", {}).get("type", "unknown")
    return f"{logical} [QNN {qnn_dtype}; {recipe}/{encoding}]"


def attach_manifest_io(records, tensors, operations):
    tensors_by_graph_and_name = {
        (tensor.get("graph"), tensor.get("name")): tensor for tensor in tensors.values()
    }
    operations_by_name = defaultdict(list)
    for operation in operations.values():
        operations_by_name[operation.get("name")].append(operation)
    for record in records:
        candidates = operations_by_name.get(record["qnn_name"], [])
        if not candidates:
            record["qnn_input_types"] = ""
            record["qnn_output_types"] = ""
            continue
        operation = candidates[0]
        graph = operation.get("graph")
        input_labels = []
        output_labels = []
        for index, name in enumerate(operation.get("inputs", [])):
            tensor = tensors_by_graph_and_name.get((graph, name), {})
            role = "weight" if tensor.get("tensor_type") == "STATIC" else f"in{index}"
            input_labels.append(f"{role}={manifest_tensor_type(tensor)}")
        for index, name in enumerate(operation.get("outputs", [])):
            tensor = tensors_by_graph_and_name.get((graph, name), {})
            output_labels.append(f"out{index}={manifest_tensor_type(tensor)}")
        record["qnn_input_types"] = ", ".join(input_labels) or "NONE"
        record["qnn_output_types"] = ", ".join(output_labels) or "NONE"


def manifest_summary(tensors):
    recipe = Counter()
    dtype = Counter()
    encoding = Counter()
    for tensor in tensors.values():
        recipe[tensor.get("quant_recipe", {}).get("type", "unknown")] += 1
        dtype[tensor.get("qnn_dtype", "unknown")] += 1
        encoding[tensor.get("qnn_quantization", {}).get("encoding", "unknown")] += 1
    return recipe, dtype, encoding


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cooked = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (set, dict, Counter, list, tuple)):
                    value = json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
                cooked[field] = value
            writer.writerow(cooked)


def pct(value, total):
    return 100.0 * value / total if total else 0.0


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_html(path, model_id, runtime_rows, stage_rows, graphs, tensors, graph_work_cycles, graph_wall_cycles):
    runtime_rows = sorted(runtime_rows, key=lambda row: CATEGORY_INFO[row["category"]][0])
    total_work = sum(row["work_cycles"] for row in runtime_rows)
    total_dominant = sum(row["dominant_path_cycles"] for row in runtime_rows)
    expansion = next((row for row in runtime_rows if row["category"] == "lpbq_weight_expand_dequant"), None)
    hmx = next((row for row in runtime_rows if row["category"] == "lpbq_hmx_mac"), None)
    conversion_categories = {
        "lpbq_weight_expand_dequant", "explicit_convert_requant", "fused_linearclip_requant",
        "matmul_signed_conversion",
    }
    conversion_work = sum(row["work_cycles"] for row in runtime_rows if row["category"] in conversion_categories)
    conversion_dominant = sum(
        row["dominant_path_cycles"] for row in runtime_rows if row["category"] in conversion_categories
    )
    recipe, dtype, encoding = manifest_summary(tensors)

    runtime_table = table(
        ["Runtime phase", "HTP resource", "Events", "QNN nodes", "Timeline work", "% quant work", "HW active cycles",
         "Direct dominant", "% quant dominant", "Active interval union", "Max overlap", "HTP physical input dtype",
         "HTP physical output dtype"],
        [[CATEGORY_INFO[row["category"]][1], row["flags_text"], f"{row['event_count']:,}",
          f"{row['qnn_op_instances']:,}", f"{row['work_cycles']:,}", f"{pct(row['work_cycles'], total_work):.1f}%",
          f"{row['hardware_active_cycles']:,}",
          f"{row['dominant_path_cycles']:,}", f"{pct(row['dominant_path_cycles'], total_dominant):.1f}%",
          f"{row['active_union_cycles']:,}", f"{row['max_parallelism']:.1f}x",
          row["physical_input_dtypes_text"], row["physical_dtypes_text"]]
         for row in runtime_rows],
    )
    stage_table = table(
        ["Qwen3 stage", "Runtime phase", "QNN logical/pre-finalize inputs", "QNN logical/pre-finalize outputs",
         "HTP physical inputs", "HTP physical outputs", "Timeline work", "HW active cycles", "Direct dominant",
         "Active interval union", "QNN nodes"],
        [[row["stage_label"], CATEGORY_INFO[row["category"]][1],
          row["qnn_input_types_text"], row["qnn_output_types_text"], row["physical_input_dtypes_text"],
          row["physical_dtypes_text"], f"{row['work_cycles']:,}", f"{row['hardware_active_cycles']:,}",
          f"{row['dominant_path_cycles']:,}",
          f"{row['active_union_cycles']:,}", row["qnn_op_instances"]]
         for row in sorted(stage_rows, key=lambda row: (STAGE_INFO[row["stage"]]["order"], CATEGORY_INFO[row["category"]][0]))],
    )
    manifest_tables = "<p>No compile-time manifest supplied. Runtime HTP evidence remains available.</p>"
    if tensors:
        manifest_tables = (
            f"<p>Graphs: {html.escape(', '.join(graphs))}; {len(tensors):,} pre-finalize tensors.</p>"
            + table(["Logical quant recipe", "Tensor count"], recipe.most_common())
            + table(["Pre-finalize QNN dtype", "Tensor count"], dtype.most_common())
            + table(["QNN encoding", "Tensor count"], encoding.most_common())
        )
    finding = ""
    if expansion and hmx:
        finding = (
            f"LPBQ weight expansion/dequant uses <b>{expansion['work_cycles']:,}</b> work cycles, "
            f"{expansion['work_cycles'] / max(hmx['work_cycles'], 1):.1f}× the HMX MAC work "
            f"({hmx['work_cycles']:,}), and {pct(expansion['work_cycles'], graph_work_cycles):.1f}% of whole-graph "
            f"summed work. All visible conversion/requant phases together account for {conversion_work:,} work cycles "
            f"({pct(conversion_work, graph_work_cycles):.1f}% of graph work) and {conversion_dominant:,} directly "
            f"attributed dominant-path cycles ({pct(conversion_dominant, graph_wall_cycles):.1f}% of the HTP timeline). "
            "The work ratio is not serialized latency."
        )
    path.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen3 quantization profiling</title>
<style>body{{font-family:system-ui,sans-serif;margin:28px;color:#18212b}}h1,h2{{color:#17365d}}.table-wrap{{overflow-x:auto;margin:12px 0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd5df;padding:6px 8px;text-align:right;vertical-align:top;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf0f6;position:sticky;top:0}}.note{{background:#fff4d6;border-left:5px solid #e0a500;padding:12px}}.finding{{background:#e8f5e9;border-left:5px solid #388e3c;padding:12px}}code{{background:#eef2f5;padding:1px 4px}}</style></head>
<body><h1>Qwen3 Quantization View — {html.escape(model_id)}</h1>
<p>This view joins logical quantization intent, QNN tensor encodings, and post-lowering HTP execution.</p>
<div class="finding">{finding}</div>
<div class="note"><b>Type notation:</b> QNN columns show logical quant dtype plus the pre-finalize QNN carrier/encoding. HTP columns show post-lowering physical event I/O. Values after a physical dtype are timeline work cycles, not bytes or tensor counts. <code>CONTROL(no tensor)</code> is an intentional rank-0 synchronization/DMA-wait event, not a missing dtype.</div>
<div class="note"><b>Latency accounting:</b> timeline work is the sum of Chrome Trace event durations; HW active cycles are the HTP event payload's active-cycle count. Both are additive engine work and may overlap. Do not add HVX expansion, DMA, and HMX MAC as wall latency. Use direct dominant-path cycles for directly attributed critical-path contribution, and active interval union for overlap-aware activity windows. Even unions from different rows can overlap with each other.</div>
<h2>Runtime quantized-kernel phases</h2>{runtime_table}
<h2>Qwen3 structure × quantization phase</h2>{stage_table}
<h2>Compile-time dtype / encoding manifest</h2>{manifest_tables}
<h2>Interpretation boundaries</h2>
<ul><li><b>Logical W4A8O8 G32</b> is a model/recipe property. The HTP trace may show <code>QInt8</code> because V79 expands block-scaled W4 weights into an internal per-channel int8 form before HMX.</li><li>A missing standalone Quantize/Dequantize op does not imply zero conversion cost: QNN can fuse Q/DQ into a consumer. The report measures visible lowered kernels and labels fused boundaries separately in the manifest.</li><li><code>Step Size</code> and <code>Zero Offset</code> in Optrace describe the physical HTP tensor. The manifest's scale/zero-point describe the pre-finalize QNN tensor; both are retained because lowering may transform them.</li></ul>
</body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="QNN Chrome Trace JSON")
    parser.add_argument("--qhas-json", required=True, type=Path)
    parser.add_argument("--quant-manifest", action="append", default=[], type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    graphs, tensors, operations = load_manifests(args.quant_manifest)
    records = load_runtime_events(args.trace, args.qhas_json)
    attach_manifest_io(records, tensors, operations)
    runtime_rows = aggregate_runtime(records, ["category"])
    stage_rows = aggregate_runtime(records, ["stage", "stage_label", "category"])
    operator_rows = aggregate_runtime(
        records, ["stage", "stage_label", "category", "qnn_name", "qnn_type", "input_data_types", "data_type"]
    )
    with args.qhas_json.open() as stream:
        qhas = json.load(stream)
    model_id = qhas.get("model_id", args.trace.stem)
    graph_work_cycles = sum(row["cycles"] for row in qhas["data"]["qnn_op_instances_nodes"]["data"])
    graph_wall_cycles = qhas["data"]["htp_overall_summary"]["data"][0]["timeline_cycles"]

    runtime_fields = ["category", "event_count", "qnn_op_instances", "work_cycles", "hardware_active_cycles",
                      "dominant_path_cycles",
                      "active_union_cycles", "wall_span_cycles", "max_parallelism", "qnn_input_types_text",
                      "qnn_output_types_text", "physical_input_dtypes_text", "physical_dtypes_text",
                      "step_sizes_text", "zero_offsets_text", "flags_text", "top_htp_types"]
    stage_fields = ["stage", "stage_label", "category"] + runtime_fields[1:]
    write_csv(Path(f"{args.output_prefix}-quant-kernels.csv"), runtime_rows, runtime_fields)
    write_csv(Path(f"{args.output_prefix}-quant-stage.csv"), stage_rows, stage_fields)
    operator_fields = ["stage", "stage_label", "category", "qnn_name", "qnn_type", "input_data_types",
                       "data_type"] + runtime_fields[1:]
    write_csv(Path(f"{args.output_prefix}-quant-operators.csv"), operator_rows, operator_fields)

    edge_rows = []
    for tensor in tensors.values():
        quant = tensor.get("qnn_quantization", {})
        recipe = tensor.get("quant_recipe", {})
        for consumer in tensor.get("consumers", []):
            producer = tensor.get("producer")
            producer_op = operations.get(f"{tensor.get('graph')}::{producer}", {})
            consumer_op = operations.get(f"{tensor.get('graph')}::{consumer}", {})
            if producer_op.get("qnn_op_type") in {"Convert", "CastType"}:
                boundary_kind = "explicit_conversion_output"
            elif recipe.get("type") == "lpbq":
                boundary_kind = "lpbq_static_weight"
            elif quant.get("defined"):
                boundary_kind = "quantized_tensor_or_fused_boundary"
            else:
                boundary_kind = "unquantized_or_index_control"
            edge_rows.append({
                "graph": tensor.get("graph"), "tensor": tensor.get("name"), "producer": producer,
                "producer_qnn_op_type": producer_op.get("qnn_op_type"), "consumer": consumer,
                "consumer_qnn_op_type": consumer_op.get("qnn_op_type"), "boundary_kind": boundary_kind,
                "ir_storage_dtype": tensor.get("ir_storage_dtype", tensor.get("logical_dtype")),
                "logical_quant_dtype": tensor.get("logical_quant_dtype", recipe.get("quant_to_dtype", recipe.get("storage_dtype"))),
                "qnn_dtype": tensor.get("qnn_dtype"),
                "dimensions": tensor.get("dimensions"), "quant_recipe": recipe.get("type"),
                "quant_to_dtype": recipe.get("quant_to_dtype", recipe.get("storage_dtype", "")),
                "qnn_encoding": quant.get("encoding"), "scale": quant.get("scale"),
                "zero_point": quant.get("zero_point"), "axis": quant.get("axis"),
                "block_size": recipe.get("block_size"), "block_scale_bitwidth": quant.get("block_scale_bitwidth"),
                "num_blocks_per_axis": quant.get("num_blocks_per_axis"),
            })
    edge_fields = ["graph", "tensor", "producer", "producer_qnn_op_type", "consumer", "consumer_qnn_op_type",
                   "boundary_kind", "ir_storage_dtype", "logical_quant_dtype", "qnn_dtype", "dimensions",
                   "quant_recipe", "quant_to_dtype", "qnn_encoding", "scale", "zero_point", "axis", "block_size",
                   "block_scale_bitwidth", "num_blocks_per_axis"]
    write_csv(Path(f"{args.output_prefix}-dtype-edges.csv"), edge_rows, edge_fields)
    write_html(Path(f"{args.output_prefix}-quantization.html"), model_id, runtime_rows, stage_rows, graphs, tensors,
               graph_work_cycles, graph_wall_cycles)

    print(f"quant events: {len(records):,}; runtime categories: {len(runtime_rows)}; manifest tensors: {len(tensors):,}")


if __name__ == "__main__":
    main()
