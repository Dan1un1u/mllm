#!/usr/bin/env python3
"""Build a Qwen3-structure view from QNN HTP Optrace artifacts.

The official QHAS report groups work by QNN operator type.  This tool keeps
those results intact and adds a model-oriented view:

  layer RMSNorm -> Q/K/V -> Q/K Norm -> RoPE -> KV cache -> QK^T
  -> scale/mask/Softmax -> attention*V -> O projection -> MLP -> residual

It uses raw Chrome Trace intervals for overlap-aware interval unions and QHAS
for dominant-path cycles and DRAM/VTCM accounting.
"""

import argparse
import csv
import gc
import html
import json
import re
from collections import defaultdict
from pathlib import Path

from qnn_optrace_summary import interval_stats, lane_type, load_htp_flags


STAGES = [
    (0, "graph_input", "Graph input", "Data movement / runtime"),
    (10, "embedding", "Embedding and position gathers", "DMA / lookup"),
    (20, "input_rmsnorm", "Input RMSNorm", "HVX reduction + VTCM"),
    (30, "q_projection", "Q projection", "LPBQ weight stream + HMX"),
    (31, "k_projection", "K projection", "LPBQ weight stream + HMX"),
    (32, "v_projection", "V projection", "LPBQ weight stream + HMX"),
    (40, "q_head_rmsnorm", "Q head RMSNorm", "HVX reduction + VTCM"),
    (41, "k_head_rmsnorm", "K head RMSNorm", "HVX reduction + VTCM"),
    (50, "q_rope", "Q RoPE", "HVX elementwise + VTCM"),
    (51, "k_rope", "K RoPE", "HVX elementwise + VTCM"),
    (60, "kv_cache_update", "KV-cache convert/update", "DMA/HVX data movement"),
    (70, "qk_similarity", "QK^T similarity", "HMX/HVX + VTCM"),
    (80, "scale_mask_softmax", "Scale + mask + Softmax (fused attribution)", "HVX reduction + VTCM"),
    (90, "attention_value", "Attention x V", "HMX/HVX + VTCM"),
    (100, "head_merge", "Head merge/reshape", "HVX data movement"),
    (110, "o_projection", "O projection", "LPBQ weight stream + HMX"),
    (120, "attention_residual", "Attention residual", "HVX elementwise + VTCM"),
    (130, "post_attention_rmsnorm", "Post-attention RMSNorm", "HVX reduction + VTCM"),
    (140, "mlp_gate_projection", "MLP gate projection", "LPBQ weight stream + HMX"),
    (141, "mlp_up_projection", "MLP up projection", "LPBQ weight stream + HMX"),
    (150, "mlp_silu", "MLP SiLU", "HVX elementwise + VTCM"),
    (151, "mlp_gate_product", "MLP gate x up", "HVX elementwise + VTCM"),
    (160, "mlp_down_projection", "MLP down projection", "LPBQ weight stream + HMX"),
    (170, "mlp_residual", "MLP residual", "HVX elementwise + VTCM"),
    (180, "final_rmsnorm", "Final RMSNorm", "HVX reduction + VTCM"),
    (190, "lm_head", "LM head", "LPBQ weight stream + HMX"),
    (200, "graph_output_runtime", "Graph output/runtime", "Data movement / runtime"),
    (999, "unclassified", "Unclassified", "Needs mapping review"),
]

STAGE_INFO = {
    key: {"order": order, "label": label, "bottleneck": bottleneck}
    for order, key, label, bottleneck in STAGES
}
WEIGHT_STREAM_STAGES = {
    "q_projection", "k_projection", "v_projection", "o_projection",
    "mlp_gate_projection", "mlp_up_projection", "mlp_down_projection", "lm_head",
}
LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.*)")
NUMBERED_PATTERN = re.compile(r"([A-Za-z]+)\.(\d+)$")


def numbered_suffix(value):
    match = NUMBERED_PATTERN.search(value)
    return (match.group(1), int(match.group(2))) if match else (None, None)


def infer_head_counts(qhas_rows):
    q_heads = set()
    kv_heads = set()
    for row in qhas_rows:
        name = row["qnn_op"]
        q_match = re.search(r"\.self_attn\.q_proj\.(\d+)$", name)
        kv_match = re.search(r"\.self_attn\.[kv]_proj\.(\d+)$", name)
        if q_match:
            q_heads.add(int(q_match.group(1)))
        if kv_match:
            kv_heads.add(int(kv_match.group(1)))
    if not q_heads or not kv_heads:
        raise ValueError("Cannot infer Qwen3 Q/KV head counts from per-head projection names")
    return max(q_heads) + 1, max(kv_heads) + 1


def classification(qnn_name, qnn_type, num_q_heads, num_kv_heads):
    """Return stage, layer, head and a finer semantic detail."""
    if qnn_name in {"lm_head", "model.lm_head"} or qnn_name.startswith("lm_head."):
        return "lm_head", None, None, "vocabulary_projection"
    if qnn_name == "model.norm":
        return "final_rmsnorm", None, None, "final_norm"
    if qnn_type == "Gather" or qnn_name.startswith("model.Gather") or "embed_tokens" in qnn_name:
        return "embedding", None, None, "embedding_or_position_lookup"
    if qnn_name == "Input":
        return "graph_input", None, None, "graph_input"
    if qnn_name == "Output" or qnn_name.startswith("SystemService") or qnn_name.startswith("$Const"):
        return "graph_output_runtime", None, None, "graph_output_or_runtime"

    match = LAYER_PATTERN.fullmatch(qnn_name)
    if not match:
        return "unclassified", None, None, "unknown"
    layer = int(match.group(1))
    rest = match.group(2)

    direct = {
        "input_layernorm": ("input_rmsnorm", "input_norm"),
        "post_attention_layernorm": ("post_attention_rmsnorm", "post_attention_norm"),
        "self_attn.o_proj": ("o_projection", "attention_output_projection"),
        "Add.0": ("attention_residual", "attention_residual_add"),
        "Add.1": ("mlp_residual", "mlp_residual_add"),
        "mlp.gate_proj": ("mlp_gate_projection", "mlp_gate_projection"),
        "mlp.up_proj": ("mlp_up_projection", "mlp_up_projection"),
        "mlp.down_proj": ("mlp_down_projection", "mlp_down_projection"),
        "mlp.Mul.0": ("mlp_silu", "silu_gate_multiply"),
        "mlp.Mul.1": ("mlp_gate_product", "gate_times_up"),
    }
    if rest in direct:
        stage, detail = direct[rest]
        return stage, layer, None, detail

    for prefix, stage in (
        ("self_attn.q_proj.", "q_projection"),
        ("self_attn.k_proj.", "k_projection"),
        ("self_attn.v_proj.", "v_projection"),
        ("self_attn.q_norm.", "q_head_rmsnorm"),
        ("self_attn.k_norm.", "k_head_rmsnorm"),
        ("self_attn.Softmax.", "scale_mask_softmax"),
    ):
        if rest.startswith(prefix):
            head = int(rest.removeprefix(prefix))
            return stage, layer, head, stage

    if rest.startswith("self_attn.MatMul."):
        index = int(rest.removeprefix("self_attn.MatMul."))
        if index % 2 == 0:
            return "qk_similarity", layer, index // 2, "q_times_k_transpose"
        return "attention_value", layer, index // 2, "attention_times_v"

    if rest.startswith("self_attn.View."):
        return "head_merge", layer, None, "head_merge_reshape"

    # Only self-attention operators can be assigned to the KV-cache stage.
    # MLP CastType/Transpose/Slice operators have the same QNN type names but
    # are unrelated data conversions; accepting them here silently inflated
    # the reported KV-cache cost on host-generated reports.
    if rest.startswith("self_attn."):
        self_attention_name = rest.removeprefix("self_attn.")
        kind, index = numbered_suffix(self_attention_name)
        if kind in {"Mul", "Add", "Neg"}:
            q_limit = 2 * num_q_heads if kind == "Mul" else num_q_heads
            divisor = 2 if kind == "Mul" else 1
            if index < q_limit:
                return "q_rope", layer, index // divisor, f"q_rope_{kind.lower()}"
            return "k_rope", layer, (index - q_limit) // divisor, f"k_rope_{kind.lower()}"

        if kind == "Concat":
            # Graph construction creates Q rotate-half concats, then K rotate-half
            # concats, two cache concats per KV head, attention head merge, and final
            # K/V output concats. QNN may fuse away a subset.
            q_rope_end = num_q_heads - 1
            k_rope_end = q_rope_end + num_kv_heads
            cache_end = k_rope_end + 2 * num_kv_heads
            if index <= q_rope_end:
                return "q_rope", layer, index, "q_rotate_half_concat"
            if index <= k_rope_end:
                return "k_rope", layer, index - num_q_heads, "k_rotate_half_concat"
            if index <= cache_end:
                return "kv_cache_update", layer, (index - k_rope_end - 1) // 2, "kv_cache_concat"
            if index == cache_end + 1:
                return "head_merge", layer, None, "attention_head_concat"
            return "kv_cache_update", layer, None, "kv_output_concat"

        if kind in {"Slice", "CastType", "Transpose"}:
            rope_slice_count = 2 * (num_q_heads + num_kv_heads)
            if kind == "Slice" and index >= rope_slice_count:
                head = (index - rope_slice_count) // 2
            elif kind == "CastType":
                head = index // 2
            else:
                head = index
            return "kv_cache_update", layer, head, f"kv_cache_{kind.lower()}"

    return "unclassified", layer, None, rest


def load_trace_intervals(path):
    with path.open() as stream:
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
    del document, events
    gc.collect()
    return intervals, lanes


def load_qhas(path):
    with path.open() as stream:
        document = json.load(stream)
    data = document["data"]
    overall = data["htp_overall_summary"]["data"][0]
    rows = data["qnn_op_instances_nodes"]["data"]
    model_id = document.get("model_id", "unknown")
    return model_id, overall, rows


SUM_FIELDS = (
    "cycles", "num_dominant_path_cycles_htp_0", "num_htp_ops",
    "dram_read", "dram_write", "vtcm_read", "vtcm_write",
)


def build_operator_records(qhas_rows, trace_intervals, lanes, topology_flags, num_q_heads, num_kv_heads):
    records = []
    for source in qhas_rows:
        name = source["qnn_op"]
        stage, layer, head, detail = classification(
            name, source["qnn_op_type"], num_q_heads, num_kv_heads
        )
        first, last, wall, active, trace_work, parallel = interval_stats(trace_intervals.get(name, []))
        record = dict(source)
        record.update({
            "stage": stage,
            "stage_order": STAGE_INFO[stage]["order"],
            "stage_label": STAGE_INFO[stage]["label"],
            "layer": layer,
            "head": head,
            "semantic_detail": detail,
            "kernel_resources": "+".join(sorted(topology_flags.get(name, []))) or "UNKNOWN",
            "trace_lanes": "+".join(sorted(lanes.get(name, []))) or "UNKNOWN",
            "start_cycle": first,
            "end_cycle": last,
            "wall_span_cycles": wall,
            "active_union_cycles": active,
            "trace_work_cycles": trace_work,
            "max_parallelism": parallel,
        })
        records.append(record)
    return records


def aggregate(records, key_function, overall):
    groups = {}
    total_work = sum(record["cycles"] for record in records)
    total_critical = sum(record["num_dominant_path_cycles_htp_0"] for record in records)
    total_dram = overall["total_dram_read"] + overall["total_dram_write"]
    for record in records:
        key = key_function(record)
        if key is None:
            continue
        group = groups.setdefault(key, {
            "qnn_op_instances": 0,
            "intervals": [],
            "kernel_resources": set(),
            "trace_lanes": set(),
        })
        group["qnn_op_instances"] += 1
        group["intervals"].extend(record["_intervals"])
        group["kernel_resources"].update(record["kernel_resources"].split("+"))
        group["trace_lanes"].update(record["trace_lanes"].split("+"))
        for field in SUM_FIELDS:
            group[field] = group.get(field, 0) + record[field]

    rows = []
    for key, group in groups.items():
        first, last, wall, active, trace_work, parallel = interval_stats(group.pop("intervals"))
        row = dict(group)
        row.update({
            "group_key": key,
            "start_cycle": first,
            "end_cycle": last,
            "wall_span_cycles": wall,
            "active_union_cycles": active,
            "trace_work_cycles": trace_work,
            "overlapped_trace_work_cycles": max(0, trace_work - active),
            "max_parallelism": parallel,
            "work_percent": 100.0 * group["cycles"] / total_work if total_work else 0.0,
            "critical_path_percent": (
                100.0 * group["num_dominant_path_cycles_htp_0"] / total_critical if total_critical else 0.0
            ),
            "critical_path_us_estimate": (
                overall["graph_execute_us"] * group["num_dominant_path_cycles_htp_0"] / total_critical
                if total_critical else 0.0
            ),
            "dram_percent": (
                100.0 * (group["dram_read"] + group["dram_write"]) / total_dram if total_dram else 0.0
            ),
            "dram_bytes_per_critical_cycle": (
                (group["dram_read"] + group["dram_write"]) / group["num_dominant_path_cycles_htp_0"]
                if group["num_dominant_path_cycles_htp_0"] else 0.0
            ),
            "vtcm_bytes_per_work_cycle": (
                (group["vtcm_read"] + group["vtcm_write"]) / group["cycles"] if group["cycles"] else 0.0
            ),
            "kernel_resources": "+".join(sorted(group["kernel_resources"])),
            "trace_lanes": "+".join(sorted(group["trace_lanes"])),
        })
        rows.append(row)
    return rows


COMMON_COLUMNS = [
    "qnn_op_instances", "kernel_resources", "trace_lanes", "start_cycle", "end_cycle",
    "wall_span_cycles", "active_union_cycles", "trace_work_cycles", "overlapped_trace_work_cycles",
    "max_parallelism", "cycles", "work_percent", "num_dominant_path_cycles_htp_0",
    "critical_path_percent", "critical_path_us_estimate", "num_htp_ops", "dram_read", "dram_write",
    "dram_percent", "dram_bytes_per_critical_cycle", "vtcm_read", "vtcm_write",
    "vtcm_bytes_per_work_cycle",
]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def escaped(value):
    return html.escape(str(value))


def fmt(value, digits=2):
    return f"{value:,.{digits}f}"


def render_html(path, model_id, overall, stage_rows, layer_rows, head_rows, unclassified):
    stage_rows = sorted(stage_rows, key=lambda row: row["stage_order"])
    layer_stages = [key for _, key, _, _ in STAGES if any(row["stage"] == key for row in layer_rows)]
    layers = sorted({row["layer"] for row in layer_rows})
    layer_lookup = {(row["layer"], row["stage"]): row for row in layer_rows}
    stage_max = {
        stage: max((layer_lookup.get((layer, stage), {}).get("critical_path_us_estimate", 0) for layer in layers), default=0)
        for stage in layer_stages
    }
    weight_critical = sum(row["critical_path_percent"] for row in stage_rows if row["stage"] in WEIGHT_STREAM_STAGES)
    weight_dram = sum(row["dram_read"] + row["dram_write"] for row in stage_rows if row["stage"] in WEIGHT_STREAM_STAGES)
    total_dram = overall["total_dram_read"] + overall["total_dram_write"]
    hottest = max(stage_rows, key=lambda row: row["critical_path_percent"])

    stage_table = []
    for row in stage_rows:
        stage_table.append(
            "<tr>"
            f"<td>{row['stage_order']}</td><td><code>{escaped(row['stage'])}</code><br><small>{escaped(row['stage_label'])}</small></td>"
            f"<td>{fmt(row['critical_path_percent'])}%</td><td>{fmt(row['critical_path_us_estimate'], 1)}</td>"
            f"<td>{fmt(row['work_percent'])}%</td><td>{fmt((row['dram_read'] + row['dram_write']) / 1e6, 2)}</td>"
            f"<td>{fmt(row['dram_percent'])}%</td><td>{fmt((row['vtcm_read'] + row['vtcm_write']) / 1e9, 3)}</td>"
            f"<td>{escaped(row['kernel_resources'])}</td><td>{escaped(row['bottleneck_class'])}</td>"
            "</tr>"
        )

    heat_header = "".join(f"<th title='{escaped(STAGE_INFO[s]['label'])}'>{escaped(s)}</th>" for s in layer_stages)
    heat_rows = []
    for layer in layers:
        cells = []
        for stage in layer_stages:
            value = layer_lookup.get((layer, stage), {}).get("critical_path_us_estimate", 0)
            maximum = stage_max[stage]
            alpha = 0.08 + 0.82 * value / maximum if maximum else 0.0
            cells.append(
                f"<td style='background:rgba(239,108,0,{alpha:.3f})' title='{escaped(STAGE_INFO[stage]['label'])}: {value:.2f} us'>{value:.1f}</td>"
            )
        total = sum(layer_lookup.get((layer, stage), {}).get("critical_path_us_estimate", 0) for stage in layer_stages)
        heat_rows.append(f"<tr><th>Layer {layer}</th><td><b>{total:.1f}</b></td>{''.join(cells)}</tr>")

    hottest_heads = sorted(head_rows, key=lambda row: row["critical_path_us_estimate"], reverse=True)[:80]
    head_table = "".join(
        "<tr>"
        f"<td>{row['layer']}</td><td>{row['head']}</td><td><code>{escaped(row['stage'])}</code></td>"
        f"<td>{fmt(row['critical_path_us_estimate'], 2)}</td><td>{fmt(row['work_percent'], 3)}%</td>"
        f"<td>{fmt((row['dram_read'] + row['dram_write']) / 1e6, 3)}</td><td>{escaped(row['kernel_resources'])}</td>"
        "</tr>"
        for row in hottest_heads
    )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Qwen3 structure profiling - {escaped(model_id)}</title>
<style>
html{{color-scheme:light;background:#fff}} body{{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#202124;background:#fff}} h1,h2{{margin-top:28px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px}} .card{{border:1px solid #ddd;border-radius:8px;padding:12px 16px;min-width:190px}}
.big{{font-size:24px;font-weight:700}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:6px 8px;text-align:right;white-space:nowrap}}
th{{background:#f5f5f5}} td:nth-child(2),th:nth-child(2){{text-align:left}} .scroll{{overflow:auto}} small{{color:#666}}
.note{{background:#fff8e1;border-left:4px solid #ffb300;padding:10px 14px}} code{{font-size:12px}}
</style></head><body>
<h1>Qwen3 模型结构视角 Profiling</h1>
<p>Graph: <code>{escaped(model_id)}</code>。表格按模型 DAG 的逻辑顺序排列；同层不同 head 会被 HTP 并行、交错调度。关键路径时间是按 QHAS dominant-path cycles 对 <code>graphExecute</code> 的归因，不是把 operator cycles 简单相加。</p>
<div class="cards">
<div class="card"><small>graphExecute</small><div class="big">{fmt(overall['graph_execute_us'] / 1000, 3)} ms</div></div>
<div class="card"><small>最大关键路径阶段</small><div class="big">{fmt(hottest['critical_path_percent'], 2)}%</div><code>{escaped(hottest['stage'])}</code></div>
<div class="card"><small>全部 LPBQ projection</small><div class="big">{fmt(weight_critical, 2)}%</div>关键路径</div>
<div class="card"><small>LPBQ projection DRAM</small><div class="big">{fmt(weight_dram / 1e6, 1)} MB</div>{fmt(100 * weight_dram / total_dram if total_dram else 0)}% total</div>
<div class="card"><small>HTP resources</small><div class="big">1 HMX + 6 HVX</div>Optrace observed</div>
</div>
<h2>顺序模型阶段</h2>
<div class="scroll"><table><thead><tr><th>Order</th><th>Stage</th><th>Critical</th><th>Attributed us</th><th>Work</th><th>DRAM MB</th><th>DRAM</th><th>VTCM GB</th><th>Resources</th><th>Bound hypothesis</th></tr></thead>
<tbody>{''.join(stage_table)}</tbody></table></div>
<p class="note"><b>融合边界：</b>attention scale、causal mask、QDQ 和部分 reshape 已被 QNN/HTP 融合，不能从 QNN node 级结果独立计时。因此报告把它们归入 <code>scale_mask_softmax</code>；如需继续拆分，必须使用 HTP kernel dependency/schematic，而不能虚构独立 latency。</p>
<h2>Layer × Stage 关键路径热力图（us）</h2>
<p>颜色在每一列内归一化，用于发现同形状 layer 的异常，而不是跨列比较绝对颜色。</p>
<div class="scroll"><table><thead><tr><th>Layer</th><th>Total us</th>{heat_header}</tr></thead><tbody>{''.join(heat_rows)}</tbody></table></div>
<h2>逐 Head 热点（前 80 项）</h2>
<p>完整结果见 <code>*-qwen3-head-stage.csv</code>。</p>
<table><thead><tr><th>Layer</th><th>Head</th><th>Stage</th><th>Attributed us</th><th>Global work</th><th>DRAM MB</th><th>Resources</th></tr></thead><tbody>{head_table}</tbody></table>
<h2>Memory-bound 判读边界</h2>
<ul>
<li><b>LPBQ projections：</b>大量 DRAM 权重流量且使用 DMA+HMX+HVX，属于 weight-bandwidth-sensitive 与 HMX/解码混合瓶颈；仅凭 Optrace 不能断言是纯 DRAM-bound。</li>
<li><b>KV-cache：</b>低算术强度、DMA/HVX 和 DRAM 流量明显，是最明确的数据搬运候选。</li>
<li><b>RMSNorm/RoPE/Softmax：</b>DRAM 很少，主要是 HVX reduction/elementwise 与 VTCM 流量，不属于 off-chip DRAM-bound。</li>
<li><b>QK^T/Attention x V：</b>主要使用 HMX/HVX 与 VTCM。DRAM cache/preload 流量可能归属到相邻 cache/runtime 节点。</li>
</ul>
<p>未分类 QNN 节点：<b>{len(unclassified)}</b>。HTP Optrace 只覆盖 HTP graph；graph 外 CPU 工作需要 Perfetto/FastRPC 视角补充。</p>
</body></html>"""
    path.write_text(document)


def main():
    parser = argparse.ArgumentParser(description="Generate Qwen3 stage/layer/head reports from QNN HTP Optrace.")
    parser.add_argument("chrometrace", type=Path, help="qnn-profile-viewer Chrome Trace JSON")
    parser.add_argument("--htp-json", type=Path, required=True, help="qnn-profile-viewer *_htp.json")
    parser.add_argument("--qhas-json", type=Path, required=True, help="official *_qnn_htp_analysis_summary.json")
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="Output path prefix, for example results/qnn_optrace")
    args = parser.parse_args()

    topology_flags = load_htp_flags(args.htp_json)
    trace_intervals, lanes = load_trace_intervals(args.chrometrace)
    model_id, overall, qhas_rows = load_qhas(args.qhas_json)
    num_q_heads, num_kv_heads = infer_head_counts(qhas_rows)
    records = build_operator_records(
        qhas_rows, trace_intervals, lanes, topology_flags, num_q_heads, num_kv_heads
    )
    for record in records:
        record["_intervals"] = trace_intervals.get(record["qnn_op"], [])

    stage_rows = aggregate(records, lambda record: record["stage"], overall)
    for row in stage_rows:
        row["stage"] = row.pop("group_key")
        stage_info = STAGE_INFO[row["stage"]]
        row["stage_order"] = stage_info["order"]
        row["stage_label"] = stage_info["label"]
        row["bottleneck_class"] = stage_info["bottleneck"]
    stage_rows.sort(key=lambda row: row["stage_order"])

    layer_rows = aggregate(
        records,
        lambda record: (record["layer"], record["stage"]) if record["layer"] is not None else None,
        overall,
    )
    for row in layer_rows:
        row["layer"], row["stage"] = row.pop("group_key")
        row["stage_order"] = STAGE_INFO[row["stage"]]["order"]
        row["stage_label"] = STAGE_INFO[row["stage"]]["label"]
        row["bottleneck_class"] = STAGE_INFO[row["stage"]]["bottleneck"]
    layer_rows.sort(key=lambda row: (row["layer"], row["stage_order"]))

    head_rows = aggregate(
        records,
        lambda record: (record["layer"], record["head"], record["stage"])
        if record["layer"] is not None and record["head"] is not None else None,
        overall,
    )
    for row in head_rows:
        row["layer"], row["head"], row["stage"] = row.pop("group_key")
        row["stage_order"] = STAGE_INFO[row["stage"]]["order"]
        row["stage_label"] = STAGE_INFO[row["stage"]]["label"]
    head_rows.sort(key=lambda row: (row["layer"], row["head"], row["stage_order"]))

    operator_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "_intervals"}
        row["bottleneck_class"] = STAGE_INFO[row["stage"]]["bottleneck"]
        operator_rows.append(row)
    operator_rows.sort(key=lambda row: (
        row["layer"] is None, row["layer"] if row["layer"] is not None else 999,
        row["stage_order"], row["head"] if row["head"] is not None else 999, row["qnn_op"],
    ))

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stage_path = Path(f"{prefix}-qwen3-stage-summary.csv")
    layer_path = Path(f"{prefix}-qwen3-layer-stage.csv")
    head_path = Path(f"{prefix}-qwen3-head-stage.csv")
    operator_path = Path(f"{prefix}-qwen3-operator-structure.csv")
    html_path = Path(f"{prefix}-qwen3-structure.html")

    stage_columns = ["stage_order", "stage", "stage_label", "bottleneck_class"] + COMMON_COLUMNS
    layer_columns = ["layer", "stage_order", "stage", "stage_label", "bottleneck_class"] + COMMON_COLUMNS
    head_columns = ["layer", "head", "stage_order", "stage", "stage_label"] + COMMON_COLUMNS
    operator_columns = [
        "layer", "head", "stage_order", "stage", "stage_label", "semantic_detail", "bottleneck_class",
        "qnn_op", "qnn_op_type", "kernel_resources", "trace_lanes", "start_cycle", "end_cycle",
        "wall_span_cycles", "active_union_cycles", "trace_work_cycles", "max_parallelism",
        "cycles", "num_dominant_path_cycles_htp_0", "num_htp_ops", "dram_read", "dram_write",
        "vtcm_read", "vtcm_write",
    ]
    write_csv(stage_path, stage_rows, stage_columns)
    write_csv(layer_path, layer_rows, layer_columns)
    write_csv(head_path, head_rows, head_columns)
    write_csv(operator_path, operator_rows, operator_columns)
    unclassified = [record for record in records if record["stage"] == "unclassified"]
    render_html(html_path, model_id, overall, stage_rows, layer_rows, head_rows, unclassified)

    total_work = sum(row["cycles"] for row in records)
    total_critical = sum(row["num_dominant_path_cycles_htp_0"] for row in records)
    trace_work = sum(row["trace_work_cycles"] for row in records)
    if trace_work != total_work:
        raise RuntimeError(f"Trace/QHAS work-cycle mismatch: trace={trace_work}, qhas={total_work}")
    for field, overall_field in (
        ("dram_read", "total_dram_read"), ("dram_write", "total_dram_write"),
        ("vtcm_read", "total_vtcm_read"), ("vtcm_write", "total_vtcm_write"),
    ):
        value = sum(row[field] for row in records)
        if value != overall[overall_field]:
            raise RuntimeError(f"QHAS {field} conservation failure: nodes={value}, overall={overall[overall_field]}")
    print(f"model={model_id} q_heads={num_q_heads} kv_heads={num_kv_heads} "
          f"operators={len(records)} work_cycles={total_work} "
          f"dominant_path_cycles={total_critical} unclassified={len(unclassified)}")
    print(f"generated: {stage_path}, {layer_path}, {head_path}, {operator_path}, {html_path}")
    if unclassified:
        print("unclassified sample: " + ", ".join(record["qnn_op"] for record in unclassified[:10]))


if __name__ == "__main__":
    main()
