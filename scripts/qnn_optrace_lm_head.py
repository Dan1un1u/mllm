#!/usr/bin/env python3
"""Generate an internal LM-head QHAS/Optrace profiling report."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_INFO = {
    "weight_transfer": (10, "Weight DMA transfer", "Packed W4 weights/scales moved toward VTCM"),
    "weight_wait": (20, "Weight DMA wait", "Critical dependency waiting for weight data"),
    "weight_expand": (30, "W4 expand / dequant", "HVX expands block-scaled W4 weights into QInt8 tiles"),
    "bias_transfer": (40, "Bias DMA transfer", "Bias data moved toward VTCM"),
    "bias_wait": (50, "Bias DMA wait", "Critical dependency waiting for bias data"),
    "hmx_mac": (60, "HMX MAC", "Matrix/tensor arithmetic for the vocabulary projection"),
    "output_format": (70, "Output format / requant", "Format conversion or output-side post-processing"),
    "sync": (80, "Checkpoint / sync", "DMA checkpoint and scheduling synchronization"),
    "other": (90, "Other lowered work", "Remaining LM-head HTP work"),
}


def n(row, key):
    return int(float(row.get(key, 0) or 0))


def resource(row):
    if row.get("hmx"):
        return "hmx"
    if row.get("hvx"):
        return "hvx"
    if row.get("dma_wait"):
        return "dma_wait"
    if row.get("dma"):
        return "dma_transfer"
    if row.get("dma_set") or row.get("sync"):
        return "sync"
    return "unresolved"


def category(row):
    name = row.get("htp_op", "").lower()
    if "expand_block_quant" in name:
        return "weight_expand"
    if "weights_to_vtcm" in name:
        return "weight_wait" if row.get("dma_wait") else "weight_transfer"
    if "bias_to_vtcm" in name:
        return "bias_wait" if row.get("dma_wait") else "bias_transfer"
    if row.get("hmx"):
        return "hmx_mac"
    if "forceformat" in name or "linearclip" in name or "convert" in name:
        return "output_format"
    if row.get("dma_set") or row.get("sync") or "checkpoint" in name or "sync" in name:
        return "sync"
    return "other"


def lane(row):
    kind = resource(row)
    if kind == "hmx":
        return "HMX"
    if kind == "hvx":
        tid = n(row, "tid")
        return f"HVX {tid - 512}" if 512 <= tid <= 517 else f"HVX tid={tid}"
    return {
        "dma_transfer": "DMA transfer",
        "dma_wait": "DMA wait",
        "sync": "Checkpoint / sync",
        "unresolved": "Unresolved",
    }[kind]


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def intersection_length(left, right):
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def interval_coverage(intervals, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def build_graph(path, graph):
    document = json.loads(path.read_text())
    data = document["data"]
    overall = data["htp_overall_summary"]["data"][0]
    qnn_rows = data["qnn_op_instances_nodes"]["data"]
    qnn = next(row for row in qnn_rows if row["qnn_op"] == "lm_head")
    rows = [row for row in data["htp_op_instances"]["data"] if row["qnn_op"] == "lm_head"]
    path_rows = [row for row in data["dominant_path_htp_0"]["data"] if row["qnn_op"] == "lm_head"]
    qnn_direct_total = sum(n(row, "num_dominant_path_cycles_htp_0") for row in qnn_rows)

    categories = defaultdict(lambda: {
        "instances": 0, "work": 0, "direct": 0, "dram_read": 0, "dram_write": 0,
        "vtcm_read": 0, "vtcm_write": 0, "resources": set(), "cycles": [],
    })
    htp_types = defaultdict(lambda: {
        "instances": 0, "work": 0, "direct": 0, "dram_read": 0,
        "vtcm_read": 0, "vtcm_write": 0, "resources": set(),
    })
    resources = Counter()
    resource_intervals = defaultdict(list)
    lane_intervals = defaultdict(list)
    op_by_id = defaultdict(list)
    for row in rows:
        direct, work = n(row, "num_dominant_path_cycles"), n(row, "cycles")
        cat, res, lane_name = category(row), resource(row), lane(row)
        group = categories[cat]
        group["instances"] += 1
        group["work"] += work
        group["direct"] += direct
        group["cycles"].append(work)
        group["resources"].add(res)
        for key in ("dram_read", "dram_write", "vtcm_read", "vtcm_write"):
            group[key] += n(row, key)
        kernel = htp_types[row["htp_op"]]
        kernel["instances"] += 1
        kernel["work"] += work
        kernel["direct"] += direct
        kernel["resources"].add(res)
        for key in ("dram_read", "vtcm_read", "vtcm_write"):
            kernel[key] += n(row, key)
        resources[res] += direct
        start, end = n(row, "start_cycle"), n(row, "start_cycle") + work
        resource_intervals[res].append((start, end))
        lane_intervals[lane_name].append((start, end))
        # HTP op_id is not globally unique across the six HVX worker lanes.
        # Retain all candidates and resolve a dominant-path segment by its
        # actual time overlap below.
        op_by_id[row["op_id"]].append(row)

    for group in categories.values():
        group["resources"] = "+".join(sorted(group["resources"]))
        group["mean_cycles"] = group["work"] / group["instances"] if group["instances"] else 0
        group["p95_cycles"] = percentile(group.pop("cycles"), 0.95)
    category_rows = []
    for key, meta in CATEGORY_INFO.items():
        if key not in categories:
            continue
        row = dict(categories[key])
        row.update({"key": key, "order": meta[0], "label": meta[1], "description": meta[2]})
        category_rows.append(row)

    kernel_rows = []
    for name, values in sorted(htp_types.items(), key=lambda item: -item[1]["direct"]):
        row = dict(values)
        row.update({"name": name, "resources": "+".join(sorted(values["resources"]))})
        kernel_rows.append(row)

    all_intervals = [interval for values in lane_intervals.values() for interval in values]
    start = min(a for a, _ in all_intervals)
    end = max(b for _, b in all_intervals)
    span = end - start
    critical_by_lane = defaultdict(list)
    for item in path_rows:
        path_start, path_end = n(item, "start_cycle"), n(item, "end_cycle")
        candidates = op_by_id.get(item["op_id"], [])
        if candidates:
            def match_score(source):
                source_start = n(source, "start_cycle")
                source_end = source_start + n(source, "cycles")
                overlap = max(0, min(path_end, source_end) - max(path_start, source_start))
                distance = abs(source_start - path_start)
                return overlap, -distance

            source = max(candidates, key=match_score)
            critical_by_lane[lane(source)].append((n(item, "start_cycle"), n(item, "end_cycle")))

    lane_order = ["HMX"] + [f"HVX {i}" for i in range(6)] + [
        "DMA transfer", "DMA wait", "Checkpoint / sync", "Unresolved"
    ]
    bins = 180
    tracks = []
    for lane_name in lane_order:
        active = merge_intervals(lane_intervals.get(lane_name, []))
        critical = merge_intervals(critical_by_lane.get(lane_name, []))
        if not active and not critical:
            continue
        cells = []
        for index in range(bins):
            left = start + span * index / bins
            right = start + span * (index + 1) / bins
            width = right - left
            cells.append({
                "occupancy": min(1, interval_coverage(active, left, right) / width),
                "critical": interval_coverage(critical, left, right) > 0,
            })
        tracks.append({"lane": lane_name, "cells": cells})

    merged_resources = {key: merge_intervals(values) for key, values in resource_intervals.items()}
    active_by_resource = {
        key: sum(end - start for start, end in values) for key, values in merged_resources.items()
    }
    overlaps = {
        "hmx_hvx": intersection_length(resource_intervals["hmx"], resource_intervals["hvx"]),
        "hmx_dma": intersection_length(resource_intervals["hmx"], resource_intervals["dma_transfer"]),
        "hvx_dma": intersection_length(resource_intervals["hvx"], resource_intervals["dma_transfer"]),
        "wait_compute": intersection_length(
            resource_intervals["dma_wait"], resource_intervals["hmx"] + resource_intervals["hvx"]
        ),
    }
    hmx_rows = [row for row in rows if row.get("hmx")]
    expand_rows = [row for row in rows if "expand_block_quant" in row.get("htp_op", "")]
    common_hmx_dims = Counter(tuple(row.get("dims", [])) for row in hmx_rows).most_common(1)
    return {
        "id": graph,
        "overall": {
            "graph_execute_us": n(overall, "graph_execute_us"),
            "qhas_time_us": n(overall, "time_us"),
            "timeline_cycles": n(overall, "timeline_cycles"),
            "qnn_direct_total": qnn_direct_total,
        },
        "qnn": qnn,
        "attributed_us": n(overall, "graph_execute_us") * n(qnn, "num_dominant_path_cycles_htp_0") / qnn_direct_total,
        "categories": category_rows,
        "kernels": kernel_rows,
        "resources": dict(resources),
        "active_by_resource": active_by_resource,
        "overlaps": overlaps,
        "tracks": tracks,
        "envelope_cycles": span,
        "dp_events": len(path_rows),
        "tile": {
            "hmx_instances": len(hmx_rows),
            "expand_instances": len(expand_rows),
            "expand_per_hmx": len(expand_rows) / len(hmx_rows) if hmx_rows else 0,
            "hmx_dims": list(common_hmx_dims[0][0]) if common_hmx_dims else [],
            "hvx_threads": sorted({n(row, "tid") for row in expand_rows}),
        },
    }


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3 SM8750 V79 — LM head internal profiling</title><style>
:root{--ink:#19283b;--muted:#68798d;--line:#d7e0e9;--paper:#fff;--bg:#eef2f6;--hmx:#2868aa;--hvx:#2a9b80;--dma:#d98a22;--wait:#c65359;--sync:#778393;--gold:#f4d35e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,"Noto Sans SC",system-ui,sans-serif}header{padding:29px 40px;background:linear-gradient(125deg,#183a62,#237789);color:#fff}h1{margin:0 0 6px;font-size:28px}header p{margin:0;color:#d9e9f3}main{max-width:1450px;margin:16px auto;padding:0 16px 40px}.panel{margin:14px 0;padding:17px;background:var(--paper);border:1px solid var(--line);border-radius:12px}.toolbar{display:flex;gap:8px;align-items:center}.tab{padding:7px 13px;border:1px solid #b9c9d8;border-radius:7px;background:#fff;color:#285174;font:inherit}.tab.active{background:#246f9e;color:#fff;border-color:#246f9e}.cards{display:grid;grid-template-columns:repeat(4,minmax(190px,1fr));gap:10px;margin-top:13px}.card{padding:12px;border:1px solid var(--line);border-radius:9px}.card small{display:block;color:var(--muted)}.card b{display:block;color:#24557f;font-size:20px}.note{padding:10px 13px;border-left:4px solid #e19a27;background:#fff7e9}.resource-bar{display:flex;height:22px;margin:12px 0;overflow:hidden;border-radius:4px;box-shadow:0 0 0 1px #8a99a844}.resource-bar i{height:100%}.legend{display:flex;flex-wrap:wrap;gap:14px;color:#4c6075}.sw{display:inline-block;width:22px;height:10px;margin-right:5px;border-radius:2px}.hmx{background:var(--hmx)}.hvx{background:var(--hvx)}.dma{background:var(--dma)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:6px 7px;border:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#edf3f8;color:#405870}th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{text-align:left}.scroll{overflow-x:auto}.timeline{min-width:1100px}.lane{display:grid;grid-template-columns:135px 1fr;gap:8px;min-height:25px;border-bottom:1px solid #edf1f5}.lane-label{text-align:right;padding:3px 5px;font-size:12px;font-weight:650}.cells{display:grid;grid-template-columns:repeat(180,1fr);gap:0;background:#f2f5f8}.cell{min-width:2px;height:17px;margin-top:3px}.critical{box-shadow:inset 0 -3px 0 var(--gold)}.metrics{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}.metric{padding:3px 8px;border:1px solid #cad9e6;border-radius:12px;background:#f8fbfd;font-size:11px}.finding{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.finding div{padding:12px;border:1px solid var(--line);border-radius:9px;background:#fafcfe}.finding b{display:block;color:#24557f;margin-bottom:4px}@media(max-width:850px){.cards,.finding{grid-template-columns:1fr 1fr}}@media(max-width:550px){.cards,.finding{grid-template-columns:1fr}}</style></head><body>
<header><h1>Qwen3-1.7B × SM8750 V79：LM head 内部 Profiling</h1><p>LPBQ vocabulary projection · QHAS direct-critical decomposition · HMX/HVX/DMA timeline</p></header><main>
<section class="panel"><div class="toolbar"><b>Graph</b><button class="tab active" data-g="s1">s1 · decode</button><button class="tab" data-g="s32">s32 · 32-token</button></div><div class="cards" id="cards"></div><div id="resource"></div></section>
<section class="panel"><h2>内部关键路径阶段</h2><p class="note">Direct critical 来自官方 QHAS <code>htp_op_instances.num_dominant_path_cycles</code>，阶段之间可以相加回 LM head 的 QNN-node contribution；Work cycles 仍然可能并行，不能当作 latency。</p><div class="scroll"><table><thead><tr><th>#</th><th>Lowered stage</th><th>Resource</th><th>Instances</th><th>Direct cycles</th><th>LM-head DP</th><th>Attributed μs</th><th>Work cycles</th><th>Mean/instance</th><th>P95/instance</th><th>DRAM read</th><th>VTCM R/W</th><th>Description</th></tr></thead><tbody id="categories"></tbody></table></div></section>
<section class="panel"><h2>LM head 局部资源时间线</h2><p class="note">颜色深度表示该时间窗内的 lane occupancy；底部黄色表示该 bin 中出现官方 dominant-path segment。这里用于观察六条 HVX 权重展开与单条 HMX、DMA/wait 之间的流水，而不是把各行长度相加。</p><div class="scroll"><div class="timeline" id="timeline"></div></div><div class="metrics" id="metrics"></div></section>
<section class="panel"><h2>底层 kernel 明细</h2><div class="scroll"><table><thead><tr><th>#</th><th>HTP kernel</th><th>Resource</th><th>Instances</th><th>Direct cycles</th><th>LM-head DP</th><th>Work cycles</th><th>DRAM read</th><th>VTCM R/W</th></tr></thead><tbody id="kernels"></tbody></table></div></section>
<section class="panel"><h2>为什么 LM head 慢？</h2><div class="finding" id="findings"></div></section>
</main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent);let graph='s1';function fmt(n,d=2){return Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}function pct(n,d){return 100*n/Math.max(d,1)}function mb(n){return fmt(n/1e6,2)+' MB'}function resColor(r){return r==='hmx'?'var(--hmx)':r==='hvx'?'var(--hvx)':r==='dma_wait'?'var(--wait)':r==='dma_transfer'?'var(--dma)':'var(--sync)'}
function cards(g){const q=g.qnn,d=g.tile;document.getElementById('cards').innerHTML=`<div class="card"><small>LM-head direct / full graph</small><b>${fmt(q.num_dominant_path_cycles_htp_0)} cyc</b><small>${fmt(q.percent_dominant_path_cycles_htp_0,3)}% · ≈ ${fmt(g.attributed_us,2)} μs</small></div><div class="card"><small>DRAM weight read</small><b>${mb(q.dram_read)}</b><small>每次执行完整流式读取</small></div><div class="card"><small>Lowered HTP instances</small><b>${fmt(q.num_htp_ops)}</b><small>${fmt(d.hmx_instances)} HMX tiles · ${fmt(d.expand_instances)} expands</small></div><div class="card"><small>Observed tile pipeline</small><b>${fmt(d.expand_per_hmx,2)} HVX / HMX tile</b><small>HMX output ${d.hmx_dims.join('×')} · HVX tids ${d.hvx_threads.join(', ')}</small></div>`}
function resource(g){const r=g.resources,hmx=r.hmx||0,hvx=r.hvx||0,dma=(r.dma_transfer||0)+(r.dma_wait||0)+(r.sync||0),total=hmx+hvx+dma;document.getElementById('resource').innerHTML=`<div class="resource-bar"><i style="width:${pct(hmx,total)}%;background:var(--hmx)"></i><i style="width:${pct(hvx,total)}%;background:var(--hvx)"></i><i style="width:${pct(dma,total)}%;background:var(--dma)"></i></div><div class="legend"><span><i class="sw hmx"></i>HMX ${fmt(hmx)} · ${fmt(pct(hmx,total),2)}%</span><span><i class="sw hvx"></i>HVX ${fmt(hvx)} · ${fmt(pct(hvx,total),2)}%</span><span><i class="sw dma"></i>DMA/wait/sync ${fmt(dma)} · ${fmt(pct(dma,total),2)}%</span></div>`}
function categories(g){const total=g.qnn.num_dominant_path_cycles_htp_0,scale=g.attributed_us/total;document.getElementById('categories').innerHTML=g.categories.map((x,i)=>`<tr><td>${i+1}</td><td><b>${x.label}</b></td><td>${x.resources}</td><td>${fmt(x.instances)}</td><td>${fmt(x.direct)}</td><td>${fmt(pct(x.direct,total),2)}%</td><td>${fmt(x.direct*scale,2)}</td><td>${fmt(x.work)}</td><td>${fmt(x.mean_cycles,1)}</td><td>${fmt(x.p95_cycles)}</td><td>${mb(x.dram_read)}</td><td>${mb(x.vtcm_read)} / ${mb(x.vtcm_write)}</td><td>${x.description}</td></tr>`).join('')}
function timeline(g){document.getElementById('timeline').innerHTML=g.tracks.map(t=>`<div class="lane"><span class="lane-label">${t.lane}</span><div class="cells">${t.cells.map(c=>`<i class="cell${c.critical?' critical':''}" style="background:${resColor(t.lane.startsWith('HMX')?'hmx':t.lane.startsWith('HVX')?'hvx':t.lane==='DMA wait'?'dma_wait':t.lane==='DMA transfer'?'dma_transfer':'sync')};opacity:${.08+.92*Math.sqrt(c.occupancy)}"></i>`).join('')}</div></div>`).join('');const a=g.active_by_resource,o=g.overlaps;document.getElementById('metrics').innerHTML=[`Envelope ${fmt(g.envelope_cycles)} cyc`,`HMX active ${fmt(a.hmx||0)}`,`HVX union ${fmt(a.hvx||0)}`,`DMA active ${fmt(a.dma_transfer||0)}`,`DMA wait active ${fmt(a.dma_wait||0)}`,`HMX∩HVX ${fmt(o.hmx_hvx)}`,`HVX∩DMA ${fmt(o.hvx_dma)}`,`HMX∩DMA ${fmt(o.hmx_dma)}`,`wait∩compute ${fmt(o.wait_compute)}`].map(x=>`<span class="metric">${x}</span>`).join('')}
function kernels(g){const total=g.qnn.num_dominant_path_cycles_htp_0;document.getElementById('kernels').innerHTML=g.kernels.slice(0,20).map((x,i)=>`<tr><td>${i+1}</td><td><code>${x.name}</code></td><td>${x.resources}</td><td>${fmt(x.instances)}</td><td>${fmt(x.direct)}</td><td>${fmt(pct(x.direct,total),2)}%</td><td>${fmt(x.work)}</td><td>${mb(x.dram_read)}</td><td>${mb(x.vtcm_read)} / ${mb(x.vtcm_write)}</td></tr>`).join('')}
function findings(g){const total=g.qnn.num_dominant_path_cycles_htp_0,r=g.resources,expand=g.categories.find(x=>x.key==='weight_expand'),wait=(g.categories.find(x=>x.key==='weight_wait')||{direct:0}).direct+(g.categories.find(x=>x.key==='bias_wait')||{direct:0}).direct,hmx=r.hmx||0;document.getElementById('findings').innerHTML=`<div><b>1. 大词表权重流是根因</b>hidden 2048 → vocab 151,936 的投影每次读取约 ${mb(g.qnn.dram_read)}。W4 已降低容量，但 block-scale 元数据和完整词表扫描仍然昂贵。</div><div><b>2. HVX 解压比 HMX MAC 更重</b>W4 expand/dequant 占 LM-head critical 的 ${fmt(pct(expand.direct,total),1)}%，HMX MAC 仅占 ${fmt(pct(hmx,total),1)}%。当前 kernel 必须先把 W4 展开为 QInt8 tile。</div><div><b>3. DMA 等待没有完全隐藏</b>Weight/bias wait 占 ${fmt(pct(wait,total),1)}%；观察到 HMX∩DMA transfer 为 ${fmt(g.overlaps.hmx_dma)} cycles，说明直接搬运没有和 HMX 同时活跃。</div><div><b>4. 六路 HVX 正在喂一条 HMX</b>${fmt(g.tile.expand_instances)} 个展开任务正好约为 ${fmt(g.tile.expand_per_hmx,1)}× ${fmt(g.tile.hmx_instances)} 个 HMX tile；瓶颈是供数流水，而非缺少 HVX 并行度。</div><div><b>5. W4A4 / native low-bit HMX 的机会</b>若 HMX 能直接消费低位权重/激活，可减少 HVX 展开和 QInt8 VTCM 中间载荷；可优先瞄准当前 expand + wait 的 ${fmt(pct(expand.direct+wait,total),1)}% critical share。</div><div><b>6. 算法层减少完整词表扫描</b>词表裁剪、分层/近似 top-k、draft/speculative 路径可以减少必须执行的 LM-head tile 数；需要单独评估精度和采样语义。</div>`}
function render(){const g=D[graph];cards(g);resource(g);categories(g);timeline(g);kernels(g);findings(g)}document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{graph=b.dataset.g;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));render()});render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {}
    for graph in ("s1", "s32"):
        path = args.results_dir / f"qwen3-sm8750-v79-{graph}-chrometrace_qnn_htp_analysis_summary.json"
        data[graph] = build_graph(path, graph)
        value = data[graph]
        print(
            f"{graph}: direct={value['qnn']['num_dominant_path_cycles_htp_0']} "
            f"share={value['qnn']['percent_dominant_path_cycles_htp_0']:.3f}% "
            f"attributed_us={value['attributed_us']:.2f} htp_ops={value['qnn']['num_htp_ops']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    args.output.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print(f"generated: {args.output}")


if __name__ == "__main__":
    main()
