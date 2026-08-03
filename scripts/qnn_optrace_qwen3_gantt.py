#!/usr/bin/env python3
"""Generate an interactive Qwen3 structure Gantt chart from archived QNN Optrace results.

The chart keeps three quantities separate:

* the real start/end envelope on the captured HTP timeline,
* overlap-aware active interval union and summed work,
* QHAS direct dominant-path attribution.

Quantization event envelopes are derived from the archived Chrome Trace and
annotated with logical W4A16 types plus post-lowering physical dtypes.
"""

import argparse
import csv
import gc
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from qnn_optrace_quantization import (
    CATEGORY_INFO,
    attach_manifest_io,
    load_manifests,
    load_runtime_events,
)
from qnn_optrace_summary import interval_stats


GLOBAL_STAGES = {
    "embedding", "final_rmsnorm", "lm_head"
}

CONVERSION_CATEGORIES = {
    "lpbq_weight_expand_dequant", "explicit_convert_requant",
    "fused_linearclip_requant", "matmul_signed_conversion",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, integer=False):
    value = row.get(key, 0) or 0
    return int(float(value)) if integer else float(value)


def shorten(value, limit=360):
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def infer_graph_execute_us(layer_rows):
    candidates = []
    for row in layer_rows:
        percent = number(row, "critical_path_percent")
        if percent:
            candidates.append(number(row, "critical_path_us_estimate") * 100.0 / percent)
    if not candidates:
        return 0.0
    candidates.sort()
    return candidates[len(candidates) // 2]


def aggregate_quant(records):
    groups = {}
    stage_intervals = defaultdict(list)
    stage_work = Counter()
    stage_dominant = Counter()
    for record in records:
        key = (record["layer"], record["stage"], record["category"])
        group = groups.setdefault(key, {
            "intervals": [], "work": 0, "dominant": 0, "events": 0,
            "physical_inputs": Counter(), "physical_outputs": Counter(),
            "logical_inputs": Counter(), "logical_outputs": Counter(), "flags": set(),
        })
        interval = (record["start_cycle"], record["start_cycle"] + record["duration_cycles"])
        group["intervals"].append(interval)
        group["work"] += record["duration_cycles"]
        group["dominant"] += record["dominant_path_cycles"]
        group["events"] += 1
        group["physical_inputs"][record["input_data_types"]] += record["duration_cycles"]
        group["physical_outputs"][record["data_type"]] += record["duration_cycles"]
        group["logical_inputs"][record.get("qnn_input_types") or "not in manifest"] += record["duration_cycles"]
        group["logical_outputs"][record.get("qnn_output_types") or "not in manifest"] += record["duration_cycles"]
        group["flags"].add(record["flags"])
        stage_key = (record["layer"], record["stage"])
        stage_intervals[stage_key].append(interval)
        stage_work[stage_key] += record["duration_cycles"]
        stage_dominant[stage_key] += record["dominant_path_cycles"]

    categories = defaultdict(list)
    for (layer, stage, category), group in groups.items():
        first, last, wall, active, _, parallel = interval_stats(group.pop("intervals"))
        categories[(layer, stage)].append({
            "category": category,
            "label": CATEGORY_INFO[category][1],
            "order": CATEGORY_INFO[category][0],
            "start": first,
            "end": last,
            "wall": wall,
            "active": active,
            "work": group["work"],
            "dominant": group["dominant"],
            "events": group["events"],
            "parallel": parallel,
            "physical_inputs": shorten(group["physical_inputs"].most_common(1)[0][0]),
            "physical_outputs": shorten(group["physical_outputs"].most_common(1)[0][0]),
            "logical_inputs": shorten(group["logical_inputs"].most_common(1)[0][0]),
            "logical_outputs": shorten(group["logical_outputs"].most_common(1)[0][0]),
            "flags": "+".join(sorted(group["flags"])),
        })
    for values in categories.values():
        values.sort(key=lambda item: item["order"])

    totals = {}
    for stage_key, intervals in stage_intervals.items():
        first, last, wall, active, _, parallel = interval_stats(intervals)
        totals[stage_key] = {
            "start": first, "end": last, "wall": wall, "active": active,
            "work": stage_work[stage_key], "dominant": stage_dominant[stage_key],
            "parallel": parallel,
        }
    return categories, totals


def cooked_stage(row, graph_execute_us, total_dominant, quant_categories, quant_totals):
    layer_text = row.get("layer", "")
    layer = int(layer_text) if layer_text not in (None, "") else None
    stage = row["stage"]
    dominant = number(row, "num_dominant_path_cycles_htp_0", integer=True)
    critical_us = (
        number(row, "critical_path_us_estimate")
        if row.get("critical_path_us_estimate") not in (None, "")
        else graph_execute_us * dominant / total_dominant if total_dominant else 0.0
    )
    stage_key = (layer, stage)
    quant = quant_totals.get(stage_key, {})
    active = number(row, "active_union_cycles", integer=True)
    return {
        "layer": layer,
        "order": number(row, "stage_order", integer=True),
        "stage": stage,
        "label": row.get("stage_label", stage),
        "bottleneck": row.get("bottleneck_class", ""),
        "resources": row.get("kernel_resources", ""),
        "start": number(row, "start_cycle", integer=True),
        "end": number(row, "end_cycle", integer=True),
        "wall": number(row, "wall_span_cycles", integer=True),
        "active": active,
        "work": number(row, "cycles", integer=True),
        "dominant": dominant,
        "critical_us": critical_us,
        "parallel": number(row, "max_parallelism", integer=True),
        "dram": number(row, "dram_read", integer=True) + number(row, "dram_write", integer=True),
        "vtcm": number(row, "vtcm_read", integer=True) + number(row, "vtcm_write", integer=True),
        "occupancy": active / max(number(row, "wall_span_cycles"), 1),
        "quant_active": quant.get("active", 0),
        "quant_work": quant.get("work", 0),
        "quant_dominant": quant.get("dominant", 0),
        "quant_active_ratio": quant.get("active", 0) / max(active, 1),
        "quant_categories": quant_categories.get(stage_key, []),
    }


def build_graph(result_dir, graph):
    prefix = result_dir / f"qwen3-sm8750-v79-{graph}"
    layer_rows = read_csv(Path(f"{prefix}-qwen3-layer-stage.csv"))
    stage_rows = read_csv(Path(f"{prefix}-qwen3-stage-summary.csv"))
    operator_rows = read_csv(Path(f"{prefix}-qwen3-operator-structure.csv"))
    graph_execute_us = infer_graph_execute_us(layer_rows)
    total_dominant = sum(number(row, "num_dominant_path_cycles_htp_0", integer=True) for row in operator_rows)

    trace = Path(f"{prefix}-chrometrace.json")
    qhas = Path(f"{prefix}-chrometrace_qnn_htp_analysis_summary.json")
    manifest = result_dir / f"model.0.{graph}_quant_manifest.json"
    quant_records = load_runtime_events(trace, qhas)
    _, tensors, operations = load_manifests([manifest])
    attach_manifest_io(quant_records, tensors, operations)
    quant_categories, quant_totals = aggregate_quant(quant_records)

    stages = [
        cooked_stage(row, graph_execute_us, total_dominant, quant_categories, quant_totals)
        for row in layer_rows
    ]
    globals_ = [
        cooked_stage(row, graph_execute_us, total_dominant, quant_categories, quant_totals)
        for row in stage_rows if row["stage"] in GLOBAL_STAGES
    ]
    all_stages = stages + globals_
    start = min(item["start"] for item in all_stages if item["end"] > item["start"])
    end = max(item["end"] for item in all_stages)
    layer_totals = defaultdict(float)
    for item in stages:
        layer_totals[item["layer"]] += item["critical_us"]
    quant_work = sum(record["duration_cycles"] for record in quant_records)
    quant_dominant = sum(record["dominant_path_cycles"] for record in quant_records)
    conversion_work = sum(
        record["duration_cycles"] for record in quant_records
        if record["category"] in CONVERSION_CATEGORIES
    )
    conversion_dominant = sum(
        record["dominant_path_cycles"] for record in quant_records
        if record["category"] in CONVERSION_CATEGORIES
    )
    del quant_records, tensors, operations
    gc.collect()
    return {
        "id": graph,
        "title": "single-token decode" if graph == "s1" else "32-token chunk",
        "graph_execute_us": graph_execute_us,
        "start": start,
        "end": end,
        "span": end - start,
        "total_dominant": total_dominant,
        "quant_work": quant_work,
        "quant_dominant": quant_dominant,
        "conversion_work": conversion_work,
        "conversion_dominant": conversion_dominant,
        "stages": stages,
        "globals": globals_,
        "layer_totals": dict(sorted(layer_totals.items())),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3 SM8750 V79 — Structure × Quantization Gantt</title>
<style>
:root{--ink:#182438;--muted:#66768b;--line:#d8e1eb;--paper:#fff;--bg:#eef2f7;--quant:#8b4ec2;--critical:#d74b4b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,"Noto Sans SC",system-ui,sans-serif}
header{padding:30px 38px;background:linear-gradient(125deg,#132f55,#176f86);color:#fff}header h1{margin:0 0 7px;font-size:28px}header p{margin:0;color:#d6e8f5}
main{max-width:1600px;margin:18px auto;padding:0 18px 40px}.panel{margin:14px 0;padding:18px;background:var(--paper);border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 15px #24384d10}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.tab,.btn,select{border:1px solid #bdcbd9;background:#fff;color:#23466c;border-radius:7px;padding:7px 11px;font:inherit}.tab.active{background:#1d6598;color:#fff;border-color:#1d6598}.btn.active{background:#e9f2fb;border-color:#6c9bc5}
.cards{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px;margin-top:14px}.card{padding:12px;border:1px solid var(--line);border-radius:9px}.card small{display:block;color:var(--muted)}.card b{font-size:20px;color:#214f7c}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0;color:#42536a}.sw{display:inline-block;width:18px;height:9px;margin-right:5px;border-radius:3px}.quant-sw{background:repeating-linear-gradient(135deg,#8b4ec2 0 4px,#d8b8ef 4px 8px)}
.note{padding:10px 13px;border-left:4px solid #e39a25;background:#fff6e7}.scroll{overflow-x:auto}.chart{min-width:1250px}.axis{position:relative;height:34px;margin-left:128px;border-bottom:1px solid #9eb0c4}.tick{position:absolute;bottom:0;height:8px;border-left:1px solid #9eb0c4}.tick span{position:absolute;bottom:9px;transform:translateX(-50%);white-space:nowrap;color:#52657b;font-size:11px}
.gantt-row{display:grid;grid-template-columns:120px 1fr;gap:8px;min-height:38px;border-bottom:1px solid #eef2f6}.row-label{padding:9px 5px;text-align:right;font-weight:650;color:#334d68}.track{position:relative;min-height:38px;background-image:linear-gradient(to right,#edf1f5 1px,transparent 1px);background-size:10% 100%}
.bar{position:absolute;height:5px;min-width:2px;border-radius:3px;border:1px solid #ffffffa8;cursor:pointer;opacity:.86}.bar:hover{z-index:20;height:9px;filter:saturate(1.4);box-shadow:0 0 0 2px #172c4933}.bar.critical{box-shadow:0 0 0 1px var(--critical)}.bar.quantized:after{content:"";position:absolute;left:0;right:0;bottom:-3px;height:2px;background:repeating-linear-gradient(90deg,#7428a8 0 4px,#e0bdf3 4px 7px)}
.detail-row{display:grid;grid-template-columns:210px 1fr 90px;gap:8px;min-height:34px;border-bottom:1px solid #edf1f5}.detail-label{padding:7px 5px;text-align:right}.detail-track{position:relative;background:#f8fafc}.detail-bar{position:absolute;top:8px;height:16px;min-width:3px;border-radius:4px;cursor:pointer;opacity:.8}.detail-bar .active-indicator{position:absolute;left:50%;top:4px;height:8px;transform:translateX(-50%);background:#fff;border-radius:5px;opacity:.78}.quant-seg{position:absolute;bottom:-4px;height:3px;min-width:2px;background:var(--quant);border-radius:2px}.metric{padding:7px 3px;text-align:right;font-variant-numeric:tabular-nums;color:#43566d}
.tooltip{position:fixed;z-index:1000;display:none;max-width:520px;padding:12px 14px;border-radius:9px;background:#10233b;color:#f4f8fc;box-shadow:0 8px 26px #0005;pointer-events:none;font-size:12px;white-space:pre-line}.tooltip b{color:#8fd4ff}.tooltip .q{color:#e2b8ff}
table{width:100%;border-collapse:collapse;font-size:12.5px}th,td{border:1px solid var(--line);padding:6px 8px;text-align:right;vertical-align:top}th{background:#edf3f8}th:first-child,td:first-child{text-align:left}.qcell{color:#64368a;font-weight:650}
.family-norm{background:#4a89c7}.family-qkv{background:#2e9f8b}.family-prep{background:#68a45d}.family-attn{background:#d49a32}.family-output{background:#d16b5b}.family-mlp{background:#7a62b6}.family-global{background:#64748b}
@media(max-width:900px){.cards{grid-template-columns:1fr 1fr}header{padding:22px}main{padding:0 8px}}
</style></head><body>
<header><h1>Qwen3-1.7B × SM8750 V79：Structure × Quantization 甘特图</h1><p>只使用 2026-07-13 归档的 s1/s32 Optrace、QHAS、structure CSV 与 Quantization Manifest</p></header>
<main>
<section class="panel"><div class="toolbar"><b>Graph</b><button class="tab active" data-graph="s1">s1 · decode</button><button class="tab" data-graph="s32">s32 · 32-token</button><span style="flex:1"></span><label>Detail <select id="layerSelect"></select></label></div><div class="cards" id="cards"></div></section>
<section class="panel"><h2>全图 Layer Pipeline</h2><p class="note">横轴是真实 captured HTP timeline；每条彩色 bar 是该 layer/stage 所有事件的 <b>start→end envelope</b>，不是连续占用。bar 纵向分成 Norm、QKV、attention preparation、attention core、output/residual、MLP 六条微泳道；紫色下划线表示该 stage 内存在量化/反量化/权重展开或 requant kernel。</p><div class="legend" id="legend"></div><div class="scroll"><div class="chart" id="overview"></div></div></section>
<section class="panel"><h2 id="detailTitle">Layer detail</h2><p>此处横轴会缩放到所选 layer 的局部时间窗口。实心白条是压缩后的 <b>active-union / envelope</b> 指示器，只表达占空比，不表达其在 envelope 内的真实位置。下方紫色细条是各量化 phase 的真实 start→end envelope；不同 phase 仍可能重叠。</p><div class="scroll"><div class="chart" id="detail"></div></div></section>
<section class="panel"><h2>Selected Layer：结构 × 量化明细</h2><div class="scroll"><table><thead><tr><th>Stage</th><th>Envelope</th><th>Active union</th><th>Work</th><th>Direct critical</th><th>Quant active</th><th>Quant work</th><th>Quant direct</th><th>Logical I/O / Physical I/O</th></tr></thead><tbody id="detailTable"></tbody></table></div></section>
<section class="panel"><h2>读图边界</h2><ul><li>不要沿横轴把 bar 宽度相加：Q/K/V、各 head、HVX/HMX/DMA 大量重叠。</li><li>Envelope 用于看调度先后；active union 用于看去重后的活动量；work 用于看引擎总工作；direct critical 用于看关键路径归因。</li><li>“W4A16”是 pre-finalize logical recipe；紫色 phase 展示 V79 lowering 后的 W4→QInt8 expand、DMA/VTCM staging、HMX MAC、signed conversion 与 requant。</li></ul></section>
</main><div class="tooltip" id="tooltip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);let graph='s1';let selected=18;
const families={norm:['input_rmsnorm','post_attention_rmsnorm'],qkv:['q_projection','k_projection','v_projection'],prep:['q_head_rmsnorm','k_head_rmsnorm','q_rope','k_rope','kv_cache_update'],attn:['qk_similarity','scale_mask_softmax','attention_value','head_merge'],output:['o_projection','attention_residual'],mlp:['mlp_gate_projection','mlp_up_projection','mlp_silu','mlp_gate_product','mlp_down_projection','mlp_residual']};
const familyNames={norm:'Norm',qkv:'Q/K/V projection',prep:'Attention preparation',attn:'Attention core',output:'O projection / residual',mlp:'MLP',global:'Embedding / final / LM head'};
const familyIndex={norm:0,qkv:1,prep:2,attn:3,output:4,mlp:5,global:2};
function family(stage){for(const [k,v] of Object.entries(families))if(v.includes(stage))return k;return 'global'}
function fmt(n,d=2){return Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}function pct(n){return Math.max(0,Math.min(100,n))}
function position(g,start,end){return {left:pct(100*(start-g.start)/g.span),width:Math.max(.13,100*(end-start)/g.span)}}
function axis(g){let s='<div class="axis">';for(let i=0;i<=10;i++){const cyc=g.span*i/10;const ms=g.graph_execute_us*i/10/1000;s+=`<div class="tick" style="left:${i*10}%"><span>${fmt(cyc/1e6,1)}M cyc<br>${fmt(ms,2)} ms</span></div>`}return s+'</div>'}
function stageTip(s){let q=s.quant_categories.map(x=>`• ${x.label}: env ${fmt(x.wall)} / active ${fmt(x.active)} / work ${fmt(x.work)} / direct ${fmt(x.dominant)} cyc\n  logical ${x.logical_inputs} → ${x.logical_outputs}\n  physical ${x.physical_inputs} → ${x.physical_outputs}`).join('\n');return `${s.label}  [${s.stage}]\nLayer: ${s.layer===null?'global':s.layer}\nEnvelope: ${fmt(s.wall)} cyc\nActive union: ${fmt(s.active)} cyc (${fmt(100*s.occupancy,1)}%)\nSummed work: ${fmt(s.work)} cyc\nDirect critical: ${fmt(s.dominant)} cyc ≈ ${fmt(s.critical_us,2)} μs\nResources: ${s.resources}\nDRAM: ${fmt(s.dram/1e6,2)} MB · VTCM: ${fmt(s.vtcm/1e6,2)} MB${q?'\n\nQUANTIZATION\n'+q:''}`}
function bindTips(root){root.querySelectorAll('[data-tip]').forEach(el=>{el.onmouseenter=e=>showTip(e,el.dataset.tip);el.onmousemove=moveTip;el.onmouseleave=hideTip})}
function showTip(e,t){const el=document.getElementById('tooltip');el.textContent=t;el.style.display='block';moveTip(e)}function moveTip(e){const el=document.getElementById('tooltip');el.style.left=Math.min(innerWidth-el.offsetWidth-12,e.clientX+14)+'px';el.style.top=Math.min(innerHeight-el.offsetHeight-12,e.clientY+14)+'px'}function hideTip(){document.getElementById('tooltip').style.display='none'}
function renderLegend(){document.getElementById('legend').innerHTML=Object.keys(familyNames).map(k=>`<span><i class="sw family-${k}"></i>${familyNames[k]}</span>`).join('')+'<span><i class="sw quant-sw"></i>quant/dequant/convert phase present</span><span><i class="sw" style="border:2px solid #d74b4b"></i>high direct-critical attribution</span>'}
function renderCards(g){const vals=Object.values(g.layer_totals),med=[...vals].sort((a,b)=>a-b)[Math.floor(vals.length/2)],l18=g.layer_totals['18'];document.getElementById('cards').innerHTML=`<div class="card"><small>Graph execute</small><b>${fmt(g.graph_execute_us/1000,3)} ms</b></div><div class="card"><small>Qwen3 structure envelope</small><b>${fmt(g.span/1e6,2)}M cyc</b></div><div class="card"><small>Conversion / requant work</small><b>${fmt(g.conversion_work/1e6,1)}M cyc</b></div><div class="card"><small>Conversion / requant direct</small><b>${fmt(g.conversion_dominant/1e6,2)}M cyc</b></div><div class="card"><small>Layer 18 / layer median</small><b>${fmt(l18,1)} / ${fmt(med,1)} μs</b></div>`}
function renderOverview(g){let out=axis(g);const rows=[{label:'Global',items:g.globals},...Object.keys(g.layer_totals).map(l=>({label:'Layer '+l,items:g.stages.filter(s=>s.layer==Number(l))}))];for(const row of rows){out+=`<div class="gantt-row"><div class="row-label">${row.label}</div><div class="track">`;for(const s of row.items){if(s.end<=s.start)continue;const p=position(g,s.start,s.end),f=family(s.stage),top=4+familyIndex[f]*5.2,crit=s.critical_us>Math.max(20,g.graph_execute_us*.004);out+=`<div class="bar family-${f}${s.quant_categories.length?' quantized':''}${crit?' critical':''}" style="left:${p.left}%;width:${p.width}%;top:${top}px" data-tip="${htmlEscape(stageTip(s))}"></div>`}out+='</div></div>'}document.getElementById('overview').innerHTML=out;bindTips(document.getElementById('overview'))}
function htmlEscape(s){return s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;')}
function renderDetail(g){const items=selected==='global'?g.globals:g.stages.filter(s=>s.layer===Number(selected));document.getElementById('detailTitle').textContent=selected==='global'?'Global stages detail':`Layer ${selected} detail`;const rawStart=Math.min(...items.map(s=>s.start)),rawEnd=Math.max(...items.map(s=>s.end)),padding=Math.max(1,(rawEnd-rawStart)*.04),start=Math.max(g.start,rawStart-padding),end=Math.min(g.end,rawEnd+padding),view={...g,start,end,span:end-start,graph_execute_us:g.graph_execute_us*(end-start)/g.span};let out=axis(view),table='';for(const s of [...items].sort((a,b)=>a.order-b.order)){const p=position(view,s.start,s.end),f=family(s.stage),occ=Math.max(2,Math.min(96,100*s.occupancy));let q='';for(const c of s.quant_categories){const qp=position(view,c.start,c.end);q+=`<i class="quant-seg" style="left:${qp.left}%;width:${qp.width}%"></i>`}out+=`<div class="detail-row"><div class="detail-label"><b>${s.label}</b><br><small>${s.resources}</small></div><div class="detail-track"><div class="detail-bar family-${f}" style="left:${p.left}%;width:${p.width}%" data-tip="${htmlEscape(stageTip(s))}"><i class="active-indicator" style="width:${occ}%"></i></div>${q}</div><div class="metric">${fmt(s.critical_us,1)} μs</div></div>`;const qt=s.quant_categories.map(c=>`<b>${c.label}</b><br>${c.logical_inputs} → ${c.logical_outputs}<br>${c.physical_inputs} → ${c.physical_outputs}`).join('<hr>')||'—';table+=`<tr><td>${s.label}</td><td>${fmt(s.wall)} cyc</td><td>${fmt(s.active)} (${fmt(100*s.occupancy,1)}%)</td><td>${fmt(s.work)}</td><td>${fmt(s.dominant)} cyc<br>${fmt(s.critical_us,2)} μs</td><td class="qcell">${fmt(s.quant_active)}</td><td class="qcell">${fmt(s.quant_work)}</td><td class="qcell">${fmt(s.quant_dominant)}</td><td style="text-align:left;white-space:normal;min-width:320px">${qt}</td></tr>`}document.getElementById('detail').innerHTML=out;document.getElementById('detailTable').innerHTML=table;bindTips(document.getElementById('detail'))}
function render(){const g=DATA[graph];renderCards(g);renderOverview(g);renderDetail(g)}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{graph=b.dataset.graph;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));render()});const sel=document.getElementById('layerSelect');sel.innerHTML='<option value="global">Global</option>'+Array.from({length:28},(_,i)=>`<option value="${i}"${i===18?' selected':''}>Layer ${i}</option>`).join('');sel.onchange=()=>{selected=sel.value;renderDetail(DATA[graph])};renderLegend();render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {graph: build_graph(args.results_dir, graph) for graph in ("s1", "s32")}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    for graph, graph_data in data.items():
        print(
            f"{graph}: stages={len(graph_data['stages'])} globals={len(graph_data['globals'])} "
            f"span={graph_data['span']} graph_execute_us={graph_data['graph_execute_us']:.0f} "
            f"quant_work={graph_data['quant_work']} quant_dominant={graph_data['quant_dominant']}"
        )
    print(f"generated: {args.output}")


if __name__ == "__main__":
    main()
