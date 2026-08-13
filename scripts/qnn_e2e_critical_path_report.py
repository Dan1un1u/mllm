#!/usr/bin/env python3
"""Generate the canonical SM8750/V79 Qwen3 E2E critical-path report.

The report deliberately retains only:
  * full-graph Qwen3 structure critical-path decomposition;
  * Layer 0 operator-envelope/critical-ownership decomposition;
  * Layer 0 and LM-head local timelines;
  * full-graph, Layer 0 and LM-head critical-resource pies;
  * measured runner-level prefill/decode throughput.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import qnn_optrace_qwen3_gantt_simple as legacy
from qnn_optrace_lm_head import build_graph as build_lm_head
from qnn_optrace_quantization import classify_quant_kernel
from qnn_optrace_qwen3_structure import STAGE_INFO, classification, infer_head_counts


FOCUS_LAYER = 0


def number(row, key):
    return int(float(row.get(key, 0) or 0))


def resource_of(row):
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


def global_structure(qhas_path):
    with qhas_path.open() as stream:
        data = json.load(stream)["data"]
    qnn_rows = data["qnn_op_instances_nodes"]["data"]
    htp_rows = data["htp_op_instances"]["data"]
    q_heads, kv_heads = infer_head_counts(qnn_rows)

    direct = defaultdict(int)
    resources = defaultdict(lambda: defaultdict(int))
    quant = defaultdict(int)
    for row in qnn_rows:
        stage, _, _, _ = classification(
            row["qnn_op"], row.get("qnn_op_type", ""), q_heads, kv_heads
        )
        direct[stage] += number(row, "num_dominant_path_cycles_htp_0")
    for row in htp_rows:
        value = number(row, "num_dominant_path_cycles")
        if not value:
            continue
        stage, _, _, _ = classification(
            row["qnn_op"], row.get("qnn_op_type", ""), q_heads, kv_heads
        )
        resources[stage][resource_of(row)] += value
        flags = tuple(
            name for field, name in (
                ("hmx", "uses_hmx"), ("hvx", "uses_hvx"), ("dma", "dma"),
                ("dma_wait", "dma_wait"), ("dma_set", "dma_set"), ("sync", "sync"),
            ) if row.get(field)
        )
        category = classify_quant_kernel(
            row.get("htp_op", ""), row.get("qnn_op_type", ""), flags
        )
        if category in legacy.CONVERSION_CATEGORIES:
            quant[stage] += value

    rows = []
    for stage, value in direct.items():
        info = STAGE_INFO[stage]
        resource_total = sum(resources[stage].values())
        tolerance = max(32, int(value * 0.0001))
        if abs(resource_total - value) > tolerance:
            raise ValueError(
                f"{qhas_path.name}: stage {stage} resource/direct mismatch: "
                f"{resource_total} vs {value}"
            )
        rows.append({
            "stage": stage,
            "label": info["label"],
            "order": info["order"],
            "direct": value,
            "resources": dict(resources[stage]),
            "quant": min(resources[stage].get("hvx", 0), quant[stage]),
        })
    return sorted(rows, key=lambda row: row["order"])


def build_graph(results_dir, graph):
    prefix = results_dir / f"qwen3-sm8750-v79-{graph}"
    qhas_path = Path(f"{prefix}-chrometrace_qnn_htp_analysis_summary.json")
    legacy.LAYERS = (FOCUS_LAYER,)
    runtime = legacy.cook_runtime_tracks(results_dir, graph)
    closure = runtime["closure"]
    signed_delta = closure["dominant_path_cycles"] - closure["qnn_direct_cycles"]
    closure["signed_closure_delta"] = signed_delta
    if signed_delta < 0:
        raise ValueError(
            f"{graph}: QNN direct contribution exceeds official dominant path by {-signed_delta} cycles"
        )
    resource_total = sum(closure["resource_totals"].values())
    if resource_total != closure["qnn_direct_cycles"]:
        raise ValueError(
            f"{graph}: full resource/direct mismatch: "
            f"{resource_total} vs {closure['qnn_direct_cycles']}"
        )
    for row in runtime["focus"][str(FOCUS_LAYER)]:
        row_resource_total = sum(row["owner_by_resource"].values())
        tolerance = max(16, int(row["owner_direct"] * 0.0001))
        if abs(row_resource_total - row["owner_direct"]) > tolerance:
            raise ValueError(
                f"{graph}: Layer {FOCUS_LAYER} {row['key']} resource/direct mismatch: "
                f"{row_resource_total} vs {row['owner_direct']}"
            )
        if row["quant_dominant"] > row["owner_by_resource"].get("hvx", 0):
            raise ValueError(f"{graph}: {row['key']} quant subset exceeds HVX critical")
    lm_head = build_lm_head(qhas_path, graph)
    lm_resource_total = sum(lm_head["resources"].values())
    lm_direct = number(lm_head["qnn"], "num_dominant_path_cycles_htp_0")
    if lm_resource_total != lm_direct:
        raise ValueError(
            f"{graph}: LM-head resource/direct mismatch: {lm_resource_total} vs {lm_direct}"
        )
    return {
        "id": graph,
        "title": "single-token decode" if graph == "s1" else "32-token prefill chunk",
        "runtime": runtime,
        "global_structure": global_structure(qhas_path),
        "lm_head": lm_head,
    }


def validate_speed(speed):
    phases = speed.get("phases", {})
    for phase in ("prefill_e2e", "decode_e2e_after_first"):
        row = phases.get(phase)
        if not isinstance(row, dict):
            raise ValueError(f"speed JSON is missing phase: {phase}")
        for key in ("tokens_median", "duration_us_median", "tokens_per_second_median"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"speed JSON has invalid {phase}.{key}: {value!r}")


def validate_accuracy(accuracy):
    total = accuracy.get("total")
    passed = accuracy.get("passed")
    cases = accuracy.get("cases")
    if not isinstance(total, int) or total <= 0:
        raise ValueError(f"accuracy JSON has invalid total: {total!r}")
    if not isinstance(passed, int) or not 0 <= passed <= total:
        raise ValueError(f"accuracy JSON has invalid passed: {passed!r}")
    if not isinstance(cases, list) or len(cases) != total:
        raise ValueError("accuracy JSON cases do not match total")
    expected_percent = 100.0 * passed / total
    if abs(float(accuracy.get("accuracy_percent", -1)) - expected_percent) > 0.001:
        raise ValueError("accuracy JSON percentage does not match passed/total")


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3 × SM8750 V79 — E2E Critical-path Profiling</title><style>
:root{--ink:#19283b;--muted:#68798d;--line:#d7e0e9;--paper:#fff;--bg:#eef2f6;--hmx:#2868aa;--hvx:#2a9b80;--dma:#d98a22;--wait:#c65359;--other:#c9d1db;--quant:#f4d35e;--critical:#f7ca45;--global:#778393;--missing:#cf5b62;--pass:#2f8f62;--warning:#cc8619;--fail:#c64b55}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,"Noto Sans SC",system-ui,sans-serif}header{padding:29px 40px;background:linear-gradient(125deg,#183a62,#237789);color:#fff}h1{margin:0 0 6px;font-size:28px}header p{margin:0;color:#d9e9f3}main{max-width:1500px;margin:16px auto;padding:0 16px 42px}.panel{margin:14px 0;padding:17px;background:var(--paper);border:1px solid var(--line);border-radius:12px}.toolbar,.legend,.metrics{display:flex;flex-wrap:wrap;gap:9px;align-items:center}.tab{padding:7px 13px;border:1px solid #b9c9d8;border-radius:7px;background:#fff;color:#285174;font:inherit}.tab.active{background:#246f9e;color:#fff;border-color:#246f9e}.cards{display:grid;grid-template-columns:repeat(5,minmax(180px,1fr));gap:10px;margin-top:13px}.card{padding:12px;border:1px solid var(--line);border-radius:9px}.card small{display:block;color:var(--muted)}.card b{display:block;color:#24557f;font-size:20px}.card.acc-pass{border-color:var(--pass);background:#f2fbf6}.card.acc-warning{border-color:var(--warning);background:#fff8e8}.card.acc-fail{border-color:var(--fail);background:#fff3f4}.card.acc-pass b{color:var(--pass)}.card.acc-warning b{color:var(--warning)}.card.acc-fail b{color:var(--fail)}.note{padding:10px 13px;border-left:4px solid #e19a27;background:#fff7e9}.closure{margin-top:13px;padding:12px;border:1px solid var(--line);border-radius:9px;background:#f8fafc}.closure-bar,.stack{display:flex;overflow:hidden;border-radius:4px;box-shadow:0 0 0 1px #8292a333}.closure-bar{height:18px}.closure-labels{display:flex;flex-wrap:wrap;gap:12px;margin-top:7px;font-size:11px;color:#52667c}.acc-table{width:100%;border-collapse:collapse;margin-top:12px}.acc-table th,.acc-table td{text-align:left;padding:7px 9px;border-bottom:1px solid #e3e9ef;vertical-align:top}.acc-table th{background:#edf3f8}.acc-table code{white-space:pre-wrap;overflow-wrap:anywhere}.acc-result{font-weight:750}.acc-result.pass{color:var(--pass)}.acc-result.fail{color:var(--fail)}.sw{display:inline-block;width:23px;height:10px;margin-right:5px;border-radius:2px;vertical-align:-1px}.hmx{background:var(--hmx)}.hvx{background:var(--hvx)}.dma{background:var(--dma)}.other{background:var(--other)}.quant-demo{position:relative;background:var(--hvx)}.quant-demo:after,.quant-overlay{content:"";position:absolute;inset:0;background:repeating-linear-gradient(135deg,transparent 0 3px,var(--quant) 3px 5px)}.structure{border:1px solid var(--line);border-radius:9px;overflow:hidden;margin-top:12px}.shead,.srow{display:grid;grid-template-columns:42px 225px 112px minmax(360px,1fr) 94px 105px 150px;align-items:center}.shead{padding:7px 9px;background:#edf3f8;font-size:11px;font-weight:700;color:#4a6076}.srow{min-height:39px;padding:5px 9px;border-top:1px solid #edf1f5}.srow>span,.shead>span{padding:0 5px}.name{font-weight:650}.sub{display:block;color:var(--muted);font-size:10px}.num{text-align:right;font-variant-numeric:tabular-nums}.scale{position:relative;height:18px;background-image:linear-gradient(to right,#e8edf2 1px,transparent 1px);background-size:10% 100%}.latency{position:relative;display:flex;height:16px;min-width:2px;border-radius:3px;overflow:hidden;box-shadow:0 0 0 1px #8796a633}.seg{height:100%;position:relative}.seg-hmx{background:var(--hmx)}.seg-hvx{background:var(--hvx)}.seg-dma{background:var(--dma)}.seg-other{background:var(--other)}.timeline{min-width:1120px}.gantt-row{display:grid;grid-template-columns:250px 1fr;gap:8px;min-height:31px;border-bottom:1px solid #edf1f5}.glabel{text-align:right;padding:4px 6px;font-weight:650;color:#40576e}.track{position:relative;height:30px;background-image:linear-gradient(to right,#e8edf2 1px,transparent 1px);background-size:12.5% 100%}.axis{height:39px;border-bottom:1px solid #9dafc3}.lane{display:grid;grid-template-columns:150px 1fr;gap:8px;min-height:25px;border-bottom:1px solid #edf1f5}.lane-label{text-align:right;padding:3px 5px;font-size:12px;font-weight:650}.cells{display:grid;grid-template-columns:repeat(180,1fr);background:#f2f5f8}.cell{min-width:2px;height:17px;margin-top:3px}.cell.critical{box-shadow:inset 0 -3px 0 var(--critical)}.scroll{overflow-x:auto}.pie-grid{display:grid;grid-template-columns:repeat(3,minmax(360px,1fr));gap:12px;margin-top:13px}.pie-card{display:grid;grid-template-columns:180px 1fr;gap:11px;align-items:center;padding:13px;border:1px solid var(--line);border-radius:10px;background:#fafcfe}.pie-card h3{grid-column:1/-1;margin:0;color:#244f76}.pie-card svg{width:175px;height:175px}.pie-legend{display:grid;gap:7px;font-size:12px}.pie-legend .item{display:grid;grid-template-columns:12px 1fr auto;gap:6px;align-items:center}.dot{width:11px;height:11px;border-radius:2px}.tip{position:fixed;z-index:100;display:none;max-width:500px;padding:11px 13px;background:#10243d;color:#f6f8fb;border-radius:8px;white-space:pre-line;pointer-events:none;box-shadow:0 8px 24px #0005;font-size:12px}.metric{padding:3px 8px;border:1px solid #cad9e6;border-radius:12px;background:#f8fbfd;font-size:11px}@media(max-width:1250px){.cards{grid-template-columns:repeat(3,1fr)}}@media(max-width:1150px){.cards{grid-template-columns:1fr 1fr}.structure{overflow-x:auto}.shead,.srow{min-width:1180px}.pie-grid{grid-template-columns:1fr}.pie-card{grid-template-columns:190px 1fr}}@media(max-width:600px){.cards{grid-template-columns:1fr}.pie-card{grid-template-columns:1fr}}</style></head><body>
<header><h1>Qwen3-1.7B × SM8750 V79：E2E 关键路径 Profiling</h1><p>full graph structure · Layer 0 representative · LM-head pipeline · measured throughput · accuracy sanity</p></header><main>
<section class="panel"><div class="toolbar"><b>Graph</b><button class="tab active" data-g="s1">s1 · decode</button><button class="tab" data-g="s32">s32 · 32-token chunk</button></div><div class="cards" id="cards"></div><div class="closure" id="closure"></div></section>
<section class="panel"><h2>W4A8G32 对 W4A16G32 runner-E2E 对比</h2><p class="note">同一 profiling 工作负载的中位数对比，仅作实验信息，不设加速门槛。正值表示 W4A8 更快；两侧都必须来自 runner CSV，禁止 QHAS 反推。</p><div class="cards" id="comparison"></div></section>
<section class="panel"><h2>全图：按 Qwen3 模型结构拆分关键路径</h2><p class="note">每行是互斥的 QNN-node direct-critical contribution；28 个 Transformer layer 的同类阶段在这里聚合。因此各行可以相加并回到全图 mapped dominant path，不使用重叠的 E2E envelope。</p><div class="legend"><span><i class="sw hmx"></i>HMX critical</span><span><i class="sw hvx"></i>HVX critical</span><span><i class="sw dma"></i>DMA/wait/sync critical</span><span><i class="sw other"></i>Unresolved resource（通常 &lt;0.01%）</span><span><i class="sw quant-demo"></i>斜纹：HVX 中可见量化/转换子集</span></div><div id="global"></div></section>
<section class="panel"><h2>Layer 0：按模型内部步骤拆分</h2><p class="note">总宽度是语义算子的 E2E envelope；彩色部分是该算子互斥拥有的 critical contribution，灰色是 residual（non-own critical、off-critical overlap 与 gap 的合计）。不同算子的 envelope 会重叠，不能相加。</p><div id="layer0"></div></section>
<section class="panel"><h2>局部甘特图</h2><p class="note">只保留代表性的 Layer 0，并加入 LM head。Layer0 横向位置来自真实 event 起止；条内资源比例不代表实际先后。LM head 直接按 HTP lane 显示实际并行窗口，黄色底边表示该 bin 命中官方 dominant path。</p><h3>Layer 0</h3><div class="scroll"><div class="timeline" id="layer-gantt"></div></div><h3>LM head</h3><div class="scroll"><div class="timeline" id="lm-gantt"></div></div><div class="metrics" id="lm-metrics"></div></section>
<section class="panel"><h2>关键路径资源构成</h2><p class="note">三张饼图分别使用全图、Layer0 和 LM head 自己的互斥 direct-critical cycles。量化斜纹是 HVX 扇区子集，不额外加入分母。</p><div class="pie-grid" id="pies"></div></section>
<section class="panel"><h2>方法与边界</h2><ul><li>关键链来自官方 QHAS <code>Dominant Path HTP0</code>；schematic 将 runtime resource ID 还原到 QNN node 和 HTP kernel，再按 Qwen3 stage 唯一归属。</li><li>HMX/HVX/DMA 拆分来自 QHAS HTP instance 的资源标志；Chrome Trace 只用于局部真实时间位置、区间 union 与 overlap 检查。</li><li>吞吐由独立的 profiling-off runner pass 实测；若页面标为 derived，则仅由 graphExecute latency 反推，不是完整 E2E。</li><li>Accuracy 是 100 道确定性短答案的 profiling-off、按答案类型抽取匹配，只用于发现明显掉点或运行时损坏，不等价于 MMLU/CMMLU 等正式精度评测。</li><li>HTP trace 不覆盖 graph 外 CPU tokenization、RPC setup 等；runner E2E 吞吐覆盖 prompt/decode 主循环，但不包含进程启动和 context 加载。</li></ul></section>
</main><div class="tip" id="tip"></div><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent);let graph='s1';const S=D.speed,A=D.accuracy,C=D.speed_comparison;function fmt(n,d=2){return Number(n||0).toLocaleString(undefined,{maximumFractionDigits:d})}function pct(n,d){return 100*(n||0)/Math.max(d||0,1)}function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;')}function dma(d){return (d.dma_transfer||0)+(d.dma_wait||0)+(d.sync||0)}function approxUs(g,c){const x=g.runtime.closure;return c*x.graph_execute_us/Math.max(x.dominant_path_cycles,1)}function resParts(d){return {hmx:d.hmx||0,hvx:d.hvx||0,dma:dma(d),unresolved:d.unresolved||0}}
function phaseMeasured(x){return (x.source||S.source)==='runner_e2e_measured'}function cards(g){const c=g.runtime.closure,p=S.phases.prefill_e2e,d=S.phases.decode_e2e_after_first;document.getElementById('cards').innerHTML=`<div class="card"><small>Prefill effective throughput</small><b>${fmt(p.tokens_per_second_median,2)} token/s</b><small>${fmt(p.tokens_median)} prompt tokens · ${fmt(p.duration_us_median/1000,2)} ms · ${phaseMeasured(p)?`${p.rounds} runner-E2E rounds`:'traced QHAS derived'}</small></div><div class="card"><small>Decode throughput after first token</small><b>${fmt(d.tokens_per_second_median,2)} token/s</b><small>${fmt(d.duration_us_median/1000,2)} ms / ${fmt(d.tokens_median)} tokens · ${phaseMeasured(d)?`${d.rounds} runner-E2E rounds`:'traced QHAS derived'}</small></div><div class="card"><small>${g.title} graphExecute</small><b>${fmt(c.graph_execute_us/1000,3)} ms</b><small>${fmt(c.dominant_path_cycles)} dominant-path cycles</small></div><div class="card"><small>QNN-node direct closure</small><b>${fmt(pct(c.qnn_direct_cycles,c.dominant_path_cycles),3)}%</b><small>${fmt(c.qnn_direct_cycles)} mapped · ${fmt(c.unmapped_cycles)} system/unmapped</small></div><div class="card acc-${A.risk}"><small>Accuracy sanity · ${A.risk.toUpperCase()}</small><b>${fmt(A.accuracy_percent,2)}%</b><small>${A.passed}/${A.total} short-answer checks · profiling off</small></div>`;const o=c.ownership,items=[['Layer 0',o.layer_0||0,'var(--hmx)'],['Other 27 layers',o.other_layers||0,'#8998a9'],['Global/runtime',(o.global_runtime||0)+(o.selected_layer_other||0),'var(--dma)'],['Unmapped',c.unmapped_cycles,'var(--missing)']];document.getElementById('closure').innerHTML=`<b>Full-graph critical-path ownership closure</b><div class="closure-bar">${items.map(x=>`<i style="width:${pct(x[1],c.dominant_path_cycles)}%;background:${x[2]}"></i>`).join('')}</div><div class="closure-labels">${items.map(x=>`<span>${x[0]} ${fmt(x[1]/1e6,3)}M (${fmt(pct(x[1],c.dominant_path_cycles),3)}%)</span>`).join('')}</div>${S.warning?`<p class="sub">${S.warning}</p>`:''}`}
function comparison(){const p=C.phases.prefill_e2e,d=C.phases.decode_e2e_after_first;document.getElementById('comparison').innerHTML=`<div class="card"><small>Reference</small><b>W4A16G32</b><small>${C.reference_id}</small></div><div class="card"><small>Prefill W4A8 / W4A16</small><b>${fmt(p.ratio_candidate_over_reference,3)}×</b><small>${fmt(p.candidate_tokens_per_second_median,2)} vs ${fmt(p.reference_tokens_per_second_median,2)} token/s · ${p.percent_change>=0?'+':''}${fmt(p.percent_change,2)}%</small></div><div class="card"><small>Decode W4A8 / W4A16</small><b>${fmt(d.ratio_candidate_over_reference,3)}×</b><small>${fmt(d.candidate_tokens_per_second_median,2)} vs ${fmt(d.reference_tokens_per_second_median,2)} token/s · ${d.percent_change>=0?'+':''}${fmt(d.percent_change,2)}%</small></div>`}
function stack(parts,total,quant=0){const h=pct(parts.hmx,total),v=pct(parts.hvx,total),d=pct(parts.dma,total),u=Math.max(0,100-h-v-d),q=Math.min(100,pct(quant,Math.max(parts.hvx,1)));return `<div class="latency"><i class="seg seg-hmx" style="width:${h}%"></i><i class="seg seg-hvx" style="width:${v}%"><i class="quant-overlay" style="width:${q}%"></i></i><i class="seg seg-dma" style="width:${d}%"></i><i class="seg seg-other" style="width:${u}%"></i></div>`}
function globalRows(g){const total=g.runtime.closure.dominant_path_cycles,max=Math.max(...g.global_structure.map(x=>x.direct));document.getElementById('global').innerHTML=`<div class="structure"><div class="shead"><span>#</span><span>Qwen3 stage</span><span class="num">Own critical</span><span>Resource composition · bar width = full-graph contribution</span><span class="num">Graph DP</span><span class="num">≈ latency</span><span>Bottleneck signal</span></div>${g.global_structure.map((x,i)=>{const p=resParts(x.resources),width=pct(x.direct,max),signal=p.dma>=Math.max(p.hmx,p.hvx)?'DMA / dependency':p.hvx>p.hmx*1.2?'HVX / conversion':p.hmx>p.hvx*1.2?'HMX compute':'mixed pipeline';return `<div class="srow"><span>${i+1}</span><span class="name">${x.label}<small class="sub"><code>${x.stage}</code></small></span><span class="num">${fmt(x.direct)}</span><span class="scale"><span style="position:absolute;width:${width}%;inset-block:1px">${stack(p,x.direct,x.quant)}</span></span><span class="num">${fmt(pct(x.direct,total),3)}%</span><span class="num">${fmt(approxUs(g,x.direct),2)} μs</span><span>${signal}</span></div>`}).join('')}</div>`}
function layerParts(x){const p=resParts(x.owner_by_resource||{});return {...p,other:Math.max(0,x.wall-p.hmx-p.hvx-p.dma)}}
function tooltip(g,x,p){return `${x.label}\nE2E envelope: ${fmt(x.wall)} cycles ≈ ${fmt(approxUs(g,x.wall),2)} μs\nOwn direct critical: ${fmt(x.owner_direct)} (${fmt(pct(x.owner_direct,x.wall),1)}% envelope)\nHMX ${fmt(p.hmx)} · HVX ${fmt(p.hvx)} · DMA ${fmt(p.dma)}\nResidual non-own / overlap / gap ${fmt(p.other)}\nVisible quant/conversion critical ${fmt(x.quant_dominant)} (HVX subset)\nOff-critical overlap with own DP ${fmt(x.offcritical_own_overlap)}`}
function bind(root){root.querySelectorAll('[data-tip]').forEach(e=>{e.onmouseenter=x=>{const t=document.getElementById('tip');t.textContent=e.dataset.tip;t.style.display='block';move(x)};e.onmousemove=move;e.onmouseleave=()=>document.getElementById('tip').style.display='none'})}function move(e){const t=document.getElementById('tip');t.style.left=Math.min(innerWidth-t.offsetWidth-10,e.clientX+13)+'px';t.style.top=Math.min(innerHeight-t.offsetHeight-10,e.clientY+13)+'px'}
function layerRows(g){const rows=g.runtime.focus['0'],max=Math.max(...rows.map(x=>x.wall)),total=g.runtime.closure.dominant_path_cycles;document.getElementById('layer0').innerHTML=`<div class="structure"><div class="shead"><span>#</span><span>Layer 0 operator</span><span class="num">E2E envelope</span><span>Critical ownership inside envelope · shared scale</span><span class="num">Own / E2E</span><span class="num">Graph DP</span><span>Bottleneck signal</span></div>${rows.map((x,i)=>{const p=layerParts(x),tip=esc(tooltip(g,x,p)),width=pct(x.wall,max);return `<div class="srow"><span>${i+1}</span><span class="name">${x.label}</span><span class="num">${fmt(x.wall)}<small class="sub">≈ ${fmt(approxUs(g,x.wall),2)} μs</small></span><span class="scale"><span style="position:absolute;width:${width}%;inset-block:1px" data-tip="${tip}">${stack(p,x.wall,x.quant_dominant)}</span></span><span class="num">${fmt(pct(x.owner_direct,x.wall),1)}%</span><span class="num">${fmt(pct(x.owner_direct,total),3)}%</span><span>${x.signal}</span></div>`}).join('')}</div>`;bind(document.getElementById('layer0'))}
function axis(v,g){let x='<div class="gantt-row"><span></span><div class="track axis">';for(let i=0;i<=8;i++){const c=v.span*i/8;x+=`<i style="position:absolute;left:${i*12.5}%;bottom:0;height:8px;border-left:1px solid #9dafc3"><span class="sub" style="position:absolute;bottom:9px;transform:translateX(-50%);white-space:nowrap">${fmt(c/1e6,2)}M cyc<br>${fmt(approxUs(g,c),1)} μs</span></i>`}return x+'</div></div>'}
function layerGantt(g){const rows=g.runtime.focus['0'],raw0=Math.min(...rows.map(x=>x.start)),raw1=Math.max(...rows.map(x=>x.end)),pad=(raw1-raw0)*.025,v={start:raw0-pad,span:(raw1-raw0)*1.05};let x=axis(v,g);rows.forEach((r,i)=>{const p=layerParts(r),left=pct(r.start-v.start,v.span),width=Math.max(.18,pct(r.wall,v.span)),tip=esc(tooltip(g,r,p));x+=`<div class="gantt-row"><span class="glabel">${i+1}. ${r.label}<small class="sub">own ${fmt(r.owner_direct)}</small></span><div class="track"><span style="position:absolute;left:${left}%;width:${width}%;top:6px" data-tip="${tip}">${stack(p,r.wall,r.quant_dominant)}</span></div></div>`});document.getElementById('layer-gantt').innerHTML=x;bind(document.getElementById('layer-gantt'))}
function laneColor(l){return l.startsWith('HMX')?'var(--hmx)':l.startsWith('HVX')?'var(--hvx)':l==='DMA wait'?'var(--wait)':l==='DMA transfer'?'var(--dma)':'var(--global)'}function lmGantt(g){const lm=g.lm_head;document.getElementById('lm-gantt').innerHTML=lm.tracks.map(t=>`<div class="lane"><span class="lane-label">${t.lane}</span><div class="cells">${t.cells.map(c=>`<i class="cell${c.critical?' critical':''}" style="background:${laneColor(t.lane)};opacity:${.08+.92*Math.sqrt(c.occupancy)}"></i>`).join('')}</div></div>`).join('');const a=lm.active_by_resource,o=lm.overlaps;document.getElementById('lm-metrics').innerHTML=[`Envelope ${fmt(lm.envelope_cycles)} cyc`,`HMX active ${fmt(a.hmx||0)}`,`HVX union ${fmt(a.hvx||0)}`,`DMA active ${fmt(a.dma_transfer||0)}`,`DMA wait ${fmt(a.dma_wait||0)}`,`HMX∩HVX ${fmt(o.hmx_hvx)}`,`HVX∩DMA ${fmt(o.hvx_dma)}`,`wait∩compute ${fmt(o.wait_compute)}`].map(x=>`<span class="metric">${x}</span>`).join('')}
function piePoint(p,r=47){const a=p*Math.PI*2/100-Math.PI/2;return [50+r*Math.cos(a),50+r*Math.sin(a)]}function sector(a,b,color,extra=''){if(b<=a)return '';if(b-a>=99.999)return `<circle cx="50" cy="50" r="47" fill="${color}" ${extra}/>`;const p0=piePoint(a),p1=piePoint(b),large=b-a>50?1:0;return `<path d="M50 50 L${p0[0]} ${p0[1]} A47 47 0 ${large} 1 ${p1[0]} ${p1[1]} Z" fill="${color}" ${extra}/>`}
function pieData(g,id){if(id==='all'){const p=resParts(g.runtime.closure.resource_totals||{});return {title:'Full graph · 28 Layers + global',...p,quant:g.runtime.closure.quant_total||0}}if(id==='layer0'){const rows=g.runtime.focus['0'],sum=k=>rows.reduce((a,x)=>a+((x.owner_by_resource||{})[k]||0),0);return {title:'Layer 0',hmx:sum('hmx'),hvx:sum('hvx'),dma:sum('dma_transfer')+sum('dma_wait')+sum('sync'),unresolved:sum('unresolved'),quant:rows.reduce((a,x)=>a+x.quant_dominant,0)}}const lm=g.lm_head,r=lm.resources,expand=(lm.categories.find(x=>x.key==='weight_expand')||{direct:0}).direct,format=(lm.categories.find(x=>x.key==='output_format')||{direct:0}).direct;return {title:'LM head',hmx:r.hmx||0,hvx:r.hvx||0,dma:(r.dma_transfer||0)+(r.dma_wait||0)+(r.sync||0),unresolved:r.unresolved||0,quant:Math.min(r.hvx||0,expand+format)}}
function pie(g,id,i){const x=pieData(g,id),total=x.hmx+x.hvx+x.dma;if(total<=0)return `<article class="pie-card"><h3>${x.title}</h3><p>N/A · no classified direct-critical cycles</p></article>`;const h=pct(x.hmx,total),v=pct(x.hvx,total),d=pct(x.dma,total),q=Math.min(v,pct(x.quant,total)),pat=`q-${g.id}-${i}`;return `<article class="pie-card"><h3>${x.title}</h3><svg viewBox="0 0 100 100"><defs><pattern id="${pat}" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(35)"><line x1="0" y1="0" x2="0" y2="5" stroke="var(--quant)" stroke-width="2.2"/></pattern></defs>${sector(0,h,'var(--hmx)')}${sector(h,h+v,'var(--hvx)')}${sector(h+v,100,'var(--dma)')}${sector(h,h+q,`url(#${pat})`,'opacity=".95"')}</svg><div class="pie-legend">${[['HMX critical',x.hmx,'var(--hmx)'],['HVX critical',x.hvx,'var(--hvx)'],['DMA/wait/sync critical',x.dma,'var(--dma)']].map(y=>`<div class="item"><i class="dot" style="background:${y[2]}"></i><b>${y[0]}</b><span>${fmt(pct(y[1],total),2)}%</span></div>`).join('')}<div class="item"><i class="dot quant-demo"></i><b>Visible quant subset</b><span>${fmt(pct(x.quant,Math.max(x.hvx,1)),2)}% HVX</span></div><small>${fmt(total)} classified direct cycles${x.unresolved?` · unresolved ${fmt(x.unresolved)} excluded (${fmt(pct(x.unresolved,total+x.unresolved),4)}%)`:''}</small></div></article>`}
function pies(g){document.getElementById('pies').innerHTML=['all','layer0','lm'].map((x,i)=>pie(g,x,i)).join('')}function render(){const g=D.graphs[graph];cards(g);comparison();globalRows(g);layerRows(g);layerGantt(g);lmGantt(g);pies(g)}document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{graph=b.dataset.g;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));render()});render();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--speed-json", required=True, type=Path)
    parser.add_argument("--accuracy-json", required=True, type=Path)
    parser.add_argument("--speed-comparison-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    speed = json.loads(args.speed_json.read_text())
    accuracy = json.loads(args.accuracy_json.read_text())
    speed_comparison = json.loads(args.speed_comparison_json.read_text())
    validate_speed(speed)
    validate_accuracy(accuracy)
    graphs = {graph: build_graph(args.results_dir, graph) for graph in ("s1", "s32")}
    accuracy_public = {
        key: accuracy[key]
        for key in (
            "suite", "scope", "method", "max_new_tokens", "passed", "total",
            "accuracy_percent", "risk", "interpretation",
        )
    }
    payload = json.dumps(
        {"speed": speed, "speed_comparison": speed_comparison, "accuracy": accuracy_public, "graphs": graphs},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    for graph, value in graphs.items():
        closure = value["runtime"]["closure"]
        print(
            f"{graph}: graph_execute_us={closure['graph_execute_us']} "
            f"closure={100 * closure['qnn_direct_cycles'] / closure['dominant_path_cycles']:.4f}%"
        )
    print(f"generated: {args.output}")


if __name__ == "__main__":
    main()
