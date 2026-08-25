#!/usr/bin/env python3
"""Generate the isolated EXP-0017 QDQ GroupQueryAttention probe graph."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


BATCH = 1
SEQUENCE = 32
PAST_CONTEXT = 992
CONTEXT = PAST_CONTEXT + SEQUENCE
QUERY_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


def scalar(name: str, value, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def qdq_input(nodes, initializers, name, shape, scale, zero_point):
    scale_name = f"{name}_scale"
    zero_name = f"{name}_zero_point"
    quantized = f"{name}_u8"
    initializers.extend(
        [scalar(scale_name, scale, np.float32), scalar(zero_name, zero_point, np.uint8)]
    )
    nodes.append(
        helper.make_node(
            "QuantizeLinear", [name, scale_name, zero_name], [quantized], name=f"{name}_quantize"
        )
    )
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape), quantized


def dq_output(nodes, initializers, name, shape, scale, zero_point):
    quantized = f"{name}_u8"
    scale_name = f"{name}_scale"
    zero_name = f"{name}_zero_point"
    initializers.extend(
        [scalar(scale_name, scale, np.float32), scalar(zero_name, zero_point, np.uint8)]
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear", [quantized, scale_name, zero_name], [name], name=f"{name}_dequantize"
        )
    )
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape), quantized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an isolated EXP-0017 U8 GroupQueryAttention probe graph."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cache-mode",
        choices=("growing", "static", "none"),
        default="growing",
        help=(
            "growing: 992-token past cache -> 1024-token present cache; "
            "static: fixed 1024-token cache capacity; none: initial S=32 prefill"
        ),
    )
    parser.add_argument(
        "--qkv-mode",
        choices=("separate", "packed"),
        default="separate",
        help="Use separate Q/K/V tensors or the documented packed-QKV query input.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=float(1.0 / np.sqrt(HEAD_DIM)),
        help="GQA scale parameter; 0 asks the backend to use 1/sqrt(head_size).",
    )
    parser.add_argument(
        "--do-rotary",
        action="store_true",
        help="Provide the complete cos/sin/position input signature and enable fused RoPE.",
    )
    args = parser.parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = []
    initializers = []
    graph_inputs = []

    query_width = (
        (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM
        if args.qkv_mode == "packed"
        else QUERY_HEADS * HEAD_DIM
    )
    query, query_u8 = qdq_input(
        nodes,
        initializers,
        "query",
        [BATCH, SEQUENCE, query_width],
        0.32302287220954895,
        122,
    )
    graph_inputs.append(query)
    key_u8 = ""
    value_u8 = ""
    if args.qkv_mode == "separate":
        key, key_u8 = qdq_input(
            nodes,
            initializers,
            "key",
            [BATCH, SEQUENCE, KV_HEADS * HEAD_DIM],
            0.33071592450141907,
            128,
        )
        value, value_u8 = qdq_input(
            nodes,
            initializers,
            "value",
            [BATCH, SEQUENCE, KV_HEADS * HEAD_DIM],
            1.4026927947998047,
            128,
        )
        graph_inputs.extend([key, value])
    past_key_u8 = ""
    past_value_u8 = ""
    present_length = SEQUENCE
    if args.cache_mode != "none":
        past_length = CONTEXT if args.cache_mode == "static" else PAST_CONTEXT
        present_length = CONTEXT
        past_key, past_key_u8 = qdq_input(
            nodes,
            initializers,
            "past_key",
            [BATCH, KV_HEADS, past_length, HEAD_DIM],
            0.33071592450141907,
            128,
        )
        past_value, past_value_u8 = qdq_input(
            nodes,
            initializers,
            "past_value",
            [BATCH, KV_HEADS, past_length, HEAD_DIM],
            1.4026927947998047,
            128,
        )
        graph_inputs.extend([past_key, past_value])
    graph_inputs.extend(
        [
            helper.make_tensor_value_info("seqlens_k_minus_one", TensorProto.INT32, [BATCH]),
            helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
        ]
    )
    cos_cache_u8 = ""
    sin_cache_u8 = ""
    position_ids = ""
    if args.do_rotary:
        cos_cache, cos_cache_u8 = qdq_input(
            nodes,
            initializers,
            "cos_cache",
            [CONTEXT, HEAD_DIM // 2],
            2.0 / 255.0,
            128,
        )
        sin_cache, sin_cache_u8 = qdq_input(
            nodes,
            initializers,
            "sin_cache",
            [CONTEXT, HEAD_DIM // 2],
            2.0 / 255.0,
            128,
        )
        position_ids = "position_ids"
        graph_inputs.extend(
            [
                cos_cache,
                sin_cache,
                helper.make_tensor_value_info(
                    position_ids, TensorProto.INT64, [BATCH, SEQUENCE]
                ),
            ]
        )

    attention, attention_u8 = dq_output(
        nodes,
        initializers,
        "attention_output",
        [BATCH, SEQUENCE, QUERY_HEADS * HEAD_DIM],
        0.5333649516105652,
        229,
    )
    graph_outputs = [attention]
    gqa_outputs = [attention_u8]
    if args.cache_mode != "none":
        present_key, present_key_u8 = dq_output(
            nodes,
            initializers,
            "present_key",
            [BATCH, KV_HEADS, present_length, HEAD_DIM],
            0.33071592450141907,
            128,
        )
        present_value, present_value_u8 = dq_output(
            nodes,
            initializers,
            "present_value",
            [BATCH, KV_HEADS, present_length, HEAD_DIM],
            1.4026927947998047,
            128,
        )
        graph_outputs.extend([present_key, present_value])
        gqa_outputs.extend([present_key_u8, present_value_u8])
    gqa = helper.make_node(
        "GroupQueryAttention",
        [
            query_u8,
            key_u8,
            value_u8,
            past_key_u8,
            past_value_u8,
            "seqlens_k_minus_one",
            "total_sequence_length",
            cos_cache_u8,
            sin_cache_u8,
            position_ids,
        ],
        gqa_outputs,
        name="exp0017_official_converter_gqa",
        domain="com.microsoft",
        num_heads=QUERY_HEADS,
        kv_num_heads=KV_HEADS,
        do_rotary=int(args.do_rotary),
        scale=args.scale,
    )
    # Keep QDQ at the graph boundaries while placing the GQA before output DQ nodes.
    input_quantize_count = 1 if args.qkv_mode == "packed" else 3
    if args.cache_mode != "none":
        input_quantize_count += 2
    if args.do_rotary:
        input_quantize_count += 2
    nodes.insert(input_quantize_count, gqa)

    graph = helper.make_graph(
        nodes,
        "exp0017_official_converter_gqa",
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="mllm-exp0017",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.save(model, output_path)
    print(
        f"{output_path} cache_mode={args.cache_mode} qkv_mode={args.qkv_mode} "
        f"scale={args.scale} do_rotary={int(args.do_rotary)} "
        f"present_length={present_length}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
