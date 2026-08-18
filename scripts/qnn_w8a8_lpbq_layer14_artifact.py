#!/usr/bin/env python3
"""Build compact layer-14 MLP artifacts for W4G32 LPBQ versus W8A8.

The accepted native-U8 RMSNorm artifact is the only source.  LPBQ tensors are
copied byte-for-byte.  W8 weights are deterministically requantized from the
*deployed LPBQ effective weights*, using one symmetric scale per projection
and a native signed Int8 carrier with zero offset.  Activation qparams remain
byte-identical between variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = 0x519A
VERSION = 2
HEADER = struct.Struct("<II512sIQ")
PARAM = struct.Struct("<IIQQQ16i256s")
DTYPE_FLOAT32 = 0
DTYPE_INT8 = 16
DTYPE_INT32 = 18
DTYPE_UINT8 = 129
DTYPE_INT8_PER_TENSOR_SYM = 140
LAYER = 14
GROUP = 32
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
QPARAM_PREFIXES = (
    "model.layers.14.mlp.up_proj_input_qdq.fake_quant",
    "model.layers.14.mlp.up_proj_output_qdq.fake_quant",
    "model.layers.14.mlp.gate_proj_output_qdq.fake_quant",
    "model.layers.14.mlp.sigmoid_output_qdq.fake_quant",
    "model.layers.14.mlp.act_output_qdq.fake_quant",
    "model.layers.14.mlp.down_proj_input_qdq.fake_quant",
    "model.layers.14.add_1_lhs_input_qdq.fake_quant",
)


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def _sha256(data: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(data)).cast("B")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Descriptor:
    dtype_id: int
    size: int
    offset: int
    shape: tuple[int, ...]
    name: str


@dataclass
class TensorPayload:
    name: str
    dtype_id: int
    shape: tuple[int, ...]
    data: np.ndarray


def read_descriptors(path: Path) -> dict[str, Descriptor]:
    result: dict[str, Descriptor] = {}
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("truncated mllm V2 header")
        magic, version, _model, count, desc_offset = HEADER.unpack(raw)
        if (magic, version, desc_offset) != (MAGIC, VERSION, HEADER.size):
            raise ValueError(f"unexpected mllm V2 header: {(magic, version, desc_offset)}")
        for expected_id in range(count):
            fields = PARAM.unpack(stream.read(PARAM.size))
            param_id, dtype_id, size, offset, rank = fields[:5]
            name = _cstring(fields[21])
            if param_id != expected_id or name in result or rank > 16:
                raise ValueError(f"invalid descriptor {expected_id}: id={param_id}, name={name!r}, rank={rank}")
            result[name] = Descriptor(dtype_id, size, offset, tuple(fields[5:5 + rank]), name)
    return result


def _dtype(dtype_id: int) -> np.dtype:
    return {
        DTYPE_FLOAT32: np.dtype("<f4"),
        DTYPE_INT8: np.dtype("i1"),
        DTYPE_INT32: np.dtype("<i4"),
        DTYPE_UINT8: np.dtype("u1"),
        DTYPE_INT8_PER_TENSOR_SYM: np.dtype("i1"),
    }[dtype_id]


def tensor_view(path: Path, desc: Descriptor) -> np.memmap:
    dtype = _dtype(desc.dtype_id)
    expected = int(np.prod(desc.shape, dtype=np.int64)) * dtype.itemsize
    if expected != desc.size:
        raise ValueError(f"payload size mismatch for {desc.name}: {desc.size} != {expected}")
    return np.memmap(path, mode="r", dtype=dtype, offset=desc.offset, shape=desc.shape, order="C")


def _source_tensor(source: Path, descriptors: dict[str, Descriptor], name: str) -> TensorPayload:
    desc = descriptors[name]
    return TensorPayload(name, desc.dtype_id, desc.shape, tensor_view(source, desc))


def write_model(path: Path, tensors: list[TensorPayload], model_name: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    name_bytes = model_name.encode("utf-8")[:511].ljust(512, b"\0")
    data_offset = HEADER.size + len(tensors) * PARAM.size
    with temp.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, name_bytes, len(tensors), HEADER.size))
        offset = data_offset
        for index, tensor in enumerate(tensors):
            data = np.ascontiguousarray(tensor.data)
            expected = int(np.prod(tensor.shape, dtype=np.int64)) * _dtype(tensor.dtype_id).itemsize
            if data.nbytes != expected:
                raise ValueError(f"size mismatch for {tensor.name}: {data.nbytes} != {expected}")
            shape = tensor.shape + (0,) * (16 - len(tensor.shape))
            raw_name = tensor.name.encode("utf-8")[:255].ljust(256, b"\0")
            stream.write(PARAM.pack(index, tensor.dtype_id, data.nbytes, offset, len(tensor.shape), *shape, raw_name))
            offset += data.nbytes
        for tensor in tensors:
            stream.write(memoryview(np.ascontiguousarray(tensor.data)).cast("B"))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _projection_prefix(projection: str) -> str:
    return f"model.layers.{LAYER}.mlp.{projection}"


def _lpbq_effective(source: Path, descriptors: dict[str, Descriptor], projection: str) -> np.ndarray:
    prefix = _projection_prefix(projection)
    weight_desc = descriptors[prefix + ".weight"]
    if weight_desc.dtype_id != DTYPE_INT8 or len(weight_desc.shape) != 4 or weight_desc.shape[:2] != (1, 1):
        raise ValueError(f"unexpected LPBQ carrier for {prefix}: dtype={weight_desc.dtype_id}, shape={weight_desc.shape}")
    k, o = weight_desc.shape[2:]
    if k % GROUP:
        raise ValueError(f"K={k} is not divisible by G{GROUP} for {prefix}")
    codes_io = np.asarray(tensor_view(source, weight_desc)).reshape(k, o)
    codes_oi = np.ascontiguousarray(codes_io.T).astype(np.int16)
    signed = np.where(codes_oi >= 8, codes_oi - 16, codes_oi)
    if int(signed.min()) < -7 or int(signed.max()) > 7:
        raise ValueError(f"invalid deployed W4 codes for {prefix}: [{signed.min()}, {signed.max()}]")
    scale1 = np.asarray(tensor_view(source, descriptors[prefix + ".scale1"]), dtype=np.float32).reshape(o, k // GROUP)
    scale2 = np.asarray(tensor_view(source, descriptors[prefix + ".scale2"]), dtype=np.float32).reshape(o)
    block_scale = scale1 * scale2[:, None]
    return signed.reshape(o, k // GROUP, GROUP).astype(np.float32) * block_scale[:, :, None]


def _w8_from_lpbq(effective_grouped: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    o, groups, group = effective_grouped.shape
    effective = effective_grouped.reshape(o, groups * group)
    max_abs = float(np.max(np.abs(effective)))
    if not math.isfinite(max_abs) or max_abs <= 0:
        raise ValueError(f"invalid effective-weight maximum: {max_abs}")
    scale_value = np.float32(max_abs / 127.0)
    signed = np.rint(effective / scale_value).clip(-127, 127).astype(np.int16)
    carrier_io = np.ascontiguousarray(signed.astype(np.int8).T).reshape(1, 1, effective.shape[1], o)
    reconstructed = signed.astype(np.float32) * scale_value
    error = reconstructed - effective
    denom = float(np.sum(effective.astype(np.float64) ** 2))
    stats: dict[str, float | int | str] = {
        "w8_scale": float(scale_value),
        "w8_signed_min": int(signed.min()),
        "w8_signed_max": int(signed.max()),
        "w8_weight_nmse_vs_effective_lpbq": float(np.sum(error.astype(np.float64) ** 2) / max(denom, 1e-30)),
        "w8_weight_max_abs_error_vs_effective_lpbq": float(np.max(np.abs(error))),
        "w8_weight_mean_abs_error_vs_effective_lpbq": float(np.mean(np.abs(error))),
        "w8_carrier_sha256": _sha256(carrier_io),
    }
    return carrier_io, np.asarray([scale_value], dtype=np.float32), stats


def _qparam(source: Path, descriptors: dict[str, Descriptor], prefix: str) -> tuple[float, int]:
    scale = float(np.asarray(tensor_view(source, descriptors[prefix + ".scale"])).reshape(-1)[0])
    zp = int(np.asarray(tensor_view(source, descriptors[prefix + ".zero_point"])).reshape(-1)[0])
    if not math.isfinite(scale) or scale <= 0 or not 0 <= zp <= 255:
        raise ValueError(f"invalid A8 qparam for {prefix}: scale={scale}, zero_point={zp}")
    return scale, zp


def _qdq(value: np.ndarray, qparam: tuple[float, int]) -> tuple[np.ndarray, np.ndarray]:
    scale, zp = qparam
    code = np.rint(value / scale + zp).clip(0, 255).astype(np.uint8)
    return code, (code.astype(np.float32) - zp) * scale


def _variant_weight(source: Path, descriptors: dict[str, Descriptor], projection: str, variant: str) -> np.ndarray:
    effective = _lpbq_effective(source, descriptors, projection).reshape(
        descriptors[_projection_prefix(projection) + ".weight"].shape[3], -1
    )
    if variant == "lpbq":
        return effective
    carrier, scale, _stats = _w8_from_lpbq(effective.reshape(effective.shape[0], -1, GROUP))
    return carrier.reshape(effective.shape[1], effective.shape[0]).T.astype(np.float32) * float(scale[0])


def simulate(source: Path, descriptors: dict[str, Descriptor], input_codes: np.ndarray, variant: str) -> dict[str, object]:
    input_qparam = _qparam(source, descriptors, QPARAM_PREFIXES[0])
    x = (input_codes.astype(np.float32) - input_qparam[1]) * input_qparam[0]

    gate_weight = _variant_weight(source, descriptors, "gate_proj", variant)
    gate_code, gate = _qdq(x @ gate_weight.T, _qparam(source, descriptors, QPARAM_PREFIXES[2]))
    del gate_weight
    up_weight = _variant_weight(source, descriptors, "up_proj", variant)
    up_code, up = _qdq(x @ up_weight.T, _qparam(source, descriptors, QPARAM_PREFIXES[1]))
    del up_weight
    sigmoid_code, sigmoid = _qdq(1.0 / (1.0 + np.exp(-gate)), _qparam(source, descriptors, QPARAM_PREFIXES[3]))
    act_code, activated = _qdq(gate * sigmoid, _qparam(source, descriptors, QPARAM_PREFIXES[4]))
    down_input_code, down_input = _qdq(activated * up, _qparam(source, descriptors, QPARAM_PREFIXES[5]))
    down_weight = _variant_weight(source, descriptors, "down_proj", variant)
    output_code, output = _qdq(down_input @ down_weight.T, _qparam(source, descriptors, QPARAM_PREFIXES[6]))
    return {
        "variant": variant,
        "output_code": output_code,
        "output": output,
        "intermediate_sha256": {
            "gate": _sha256(gate_code),
            "up": _sha256(up_code),
            "sigmoid": _sha256(sigmoid_code),
            "act": _sha256(act_code),
            "down_input": _sha256(down_input_code),
        },
        "finite": bool(np.all(np.isfinite(output))),
    }


def build(source: Path, output_dir: Path) -> dict[str, object]:
    descriptors = read_descriptors(source)
    missing = [name for prefix in QPARAM_PREFIXES for name in (prefix + ".scale", prefix + ".zero_point") if name not in descriptors]
    if missing:
        raise KeyError(f"source is missing required qparams: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shared = [
        _source_tensor(source, descriptors, name)
        for prefix in QPARAM_PREFIXES
        for name in (prefix + ".scale", prefix + ".zero_point")
    ]
    lpbq_tensors: list[TensorPayload] = []
    w8_tensors: list[TensorPayload] = []
    projection_stats: dict[str, object] = {}
    for projection in PROJECTIONS:
        prefix = _projection_prefix(projection)
        lpbq_tensors.extend(
            _source_tensor(source, descriptors, prefix + suffix)
            for suffix in (".weight", ".scale1", ".scale2")
        )
        effective = _lpbq_effective(source, descriptors, projection)
        carrier, scale, stats = _w8_from_lpbq(effective)
        w8_tensors.extend((
            TensorPayload(prefix + ".weight", DTYPE_INT8_PER_TENSOR_SYM, carrier.shape, carrier),
            TensorPayload(prefix + ".scale", DTYPE_FLOAT32, (1,), scale),
        ))
        projection_stats[projection] = {
            "shape_oi": [int(effective.shape[0]), int(effective.shape[1] * effective.shape[2])],
            **stats,
        }

    lpbq_path = output_dir / "qwen3-layer14-mlp-lpbq-w4a8.mllm"
    w8_path = output_dir / "qwen3-layer14-mlp-pertensor-w8a8.mllm"
    write_model(lpbq_path, lpbq_tensors + shared, "Qwen3 layer14 MLP LPBQ W4A8 diagnostic")
    write_model(w8_path, w8_tensors + shared, "Qwen3 layer14 MLP per-tensor W8A8 diagnostic")

    input_scale, input_zp = _qparam(source, descriptors, QPARAM_PREFIXES[0])
    rng = np.random.default_rng(20260818)
    fixtures: dict[str, object] = {}
    input_arrays: dict[int, np.ndarray] = {}
    for seq in (1, 32):
        codes = rng.integers(0, 256, size=(seq, 2048), dtype=np.uint8)
        # Guarantee boundary and exact-real-zero coverage independently of RNG.
        codes.reshape(-1)[:3] = np.asarray([0, input_zp, 255], dtype=np.uint8)
        fixture_path = output_dir / f"input_s{seq}.raw"
        fixture_path.write_bytes(codes.tobytes(order="C"))
        input_arrays[seq] = codes
        fixtures[f"s{seq}"] = {
            "path": os.fspath(fixture_path.resolve()),
            "bytes": fixture_path.stat().st_size,
            "sha256": _sha256(codes),
        }

    lpbq_sim = simulate(source, descriptors, input_arrays[1], "lpbq")
    w8_sim = simulate(source, descriptors, input_arrays[1], "w8a8")
    lpbq_code = lpbq_sim.pop("output_code")
    w8_code = w8_sim.pop("output_code")
    lpbq_output = lpbq_sim.pop("output")
    w8_output = w8_sim.pop("output")
    delta = w8_output - lpbq_output
    output_denom = float(np.sum(lpbq_output.astype(np.float64) ** 2))
    host_math = {
        "lpbq": lpbq_sim,
        "w8a8": w8_sim,
        "output_code_equal_fraction": float(np.mean(lpbq_code == w8_code)),
        "output_code_max_abs_delta": int(np.max(np.abs(lpbq_code.astype(np.int16) - w8_code.astype(np.int16)))),
        "output_nmse_w8_vs_lpbq": float(np.sum(delta.astype(np.float64) ** 2) / max(output_denom, 1e-30)),
        "output_max_abs_delta": float(np.max(np.abs(delta))),
    }

    return {
        "source": os.fspath(source.resolve()),
        "source_sha256": _sha256_file(source),
        "contract": {
            "layer": LAYER,
            "mlp": "gate_proj + sigmoid/gating + up_proj + down_proj",
            "activation": "asymmetric UInt8; source qparams copied byte-for-byte",
            "lpbq_weight": "signed W4 [-7,7], G32, UInt4 block scale, FP32 channel scale",
            "w8_weight": "per-tensor symmetric signed W8 via Int8PerTensorSym carrier and QNN SFIXED_POINT_8",
            "w8_source": "deterministic requantization of deployed LPBQ effective weight",
            "layout": "identical NHWC activation and HWIO Conv2D weight",
        },
        "input_qparam": {"scale": input_scale, "zero_point": input_zp},
        "artifacts": {
            "lpbq": {"path": os.fspath(lpbq_path.resolve()), "bytes": lpbq_path.stat().st_size,
                     "sha256": _sha256_file(lpbq_path)},
            "w8a8": {"path": os.fspath(w8_path.resolve()), "bytes": w8_path.stat().st_size,
                     "sha256": _sha256_file(w8_path)},
        },
        "fixtures": fixtures,
        "projection_stats": projection_stats,
        "host_math_s1": host_math,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = build(args.source, args.output_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    report_path = args.report or args.output_dir / "artifact_report.json"
    report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
