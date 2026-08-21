#!/usr/bin/env python3
"""Build one strict W4G32 LPBQ A16-versus-A8 projection artifact.

The accepted native-U8 RMSNorm model supplies the common static W4 carrier,
G32 block scales, and per-channel scales.  A16 and A8 activation qparams are
copied from the archived formal models into separate diagnostic namespaces.
The script also audits whether the archived models' static LPBQ payloads are
byte-identical and creates paired fixtures from the same real-valued inputs.
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
GROUP = 32
PROJECTIONS = {
    "gate_proj": (2048, 6144),
    "up_proj": (2048, 6144),
    "down_proj": (6144, 2048),
    "lm_head": (2048, 151936),
}
QPARAMS = {
    "gate_proj": (
        "model.layers.14.mlp.up_proj_input_qdq.fake_quant",
        "model.layers.14.mlp.gate_proj_output_qdq.fake_quant",
    ),
    "up_proj": (
        "model.layers.14.mlp.up_proj_input_qdq.fake_quant",
        "model.layers.14.mlp.up_proj_output_qdq.fake_quant",
    ),
    "down_proj": (
        "model.layers.14.mlp.down_proj_input_qdq.fake_quant",
        "model.layers.14.add_1_lhs_input_qdq.fake_quant",
    ),
    "lm_head": (
        "lm_head_input_qdq.fake_quant",
        "lm_head_output_qdq.fake_quant",
    ),
}


def _cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def _sha256_bytes(data: np.ndarray) -> str:
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
            raise ValueError(f"truncated mllm V2 header: {path}")
        magic, version, _model, count, desc_offset = HEADER.unpack(raw)
        if (magic, version, desc_offset) != (MAGIC, VERSION, HEADER.size):
            raise ValueError(f"unexpected mllm V2 header: {(magic, version, desc_offset)}")
        for expected_id in range(count):
            fields = PARAM.unpack(stream.read(PARAM.size))
            param_id, dtype_id, size, offset, rank = fields[:5]
            name = _cstring(fields[21])
            if param_id != expected_id or name in result or rank > 16:
                raise ValueError(f"invalid descriptor {expected_id}: {param_id}, {name!r}, {rank}")
            result[name] = Descriptor(dtype_id, size, offset, tuple(fields[5:5 + rank]), name)
    return result


def _dtype(dtype_id: int) -> np.dtype:
    return {
        DTYPE_FLOAT32: np.dtype("<f4"),
        DTYPE_INT8: np.dtype("i1"),
        DTYPE_INT32: np.dtype("<i4"),
        DTYPE_UINT8: np.dtype("u1"),
    }[dtype_id]


def tensor_view(path: Path, desc: Descriptor) -> np.memmap:
    dtype = _dtype(desc.dtype_id)
    expected = int(np.prod(desc.shape, dtype=np.int64)) * dtype.itemsize
    if expected != desc.size:
        raise ValueError(f"payload size mismatch for {desc.name}: {desc.size} != {expected}")
    return np.memmap(path, mode="r", dtype=dtype, offset=desc.offset, shape=desc.shape, order="C")


def _tensor(path: Path, descriptors: dict[str, Descriptor], source_name: str,
            target_name: str | None = None) -> TensorPayload:
    desc = descriptors[source_name]
    return TensorPayload(target_name or source_name, desc.dtype_id, desc.shape, tensor_view(path, desc))


def write_model(path: Path, tensors: list[TensorPayload]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    model_name = b"Qwen3 strict LPBQ A16 versus A8 projection diagnostic".ljust(512, b"\0")
    data_offset = HEADER.size + len(tensors) * PARAM.size
    with temp.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, model_name, len(tensors), HEADER.size))
        offset = data_offset
        for index, tensor in enumerate(tensors):
            data = np.ascontiguousarray(tensor.data)
            expected = int(np.prod(tensor.shape, dtype=np.int64)) * _dtype(tensor.dtype_id).itemsize
            if data.nbytes != expected:
                raise ValueError(f"size mismatch for {tensor.name}: {data.nbytes} != {expected}")
            raw_name = tensor.name.encode("utf-8")[:255].ljust(256, b"\0")
            shape = tensor.shape + (0,) * (16 - len(tensor.shape))
            stream.write(PARAM.pack(index, tensor.dtype_id, data.nbytes, offset,
                                    len(tensor.shape), *shape, raw_name))
            offset += data.nbytes
        for tensor in tensors:
            stream.write(memoryview(np.ascontiguousarray(tensor.data)).cast("B"))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _weight_source_prefix(projection: str) -> str:
    return "lm_head" if projection == "lm_head" else f"model.layers.14.mlp.{projection}"


def _weight_target_prefix(projection: str) -> str:
    return "model.lm_head" if projection == "lm_head" else f"model.layers.14.mlp.{projection}"


def _tensor_signature(path: Path, desc: Descriptor) -> dict[str, object]:
    view = tensor_view(path, desc)
    return {
        "dtype_id": desc.dtype_id,
        "shape": list(desc.shape),
        "bytes": desc.size,
        "sha256": _sha256_bytes(view),
    }


def _qparam(path: Path, descriptors: dict[str, Descriptor], prefix: str,
            bits: int) -> tuple[float, int]:
    scale = float(np.asarray(tensor_view(path, descriptors[prefix + ".scale"])).reshape(-1)[0])
    zero_point = int(np.asarray(tensor_view(path, descriptors[prefix + ".zero_point"])).reshape(-1)[0])
    if not math.isfinite(scale) or scale <= 0 or not 0 <= zero_point <= (1 << bits) - 1:
        raise ValueError(f"invalid A{bits} qparam for {prefix}: {scale}, {zero_point}")
    return scale, zero_point


def _quantize(real: np.ndarray, qparam: tuple[float, int], bits: int) -> np.ndarray:
    scale, zero_point = qparam
    dtype = np.uint16 if bits == 16 else np.uint8
    return np.rint(real / scale + zero_point).clip(0, (1 << bits) - 1).astype(dtype)


def _dequantize(code: np.ndarray, qparam: tuple[float, int]) -> np.ndarray:
    scale, zero_point = qparam
    return (code.astype(np.float32) - zero_point) * scale


def _effective_weight_rows(source: Path, descriptors: dict[str, Descriptor], projection: str,
                           output_indices: np.ndarray) -> np.ndarray:
    prefix = _weight_source_prefix(projection)
    weight_desc = descriptors[prefix + ".weight"]
    if weight_desc.dtype_id != DTYPE_INT8 or len(weight_desc.shape) != 4 or weight_desc.shape[:2] != (1, 1):
        raise ValueError(f"unexpected LPBQ carrier {prefix}: {weight_desc.dtype_id}, {weight_desc.shape}")
    k, o = weight_desc.shape[2:]
    if (k, o) != PROJECTIONS[projection] or k % GROUP:
        raise ValueError(f"unexpected shape for {projection}: {(k, o)}")
    carrier = tensor_view(source, weight_desc).reshape(k, o)
    codes = np.ascontiguousarray(carrier[:, output_indices].T).astype(np.int16)
    signed = np.where(codes >= 8, codes - 16, codes)
    if int(signed.min()) < -7 or int(signed.max()) > 7:
        raise ValueError(f"invalid deployed W4 code range for {projection}: {signed.min()}, {signed.max()}")
    scale1 = np.asarray(
        tensor_view(source, descriptors[prefix + ".scale1"]), dtype=np.float32
    ).reshape(o, k // GROUP)[output_indices]
    scale2 = np.asarray(
        tensor_view(source, descriptors[prefix + ".scale2"]), dtype=np.float32
    ).reshape(o)[output_indices]
    block_scale = scale1 * scale2[:, None]
    return signed.reshape(len(output_indices), k // GROUP, GROUP).astype(np.float32) \
        * block_scale[:, :, None]


def _paired_real_fixture(a16_qparam: tuple[float, int], a8_qparam: tuple[float, int],
                         seq: int, channels: int, seed: int) -> np.ndarray:
    ranges = []
    for bits, (scale, zero_point) in ((16, a16_qparam), (8, a8_qparam)):
        ranges.append((-zero_point * scale, (((1 << bits) - 1) - zero_point) * scale))
    lower = max(item[0] for item in ranges)
    upper = min(item[1] for item in ranges)
    if not lower < 0 < upper:
        raise ValueError(f"activation ranges do not overlap around zero: {ranges}")
    # Stay away from saturation while exercising both signs and broad magnitudes.
    lower *= 0.80
    upper *= 0.80
    rng = np.random.default_rng(seed)
    real = rng.uniform(lower, upper, size=(seq, channels)).astype(np.float32)
    flat = real.reshape(-1)
    anchors = np.asarray([0.0, lower, upper, lower / 2.0, upper / 2.0], dtype=np.float32)
    flat[: len(anchors)] = anchors
    return real


def _sample_reference(common_source: Path, common_desc: dict[str, Descriptor], projection: str,
                      real: np.ndarray, input_qparam: tuple[float, int],
                      output_qparam: tuple[float, int], bits: int) -> dict[str, object]:
    _k, output_channels = PROJECTIONS[projection]
    output_indices = np.unique(np.linspace(0, output_channels - 1, num=64, dtype=np.int64))
    rows = np.arange(min(real.shape[0], 4), dtype=np.int64)
    input_codes = _quantize(real, input_qparam, bits)
    reconstructed = _dequantize(input_codes, input_qparam)
    weights = _effective_weight_rows(common_source, common_desc, projection, output_indices).reshape(
        len(output_indices), -1
    )
    values = reconstructed[rows] @ weights.T
    output_codes = _quantize(values, output_qparam, bits)
    return {
        "rows": rows.tolist(),
        "output_indices": output_indices.tolist(),
        "expected_codes": output_codes.astype(np.int64).tolist(),
        "expected_real": values.astype(np.float64).tolist(),
    }


def build(a16_source: Path, a8_source: Path, output_dir: Path) -> dict[str, object]:
    a16_desc = read_descriptors(a16_source)
    a8_desc = read_descriptors(a8_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[TensorPayload] = []
    weights: dict[str, object] = {}
    for projection in PROJECTIONS:
        source_prefix = _weight_source_prefix(projection)
        target_prefix = _weight_target_prefix(projection)
        comparison: dict[str, object] = {}
        all_equal = True
        for suffix in (".weight", ".scale1", ".scale2"):
            name = source_prefix + suffix
            if name not in a16_desc or name not in a8_desc:
                raise KeyError(f"missing common LPBQ tensor: {name}")
            a16_sig = _tensor_signature(a16_source, a16_desc[name])
            a8_sig = _tensor_signature(a8_source, a8_desc[name])
            equal = a16_sig == a8_sig
            all_equal &= equal
            comparison[suffix[1:]] = {"a16": a16_sig, "a8": a8_sig, "byte_identical": equal}
            tensors.append(_tensor(a8_source, a8_desc, name, target_prefix + suffix))
        weights[projection] = {"archived_models_byte_identical": all_equal, "tensors": comparison}

    qparams: dict[str, object] = {}
    qparam_payloads: list[TensorPayload] = []
    for projection, (input_prefix, output_prefix) in QPARAMS.items():
        qparams[projection] = {}
        for activation, source, descriptors, bits in (
            ("a16", a16_source, a16_desc, 16),
            ("a8", a8_source, a8_desc, 8),
        ):
            qparams[projection][activation] = {}
            for side, source_prefix in (("input", input_prefix), ("output", output_prefix)):
                target_prefix = f"diagnostic.{activation}.{projection}.{side}"
                value = _qparam(source, descriptors, source_prefix, bits)
                qparams[projection][activation][side] = {"scale": value[0], "zero_point": value[1]}
                for suffix in (".scale", ".zero_point"):
                    qparam_payloads.append(_tensor(source, descriptors, source_prefix + suffix,
                                                   target_prefix + suffix))

    model_path = output_dir / "qwen3-lpbq-a16-a8-projections.mllm"
    write_model(model_path, tensors + qparam_payloads)

    fixtures: dict[str, object] = {}
    host_reference: dict[str, object] = {}
    for projection, (channels, _output_channels) in PROJECTIONS.items():
        fixtures[projection] = {}
        host_reference[projection] = {}
        for seq in (1, 32):
            a16_in = (qparams[projection]["a16"]["input"]["scale"],
                      qparams[projection]["a16"]["input"]["zero_point"])
            a8_in = (qparams[projection]["a8"]["input"]["scale"],
                     qparams[projection]["a8"]["input"]["zero_point"])
            real = _paired_real_fixture(a16_in, a8_in, seq, channels,
                                        20260821 + seq + list(PROJECTIONS).index(projection) * 100)
            real_path = output_dir / f"input_{projection}_s{seq}.f32.raw"
            real_path.write_bytes(real.astype("<f4").tobytes(order="C"))
            entry: dict[str, object] = {
                "real": {"path": os.fspath(real_path.resolve()), "sha256": _sha256_bytes(real)},
                "variants": {},
            }
            reconstructed: dict[str, np.ndarray] = {}
            for activation, bits, qparam in (("a16", 16, a16_in), ("a8", 8, a8_in)):
                codes = _quantize(real, qparam, bits)
                reconstructed[activation] = _dequantize(codes, qparam)
                raw_path = output_dir / f"input_{projection}_s{seq}_{activation}.raw"
                raw_path.write_bytes(codes.tobytes(order="C"))
                error = reconstructed[activation] - real
                entry["variants"][activation] = {
                    "path": os.fspath(raw_path.resolve()),
                    "bytes": raw_path.stat().st_size,
                    "sha256": _sha256_bytes(codes),
                    "saturation_count": int(np.count_nonzero((codes == 0) | (codes == (1 << bits) - 1))),
                    "reconstruction_max_abs_error": float(np.max(np.abs(error))),
                    "reconstruction_mean_abs_error": float(np.mean(np.abs(error))),
                }
                out = qparams[projection][activation]["output"]
                host_reference[projection][f"s{seq}_{activation}"] = _sample_reference(
                    a8_source, a8_desc, projection, real, qparam,
                    (out["scale"], out["zero_point"]), bits
                )
            cross_delta = reconstructed["a8"] - reconstructed["a16"]
            entry["reconstructed_a8_vs_a16"] = {
                "max_abs_delta": float(np.max(np.abs(cross_delta))),
                "mean_abs_delta": float(np.mean(np.abs(cross_delta))),
            }
            fixtures[projection][f"s{seq}"] = entry

    return {
        "contract": {
            "common_static_weight_source": os.fspath(a8_source.resolve()),
            "physical_expression": "QNN qti.aisw::Conv2d, NHWC activation, HWIO W4G32 LPBQ weight",
            "only_runtime_variable": "asymmetric activation input/output dtype and formal qparams: U16 versus U8",
            "projections": PROJECTIONS,
            "sequences": [1, 32],
        },
        "sources": {
            "a16": {"path": os.fspath(a16_source.resolve()), "sha256": _sha256_file(a16_source)},
            "a8": {"path": os.fspath(a8_source.resolve()), "sha256": _sha256_file(a8_source)},
        },
        "artifact": {"path": os.fspath(model_path.resolve()), "bytes": model_path.stat().st_size,
                     "sha256": _sha256_file(model_path)},
        "weights": weights,
        "qparams": qparams,
        "fixtures": fixtures,
        "host_reference": host_reference,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a16_source", type=Path)
    parser.add_argument("a8_source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = build(args.a16_source, args.a8_source, args.output_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    report_path = args.report or args.output_dir / "artifact_report.json"
    report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
