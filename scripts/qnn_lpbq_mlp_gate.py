#!/usr/bin/env python3
"""Generate LPBQ MLP fixtures and compare device outputs with exact math."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qnn_lpbq_mlp_artifact import (
    DTYPE_FLOAT32,
    DTYPE_INT8,
    DTYPE_UINT8,
    _array,
    read_descriptors,
)


PROJECTIONS = {
    "gate_proj": (2048, 6144),
    "up_proj": (2048, 6144),
    "down_proj": (6144, 2048),
}
SEQUENCES = (1, 32)
FIXTURES = ("zero", "qmin", "qmax", "alternating", "random", "calibration_qparam_replay")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def manifest_path(root: Path, layout: str, projection: str, seq: int) -> Path:
    case = f"{layout}_{projection}_s{seq}"
    return root / "contexts" / case / "manifests" / f"model.0.s{seq}_quant_manifest.json"


def io_contract(root: Path, projection: str, seq: int) -> dict:
    data = json.loads(manifest_path(root, "conv", projection, seq).read_text())
    input_tensor = next(value for value in data["tensors"] if value["tensor_type"] == "APP_WRITE")
    output_tensor = next(value for value in data["tensors"] if value["tensor_type"] == "APP_READ")
    return {
        "input": input_tensor["qnn_quantization"],
        "output": output_tensor["qnn_quantization"],
        "input_shape": input_tensor["dimensions"],
        "output_shape": output_tensor["dimensions"],
    }


def make_fixture(name: str, seq: int, width: int, scale: float, zero_point: int, seed: int) -> np.ndarray:
    shape = (seq, width)
    if name == "zero":
        return np.full(shape, zero_point, dtype=np.uint8)
    if name == "qmin":
        return np.zeros(shape, dtype=np.uint8)
    if name == "qmax":
        return np.full(shape, 255, dtype=np.uint8)
    if name == "alternating":
        return (np.arange(seq * width, dtype=np.uint32).reshape(shape) % 2 * 255).astype(np.uint8)
    rng = np.random.default_rng(seed)
    if name == "random":
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if name == "calibration_qparam_replay":
        # This is a deterministic replay inside the observed calibration
        # encoding, not a captured hidden state.  The distinction is explicit
        # in metadata so it cannot be mistaken for an accuracy measurement.
        real = rng.normal(loc=0.0, scale=16.0 * scale, size=shape)
        return np.clip(np.rint(real / scale) + zero_point, 0, 255).astype(np.uint8)
    raise ValueError(name)


def generate_fixtures(root: Path, output: Path) -> dict:
    rows = []
    for projection, (k, _) in PROJECTIONS.items():
        for seq in SEQUENCES:
            contract = io_contract(root, projection, seq)
            scale = float(contract["input"]["scale"])
            zero_point = int(contract["input"]["zero_point"])
            case_dir = output / f"{projection}_s{seq}"
            case_dir.mkdir(parents=True, exist_ok=True)
            for index, fixture in enumerate(FIXTURES):
                data = make_fixture(fixture, seq, k, scale, zero_point, 0x4D4C4C4D + index + seq)
                path = case_dir / f"{fixture}.raw"
                data.tofile(path)
                rows.append(
                    {
                        "projection": projection,
                        "seq": seq,
                        "fixture": fixture,
                        "path": str(path.resolve()),
                        "size": path.stat().st_size,
                        "sha256": sha256(path),
                        "input_scale": scale,
                        "input_zero_point": zero_point,
                    }
                )
    report = {
        "schema_version": 1,
        "fixture_semantics": {
            "zero": "all codes equal the calibration-derived input zero point",
            "qmin": "all UInt8 qmin",
            "qmax": "all UInt8 qmax",
            "alternating": "alternating qmin/qmax",
            "random": "fixed-seed uniform UInt8",
            "calibration_qparam_replay": (
                "fixed-seed Gaussian real values requantized with the pinned calibration qparams; "
                "not a captured layer-14 activation"
            ),
        },
        "fixtures": rows,
    }
    (output / "fixtures.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def projection_weights(model: Path, projection: str) -> np.ndarray:
    descriptors = read_descriptors(model)
    prefix = f"model.layers.14.mlp.{projection}"
    weight_desc = descriptors[prefix + ".weight"]
    scale1_desc = descriptors[prefix + ".scale1"]
    scale2_desc = descriptors[prefix + ".scale2"]
    assert weight_desc.dtype_id == DTYPE_INT8
    assert scale1_desc.dtype_id == DTYPE_UINT8
    assert scale2_desc.dtype_id == DTYPE_FLOAT32
    k, o = weight_desc.shape[2:]
    carrier_io = np.asarray(_array(model, weight_desc, np.int8)).reshape(k, o)
    carrier_oi = carrier_io.T.astype(np.int16)
    signed = np.where(carrier_oi >= 8, carrier_oi - 16, carrier_oi).astype(np.float32)
    scale1 = np.asarray(_array(model, scale1_desc, np.uint8)).reshape(o, k // 32)
    scale2 = np.asarray(_array(model, scale2_desc, np.float32)).reshape(o)
    scales = np.repeat(scale1.astype(np.float32), 32, axis=1) * scale2[:, None]
    return signed * scales


def error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.abs(actual.astype(np.int16) - reference.astype(np.int16)).astype(np.float64)
    return {
        "max_lsb": float(error.max(initial=0)),
        "p99_lsb": float(np.percentile(error, 99)),
        "rmse_lsb": float(np.sqrt(np.mean(np.square(error)))),
    }


def analyze(root: Path, model: Path, fixtures: Path, outputs: Path, report_path: Path) -> dict:
    rows = []
    failures = []
    for projection, (k, o) in PROJECTIONS.items():
        weight = projection_weights(model, projection)
        for seq in SEQUENCES:
            contract = io_contract(root, projection, seq)
            in_scale = float(contract["input"]["scale"])
            in_zp = int(contract["input"]["zero_point"])
            out_scale = float(contract["output"]["scale"])
            out_zp = int(contract["output"]["zero_point"])
            for fixture in FIXTURES:
                input_path = fixtures / f"{projection}_s{seq}" / f"{fixture}.raw"
                x_code = np.fromfile(input_path, dtype=np.uint8).reshape(seq, k)
                x = (x_code.astype(np.float32) - in_zp) * in_scale
                y = x @ weight.T
                ref_code = np.clip(np.rint(y / out_scale) + out_zp, 0, 255).astype(np.uint8)
                pair = {}
                for layout in ("conv", "matmul"):
                    output_path = outputs / f"{layout}_{projection}_s{seq}" / f"{fixture}.raw"
                    repeat_path = outputs / f"{layout}_{projection}_s{seq}" / f"{fixture}.repeat.raw"
                    if not output_path.is_file() or not repeat_path.is_file():
                        failures.append(f"missing output: {output_path} or repeat")
                        continue
                    actual = np.fromfile(output_path, dtype=np.uint8).reshape(seq, o)
                    repeat = np.fromfile(repeat_path, dtype=np.uint8).reshape(seq, o)
                    deterministic = np.array_equal(actual, repeat)
                    metrics = error_metrics(actual, ref_code)
                    pair[layout] = actual
                    rows.append(
                        {
                            "layout": layout,
                            "projection": projection,
                            "seq": seq,
                            "fixture": fixture,
                            "deterministic": deterministic,
                            "metrics": metrics,
                            "output_sha256": sha256(output_path),
                        }
                    )
                    if not deterministic:
                        failures.append(f"nondeterministic {layout}_{projection}_s{seq}/{fixture}")
                    if fixture == "zero" and not np.all(actual == out_zp):
                        failures.append(f"real-zero mismatch {layout}_{projection}_s{seq}")
                if pair.keys() == {"conv", "matmul"}:
                    conv_row = rows[-2]
                    matmul_row = rows[-1]
                    for metric in ("max_lsb", "p99_lsb", "rmse_lsb"):
                        if matmul_row["metrics"][metric] > conv_row["metrics"][metric] + 1.0:
                            failures.append(
                                f"matmul {projection}_s{seq}/{fixture} {metric} worse than Conv by >1 LSB"
                            )
                    if not np.array_equal(pair["conv"], pair["matmul"]):
                        delta = np.abs(pair["conv"].astype(np.int16) - pair["matmul"].astype(np.int16))
                        if int(delta.max()) > 1:
                            failures.append(f"Conv/MatMul code delta >1 LSB {projection}_s{seq}/{fixture}")

    report = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "reference": "float32 exact-deployed W4G32 decode and asymmetric A8 qparams",
        "rounding_note": "NumPy nearest-even is a reference; candidate is gated relative to paired Conv",
        "rows": rows,
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fixtures_parser = sub.add_parser("fixtures")
    fixtures_parser.add_argument("--artifact-root", type=Path, required=True)
    fixtures_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--artifact-root", type=Path, required=True)
    analyze_parser.add_argument("--source-model", type=Path, required=True)
    analyze_parser.add_argument("--fixtures", type=Path, required=True)
    analyze_parser.add_argument("--outputs", type=Path, required=True)
    analyze_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "fixtures":
        report = generate_fixtures(args.artifact_root, args.output)
        print(json.dumps({"status": "pass", "fixtures": len(report["fixtures"])}))
        return 0
    report = analyze(args.artifact_root, args.source_model, args.fixtures, args.outputs, args.report)
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]), "failures": report["failures"]}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
