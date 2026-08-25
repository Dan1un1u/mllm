import hashlib
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version
from transformers import AutoConfig, AutoTokenizer
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import (
    ActivationQDQ,
    FixedActivationQDQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.mobile.backends.qualcomm.transformers.core.qlinear import (
    QLinearLPBQ,
    QLinearW8A16_PerChannelSym,
)
from pymllm.mobile.backends.qualcomm.transformers.core.embedding import QEmbedding
from pymllm.mobile.backends.qualcomm.transformers.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from pymllm.mobile.backends.qualcomm.transformers.core.observer import ConcatObserver


CALIBRATION_MODES = ("legacy_minmax", "deployment_minmax")


def _copy_scalar_buffer(buffer: torch.Tensor, value: float | int) -> None:
    buffer.copy_(
        torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device).reshape_as(
            buffer
        )
    )


def apply_activation_qparams_report(model, report: dict) -> dict[str, int]:
    """Apply a complete, audited ActivationQDQ report without changing topology."""
    qparams = report.get("activation_qparams")
    if not isinstance(qparams, dict) or not qparams:
        raise ValueError("activation qparam report is empty or malformed")

    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ActivationQDQ)
    }
    missing = sorted(set(modules) - set(qparams))
    extra = sorted(set(qparams) - set(modules))
    if missing or extra:
        raise ValueError(
            "activation qparam module set mismatch: "
            f"missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)})"
        )

    for name, module in modules.items():
        values = qparams[name]
        expected_bits = int(module.bits)
        if int(values.get("bits", expected_bits)) != expected_bits:
            raise ValueError(f"activation bit mismatch at {name}")
        if int(values.get("quant_min", module.fake_quant.quant_min)) != int(
            module.fake_quant.quant_min
        ) or int(values.get("quant_max", module.fake_quant.quant_max)) != int(
            module.fake_quant.quant_max
        ):
            raise ValueError(f"activation quant range mismatch at {name}")

        minimum = float(values["min"])
        maximum = float(values["max"])
        scale = float(values["scale"])
        zero_point = int(values["zero_point"])
        if not all(math.isfinite(value) for value in (minimum, maximum, scale)):
            raise ValueError(f"non-finite activation qparam at {name}")
        if minimum > maximum or scale <= 0:
            raise ValueError(f"invalid activation range at {name}")
        if not module.fake_quant.quant_min <= zero_point <= module.fake_quant.quant_max:
            raise ValueError(f"activation zero point is outside range at {name}")

        observer = module.fake_quant.activation_post_process
        _copy_scalar_buffer(observer.min_val, minimum)
        _copy_scalar_buffer(observer.max_val, maximum)
        _copy_scalar_buffer(module.fake_quant.scale, scale)
        _copy_scalar_buffer(module.fake_quant.zero_point, zero_point)
        module.disable_observer()
        module.enable_fakequant()

    return {"applied": len(modules), "missing": 0, "extra": 0}


def recompute_scale_zp(module):
    """
    Callback function: Used to forcefully refresh scale and zero_point of all FakeQuantize modules after calibration.

    Problem solved:
        When using ConcatObserver, min/max may be updated during forward pass,
        but at the end of forward, the scale/zp stored in FakeQuantize's internal buffer are still computed from old min/max.
        This function forces a calculate_qparams call to sync the latest parameters to the buffer.

    Usage:
        model.apply(recompute_scale_zp)
    """

    # We mainly focus on FakeQuantize modules since they store the scale/zero_point buffers
    # Note: model.apply recursively traverses all submodules, so self.fake_quant inside ActivationQDQ will also be visited
    if isinstance(module, ActivationQDQ):
        observer = module.fake_quant.activation_post_process

        # 2. Check if observer is valid and contains statistics
        # We only care about MinMaxObserver or MovingAverageMinMaxObserver that have min_val/max_val
        if hasattr(observer, "min_val") and hasattr(observer, "max_val"):
            # 3. Check if data is initialized
            # If min_val is still the initial inf, this layer hasn't processed data, skip to avoid errors
            if observer.min_val.numel() == 0 or observer.max_val.numel() == 0:
                return
            if (
                torch.isinf(observer.min_val).any()
                or torch.isinf(observer.max_val).any()
            ):
                return

            # 4. Recompute Scale and Zero Point
            # calculate_qparams reads the current min_val/max_val from observer (may have been modified by ConcatObserver)
            try:
                scale, zero_point = observer.calculate_qparams()
            except Exception as e:
                # Some special Observers (e.g., FixedQParams) may not support recomputation or behave differently, safely skip
                print(e)
                return

            # 5. Force overwrite the computed results to FakeQuantize's Buffer
            # Use copy_ to keep reference unchanged, ensuring the new values are used during export
            if (
                hasattr(module.fake_quant, "scale")
                and module.fake_quant.scale is not None
            ):
                # Ensure dimension match (handle per-channel vs per-tensor)
                if module.fake_quant.scale.shape != scale.shape:
                    module.fake_quant.scale.resize_(scale.shape)
                module.fake_quant.scale.copy_(scale)
                # Try to get the registered name of module scale from _parameters or _buffers
                for key, value in module.fake_quant.named_parameters():
                    if value is module.fake_quant.scale:
                        print(f"{module._get_name()}.{key}: {module.scale}")
                        break

            if (
                hasattr(module.fake_quant, "zero_point")
                and module.fake_quant.zero_point is not None
            ):
                if module.fake_quant.zero_point.shape != zero_point.shape:
                    module.fake_quant.zero_point.resize_(zero_point.shape)
                module.fake_quant.zero_point.copy_(zero_point)


def validate_concat_observer_fn(module, results: list, name: str = ""):
    """
    Callback function: Validate that all input_observers in ConcatObserver have consistent scale and zero_point.

    Usage:
        results = []
        for name, m in model.named_modules():
            validate_concat_observer_fn(m, results, name)
    """
    if not isinstance(module, ConcatObserver):
        return

    input_observers = module.input_observers
    if len(input_observers) == 0:
        return

    # Collect scale and zero_point from all observers
    scales_zps = []
    for i, observer in enumerate(input_observers):
        try:
            scale, zp = observer.calculate_qparams()
            scales_zps.append(f"[{i}] s={scale.item():.8f} zp={zp.item()}")
        except Exception:
            scales_zps.append(f"[{i}] failed")

    # Print one line: scale and zp of all inputs for each concat observer
    print(f"ConcatObserver [{name}]: {' | '.join(scales_zps)}")

    # Original validation logic
    if len(input_observers) <= 1:
        return

    # Get scale and zero_point from the first observer as reference
    first_observer = input_observers[0]
    try:
        ref_scale, ref_zp = first_observer.calculate_qparams()
    except Exception:
        return

    # Check if all other observers have the same scale and zero_point
    for i, observer in enumerate(input_observers[1:], start=1):
        try:
            scale, zp = observer.calculate_qparams()
        except Exception:
            results.append(f"Failed to calculate qparams for observer[{i}]")
            continue

        scale_match = torch.allclose(ref_scale, scale, rtol=1e-5, atol=1e-8)
        zp_match = torch.equal(ref_zp, zp)

        if not scale_match or not zp_match:
            results.append(
                f"observer[{i}] mismatch: ref_scale={ref_scale.item():.8f}, "
                f"scale={scale.item():.8f}, ref_zp={ref_zp.item()}, zp={zp.item()}"
            )


def freeze_qwen3_rmsnorm_weight(m):
    if isinstance(m, QRMSNorm):
        m.freeze_weight()


def freeze_qwen3_linear_weight(m):
    if isinstance(m, QLinearLPBQ) or isinstance(m, QLinearW8A16_PerChannelSym):
        m.freeze_weight()


def freeze_qwen3_embed_tokens_weight(m):
    if isinstance(m, QEmbedding):
        m.freeze_weight()


def disable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.disable_observer()


def enable_qdq_observer(m):
    if isinstance(m, ActivationQDQ):
        m.enable_observer()


def enable_fake_quant(m):
    if isinstance(m, ActivationQDQ) or isinstance(m, FixedActivationQDQ):
        m.enable_fakequant()
    if isinstance(m, QLinearLPBQ):
        m.enable_fakequant()
    if isinstance(m, QRMSNorm):
        m.enable_fakequant()
    if isinstance(m, QEmbedding):
        m.enable_fakequant()


def disable_dynamic_activation_fake_quant(m):
    """Disable only observer-backed activation QDQ simulation.

    During PTQ calibration these QDQ modules do not have usable qparams yet, so
    they must act as observers rather than quantizers.  Fixed activation QDQ and
    already-frozen weights deliberately remain quantized to match deployment.
    """
    if isinstance(m, ActivationQDQ):
        m.disable_fakequant()


def disable_fake_quant(m):
    if isinstance(m, ActivationQDQ) or isinstance(m, FixedActivationQDQ):
        m.disable_fakequant()
    if isinstance(m, QLinearLPBQ):
        m.disable_fakequant()
    if isinstance(m, QRMSNorm):
        m.disable_fakequant()
    if isinstance(m, QEmbedding):
        m.disable_fakequant()


def _enabled_flag(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().reshape(-1)[0].item())
    return bool(value)


def calibration_fake_quant_state(model) -> dict[str, dict[str, int]]:
    """Return a compact audit of the fake-quant state used for calibration."""
    state = {
        "dynamic_activation": {"enabled": 0, "disabled": 0},
        "fixed_activation": {"enabled": 0, "disabled": 0},
        "lpbq_weight": {"enabled": 0, "disabled": 0},
        "rmsnorm_weight": {"enabled": 0, "disabled": 0},
        "embedding_weight": {"enabled": 0, "disabled": 0},
    }
    for module in model.modules():
        category = None
        flag = None
        if isinstance(module, ActivationQDQ):
            category = "dynamic_activation"
            flag = module.fake_quant.fake_quant_enabled
        elif isinstance(module, FixedActivationQDQ):
            category = "fixed_activation"
            flag = module.fake_quant.fake_quant_enabled
        elif isinstance(module, QLinearLPBQ):
            category = "lpbq_weight"
            flag = module.weight_quant.fake_quant_enabled
        elif isinstance(module, QRMSNorm):
            category = "rmsnorm_weight"
            flag = module.weight_fake_quant.fake_quant_enabled
        elif isinstance(module, QEmbedding):
            category = "embedding_weight"
            flag = module.weight_fake_quant.fake_quant_enabled
        if category is not None:
            key = "enabled" if _enabled_flag(flag) else "disabled"
            state[category][key] += 1
    return state


def configure_calibration_fake_quant(model, mode: str) -> dict[str, dict[str, int]]:
    """Configure either the archived or deployment-faithful calibration state."""
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"Unsupported calibration mode: {mode}")

    if mode == "legacy_minmax":
        model.apply(disable_fake_quant)
    else:
        # Start from a known all-enabled state, then suppress only dynamic
        # activation QDQ until its observers have collected valid ranges.
        model.apply(enable_fake_quant)
        model.apply(disable_dynamic_activation_fake_quant)

    state = calibration_fake_quant_state(model)
    if mode == "deployment_minmax":
        errors = []
        dynamic = state["dynamic_activation"]
        if dynamic["enabled"] or not dynamic["disabled"]:
            errors.append(f"dynamic_activation={dynamic}")
        for category in (
            "fixed_activation",
            "lpbq_weight",
            "rmsnorm_weight",
            "embedding_weight",
        ):
            counts = state[category]
            if counts["disabled"] or not counts["enabled"]:
                errors.append(f"{category}={counts}")
        if errors:
            raise RuntimeError(
                "Deployment-faithful calibration state is invalid: "
                + ", ".join(errors)
            )
    return state


def convert_weight(m):
    if isinstance(m, QLinearLPBQ) or isinstance(m, QLinearW8A16_PerChannelSym):
        m.convert_to_conv2d_deploy_hwio()
    if isinstance(m, QRMSNorm):
        m.convert_to_deploy()
    if isinstance(m, QEmbedding):
        m.convert_to_deploy()

def _check_datasets_compatibility():
    try:
        ds_ver = version("datasets")
    except PackageNotFoundError as e:
        raise RuntimeError(
            "datasets is required for calibration. "
            "Please install a compatible version such as datasets==2.21.0."
        ) from e

    if Version(ds_ver) >= Version("3.0.0"):
        raise RuntimeError(
            f"Incompatible datasets version detected: {ds_ver}. "
            "Current Qualcomm calibration depends on a modelscope-compatible "
            "datasets version. Please use datasets==2.21.0."
        )
        
class Qwen3Quantizer:
    def __init__(
        self,
        model_path: str,
        mllm_qualcomm_max_length=2048,
        activation_bits: int = 8,
        linear_block_size: int = 32,
    ):
        if activation_bits not in (8, 16):
            raise ValueError("activation_bits must be 8 or 16")
        if linear_block_size <= 0:
            raise ValueError("linear_block_size must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.activation_bits = activation_bits
        config.linear_block_size = linear_block_size
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        self.model.cuda()
        self.mllm_qualcomm_max_length = mllm_qualcomm_max_length
        self.model.mllm_qualcomm_max_length = mllm_qualcomm_max_length

        if self.model.config.tie_word_embeddings:
            self.model.copy_lm_head_weight_from_embed_tokens()

        # PTQ All Weights.
        self.model.apply(freeze_qwen3_rmsnorm_weight)
        self.model.apply(freeze_qwen3_linear_weight)
        self.model.apply(freeze_qwen3_embed_tokens_weight)
        print("All PTQ weights preparation done.")

    def _build_model_inputs(self, prompt: str, max_length: int | None = None):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )
        tokenizer_kwargs = {"return_tensors": "pt"}
        if max_length is not None:
            tokenizer_kwargs.update(
                max_length=max_length,
                truncation=True,
                padding=False,
            )
        return self.tokenizer([text], **tokenizer_kwargs).to(self.model.device)

    def freeze_activation(self):
        self.model.apply(disable_qdq_observer)

    def load_activation_qparams(self, report_path: str):
        """Load frozen offline qparams while preserving the deployed graph contract."""
        path = Path(report_path)
        serialized = path.read_bytes()
        report = json.loads(serialized)
        self.enable_fake_quant()
        audit = apply_activation_qparams_report(self.model, report)
        audit.update(
            {
                "path": str(path),
                "sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
        print("Loaded activation qparams: " + json.dumps(audit, sort_keys=True))
        return audit

    def enable_activation_update(self):
        self.model.apply(enable_qdq_observer)

    def enable_fake_quant(self):
        self.model.apply(enable_fake_quant)

    def disable_fake_quant(self):
        self.model.apply(disable_fake_quant)

    def compile(self):
        print("Compile Start.")
        self.model = torch.compile(
            self.model, mode="reduce-overhead", fullgraph=False, backend="inductor"
        )
        print("Compile done.")

    def infer(self, prompt: str, max_new_tokens: int = 8):
        model_inputs = self._build_model_inputs(prompt)
        input_length = model_inputs.input_ids.shape[1]
        available_tokens = self.mllm_qualcomm_max_length - input_length - 1
        if available_tokens < 1:
            raise ValueError("Prompt exceeds configured mllm_qualcomm_max_length")
        max_new_tokens = min(max_new_tokens, available_tokens)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        output_ids = generated_ids[0][input_length:].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)

    def capture_calibration_corpus(
        self, output_path: str, num_samples=128, max_seq_length=512
    ):
        """Capture the exact text and token IDs used by formal calibration."""
        _check_datasets_compatibility()
        from datasets import load_dataset

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        records = []
        for entry in dataset:
            source_text = entry["text"].strip()
            if len(source_text) < 1024:
                continue
            model_inputs = self._build_model_inputs(source_text, max_length=max_seq_length)
            records.append(
                {
                    "index": len(records),
                    "source": "huggingface/wikitext:wikitext-103-v1:train",
                    "text": source_text,
                    "input_ids": model_inputs.input_ids[0].cpu().tolist(),
                    "attention_mask": model_inputs.attention_mask[0].cpu().tolist(),
                }
            )
            if len(records) == num_samples:
                break
        if len(records) != num_samples:
            raise RuntimeError(f"Captured {len(records)} samples, expected {num_samples}")
        serialized = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        )
        output.write_text(serialized, encoding="utf-8")
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        print(f"Captured calibration corpus: {output} sha256={digest}")
        return digest

    def calibrate(
        self,
        calibration_corpus: str,
        num_samples=128,
        max_seq_length=512,
        calibration_mode: str = "deployment_minmax",
    ):
        """
        Perform calibration using Wikipedia dataset (PTQ)
        :param num_samples: Number of samples for calibration
        :param max_seq_length: Maximum length for each sample (not exceeding mllm_qualcomm_max_length)
        """
        print(
            f"Starting calibration, samples: {num_samples}, max length: {max_seq_length}, "
            f"mode: {calibration_mode}"
        )

        # Dynamic activation QDQ observes in floating point.  In deployment
        # mode all frozen weights and fixed-domain activation QDQ remain active,
        # so downstream observers see the same upstream model used at runtime.
        calibration_state = configure_calibration_fake_quant(
            self.model, calibration_mode
        )
        print(
            "Calibration fake-quant state: "
            + json.dumps(calibration_state, sort_keys=True)
        )

        # 1. Enable QDQ Observer for activation values
        self.enable_activation_update()
        self.model.eval()

        corpus_path = Path(calibration_corpus)
        records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        if len(records) != num_samples:
            raise RuntimeError(
                f"Pinned calibration corpus has {len(records)} samples; expected {num_samples}"
            )

        # 3. Execute forward pass (Prefill stage)
        samples_processed = 0

        # Ensure no gradient calculation during inference
        with torch.no_grad():
            pbar = tqdm(total=num_samples, desc="Calibrating")
            for expected_index, entry in enumerate(records):
                if entry.get("index") != expected_index:
                    raise RuntimeError("Calibration corpus indices are not contiguous")
                input_ids = entry.get("input_ids", [])
                attention_mask = entry.get("attention_mask", [])
                if not input_ids or len(input_ids) != len(attention_mask):
                    raise RuntimeError(f"Invalid token record at index {expected_index}")
                if len(input_ids) > max_seq_length:
                    raise RuntimeError(
                        f"Pinned sample {expected_index} exceeds max_seq_length={max_seq_length}"
                    )
                model_inputs = {
                    "input_ids": torch.tensor([input_ids], dtype=torch.long, device=self.model.device),
                    "attention_mask": torch.tensor(
                        [attention_mask], dtype=torch.long, device=self.model.device
                    ),
                }

                # Only need Prefill stage: directly call forward
                # This will trigger observer update statistics in ActivationQDQ
                self.model(**model_inputs, use_cache=False)

                samples_processed += 1
                pbar.update(1)

        # 4. Close observers, commit their final qparams, then enable the full
        # deployment fake-quant contract for the software correctness gate.
        self.freeze_activation()
        self.recompute_scale_zp()
        self.enable_fake_quant()
        print(
            "\nCalibration completed, activation quantization parameters frozen "
            "and deployment fake quantization enabled."
        )

    def convert(self):
        self.model.apply(convert_weight)
        self.model.model.convert_rope_for_deploy()

    def recompute_scale_zp(self):
        self.model.apply(recompute_scale_zp)

    def validate_concat_observer(self):
        results = []
        for name, module in self.model.named_modules():
            validate_concat_observer_fn(module, results, name)
        if results:
            print("ConcatObserver validation FAILED:")
            for msg in results:
                print(f"  {msg}")
            raise ValueError("ConcatObserver validation FAILED")
        else:
            print(
                "ConcatObserver validation PASSED: all observers have matching scale and zp"
            )
        print("ConcatObserver validation done.", flush=True)
