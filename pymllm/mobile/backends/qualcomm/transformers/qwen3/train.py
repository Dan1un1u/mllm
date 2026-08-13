import argparse
import os
from pymllm.mobile.backends.qualcomm.transformers.qwen3.runner import Qwen3Quantizer
from pymllm.mobile.convertor.model_file_v2 import ModelFileV2


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Quantizer for Qualcomm backend")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen3-1.7B",
        help="Path to the Qwen3 model directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for quantization",
    )
    parser.add_argument(
        "--num_samples", type=int, default=128, help="Number of samples for calibration"
    )
    parser.add_argument(
        "--calibration_corpus",
        type=str,
        required=True,
        help="Pinned JSONL corpus containing the exact calibration token IDs",
    )
    parser.add_argument(
        "--capture_calibration_corpus",
        action="store_true",
        help="Capture the pinned corpus before replaying it",
    )
    parser.add_argument("--activation_bits", type=int, choices=(8, 16), default=8)
    parser.add_argument("--linear_block_size", type=int, default=32)
    parser.add_argument(
        "--infer_text",
        type=str,
        default="为什么伟大不能被计划",
        help="Text to run inference on",
    )
    parser.add_argument(
        "--infer_max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens for post-calibration inference sanity check",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the quantized model",
    )
    parser.add_argument(
        "--output_mllm",
        type=str,
        help="Optional direct V2 .mllm output preserving native UInt16 tensors",
    )

    args = parser.parse_args()

    m = Qwen3Quantizer(
        args.model_path,
        mllm_qualcomm_max_length=args.max_length,
        activation_bits=args.activation_bits,
        linear_block_size=args.linear_block_size,
    )

    if args.capture_calibration_corpus:
        m.capture_calibration_corpus(
            args.calibration_corpus,
            num_samples=args.num_samples,
            max_seq_length=args.max_length,
        )
    m.disable_fake_quant()
    m.calibrate(
        args.calibration_corpus,
        num_samples=args.num_samples,
        max_seq_length=args.max_length,
    )
    m.enable_fake_quant()
    m.recompute_scale_zp()
    m.validate_concat_observer()
    m.infer(args.infer_text, max_new_tokens=args.infer_max_new_tokens)
    m.convert()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_mllm:
        state_dict = m.model.state_dict()
        writer = ModelFileV2(
            args.output_mllm,
            "Qwen3-1.7B-W4A8G32",
            "Streaming",
            max_params_descriptor_buffer_num=len(state_dict),
        )
        for name, tensor in state_dict.items():
            writer.streaming_write(name, tensor.detach().cpu().contiguous())
        writer.finalize()
        writer.file_handler.close()
        print(f"Saved native V2 model file: {args.output_mllm}")
    else:
        raise ValueError(
            "--output_mllm is required: safetensors cannot preserve torch.uint16 "
            "RMSNorm/Embedding weights without changing their dtype contract"
        )


if __name__ == "__main__":
    main()
