"""Quantization infrastructure for pymllm.

The registry is intentionally lazy.  Standalone quantization research tools
such as :mod:`pymllm.quantization.static_a8` must be importable without
loading the optional CUDA/mobile stack (notably ``flashinfer`` and the mobile
FFI extension).  Accessing one of the registry symbols below still loads the
full registry on demand, preserving the public API for serving code.
"""

__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "list_quantization_methods",
    "register_quantization",
]


def __getattr__(name: str):
    if name in __all__:
        from pymllm.quantization import quant_config

        # Import built-in methods only when the registry is requested.
        import pymllm.quantization.methods  # noqa: F401

        return getattr(quant_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
