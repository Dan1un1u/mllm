---
status: accepted
---

# Fuse one GQA core inside one mixed-resource QHPI operator

EXP-0015 will test a single `QHPI_RESOURCE_EXCLUSIVE` fusion boundary covering
QK, scale, dynamic causal masking, Softmax, and AV for the two query heads that
share one key/value head. Integer HMX performs QK and AV; HVX performs dynamic
K/V packing and the normalization work; score and probability reuse one
core-internal VTCM buffer.

This boundary is intentionally narrower than a full Attention layer and wider
than the rejected standalone custom Softmax. Eight instances may later cover a
middle layer, but the first candidate does not fuse all heads into one custom
operator. The changed QK-to-Softmax-to-AV boundary satisfies the approved
condition for revisiting Softmax code without reopening standalone Softmax
tuning.

Implementation correctness is mandatory, whereas numerical similarity to the
W4A16 comparator is diagnostic and non-blocking. Performance decisions use
equivalent-scope device wall latency and, only after local gates pass, full-model
prefill throughput.
