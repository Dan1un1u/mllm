# W4A8G32 Experiment Governance

This glossary defines the language used to govern the clean-room W4A8G32
research line.

## Language

**Project Memory**:
The single authoritative record of project constraints, selected references,
experiments, evidence, and decisions.
_Avoid_: chat memory, branch notes, session summary

**Archived Reference**:
A preserved historical implementation and result used to reconstruct the
project's starting point, not to represent the current toolchain.
_Avoid_: current baseline, comparator

**Performance Comparator**:
The approved result produced with the current toolchain and comparison protocol
against which performance claims are evaluated.
_Avoid_: archived reference, baseline

**Selected Baseline**:
The user-approved implementation that currently represents the project outcome
and is registered in Project Memory.
_Avoid_: candidate, local pass, best branch

**Candidate**:
An implementation being evaluated that has not been approved as the Selected
Baseline.
_Avoid_: baseline

**Experiment**:
An immutable, identified investigation with a declared hypothesis, variable,
gate, evidence profile, and conclusion.
_Avoid_: attempt, quick test

**Experiment Family**:
An Experiment whose variants test one shared physical hypothesis.
_Avoid_: unrelated collection

**Variant**:
A child Experiment that changes a declared dimension within an Experiment
Family.
_Avoid_: new hypothesis

**Local Gate**:
The predeclared condition that determines whether an Experiment may advance to
its next scope.
_Avoid_: adoption, baseline promotion

**Evidence Validity**:
The judgment that required evidence exists, is internally consistent, and
matches its recorded identity.
_Avoid_: performance success

**Adoption Status**:
The project decision about whether an Experiment's outcome is used.
_Avoid_: execution result, local gate

**Stateful Work**:
Any source edit, build, model generation, device execution, or profiling action
that can change project or external state.
_Avoid_: read-only analysis

**Formal Result**:
A retained result set satisfying its evidence profile and artifact identity
requirements.
_Avoid_: smoke run, console observation

**Comparable Result**:
A Formal Result whose comparison key matches except for the declared
experimental variable.
_Avoid_: cross-environment indication

**Reopen Condition**:
An objective change in execution contract required before a rejected Experiment
Family may be reconsidered.
_Avoid_: more tuning, try again

**Legacy W4A8**:
The W4A8-specific work that predates the clean-room derivation from the archived
W4A16 rotation line.
_Avoid_: new W4A8

**New W4A8**:
The clean-room W4A8 work beginning from the archived W4A16 implementation.
_Avoid_: Legacy W4A8

**Prefill Track**:
Performance work evaluated on prompt processing.
_Avoid_: decode optimization

**Decode Track**:
Performance work evaluated on autoregressive token generation after the first
token.
_Avoid_: prefill optimization

**GQA Group**:
The smallest fair Attention comparison unit: all query heads that share one
key/value head and the associated data preparation for that shared head.
_Avoid_: single query head, full Attention layer

**Fused GQA Core**:
The candidate Attention subgraph that consumes one GQA Group after positional
processing and cache assembly, performs score computation, scaling, masking,
normalization, and value aggregation within one fusion boundary, and produces
that group's Attention outputs.
_Avoid_: fused Softmax, full Attention layer, QKV projection
