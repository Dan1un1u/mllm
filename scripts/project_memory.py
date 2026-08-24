#!/usr/bin/env python3
"""Validate and mutate the mllm W4A8 project-memory control plane."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "PROJECT_STATUS.yaml"
INDEX_PATH = ROOT / "experiments" / "index.yaml"
ARTIFACT_PATH = ROOT / "artifacts" / "formal_artifacts.yaml"
CONTRACT_PATH = ROOT / "PROJECT_CONTRACT.md"

EXPECTED_REPOSITORY = "https://github.com/Dan1un1u/mllm.git"
EXPECTED_MEMORY_BRANCH = "codex/w4a8g32-project-memory"
EXPECTED_QAIRT = "2.49.0.260730"
EXPECTED_CLEAN_ROOM_ROOT = "462c9207"
QUICK_HASH_LIMIT = 256 * 1024 * 1024
CONTROL_FILE_LIMIT = 5 * 1024 * 1024

EXECUTION_VALUES = ["proposed", "approved", "running", "completed", "aborted"]
EVIDENCE_VALUES = ["not_checked", "valid", "invalid", "inconclusive"]
GATE_VALUES = ["not_run", "pass", "fail", "not_applicable"]
ADOPTION_VALUES = ["pending", "accepted", "rejected", "superseded", "not_applicable"]

REQUIRED_ARTIFACT_ROLES = {
    "archived_reference": {
        "source_model", "source_export_manifest", "compiled_context",
        "quant_manifest_s1", "quant_manifest_s32", "schematic_s1", "schematic_s32",
        "result_artifact_manifest", "result_metadata", "speed_summary",
        "accuracy_summary", "critical_path_report", "raw_optrace_s1", "raw_optrace_s32",
    },
    "performance_comparator": {
        "source_model", "source_export_manifest", "compiled_context",
        "quant_manifest_s1", "quant_manifest_s32", "schematic_s1", "schematic_s32",
        "result_artifact_manifest", "result_metadata", "speed_summary",
        "accuracy_summary", "raw_optrace_s1", "raw_optrace_s32",
    },
    "selected_w4a8_baseline": {
        "source_model", "compiled_context", "profile_contract",
        "quant_manifest_s1", "quant_manifest_s32", "schematic_s1", "schematic_s32",
        "result_artifact_manifest", "result_metadata", "speed_summary",
        "accuracy_summary", "critical_path_report", "raw_optrace_s1", "raw_optrace_s32",
    },
}

RESULT_ARTIFACT_ROLES = {
    "result_artifact_manifest", "result_metadata", "speed_summary",
    "accuracy_summary", "critical_path_report", "raw_optrace_s1", "raw_optrace_s32",
}


class ValidationFailure(RuntimeError):
    pass


def run(args: list[str], *, cwd: Path | None = None, check: bool = True,
        timeout: int = 120) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args, cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    if check and proc.returncode != 0:
        command = " ".join(args)
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise ValidationFailure(f"command failed: {command}: {detail}")
    return proc


def git(args: list[str], *, cwd: Path = ROOT,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], cwd=cwd, check=check)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationFailure(f"{path.relative_to(ROOT)} must contain a YAML mapping")
    return data


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValidationFailure(f"{path.relative_to(ROOT)} must contain a JSON object")
    return data


def atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def canonical_git_url(url: str) -> str:
    return url[:-4] if url.endswith(".git") else url


def git_common_dir(worktree: Path) -> Path:
    raw = git(["rev-parse", "--git-common-dir"], cwd=worktree).stdout.strip()
    path = Path(raw)
    if not path.is_absolute():
        path = worktree / path
    return path.resolve()


def git_branch(worktree: Path) -> str:
    proc = git(["symbolic-ref", "--quiet", "--short", "HEAD"],
               cwd=worktree, check=False)
    if proc.returncode != 0:
        raise ValidationFailure(f"{worktree} is detached or has no branch")
    return proc.stdout.strip()


def git_head(worktree: Path) -> str:
    return git(["rev-parse", "HEAD"], cwd=worktree).stdout.strip()


def git_origin(worktree: Path) -> str:
    return git(["remote", "get-url", "origin"], cwd=worktree).stdout.strip()


def git_dirty(worktree: Path) -> list[str]:
    output = git(["status", "--porcelain=v1", "--untracked-files=all"],
                 cwd=worktree).stdout
    return [line for line in output.splitlines() if line]


def resolve_commit(commit: str) -> str | None:
    proc = git(["rev-parse", "--verify", f"{commit}^{{commit}}"], check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def ref_commit(ref: str) -> str | None:
    proc = git(["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def is_ancestor(commit: str, ref: str) -> bool:
    return git(["merge-base", "--is-ancestor", commit, ref],
               check=False).returncode == 0


def windows_to_wsl(root_windows: str, relative_path: str) -> Path:
    match = re.fullmatch(r"([A-Za-z]):\\(.+)", root_windows)
    if not match:
        raise ValidationFailure(f"invalid Windows root: {root_windows}")
    drive = match.group(1).lower()
    if drive != "d":
        raise ValidationFailure(f"formal artifacts must be on D:, got {root_windows}")
    root_parts = re.split(r"[\\/]+", match.group(2))
    rel_parts = re.split(r"[\\/]+", relative_path)
    for part in [*root_parts, *rel_parts]:
        if part in {"", ".", ".."}:
            raise ValidationFailure(
                f"non-canonical artifact path: {root_windows} / {relative_path}"
            )
    path = (Path("/mnt") / drive / Path(*root_parts) / Path(*rel_parts)).resolve()
    allowed = Path("/mnt/d/llm_exp").resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise ValidationFailure(f"artifact escapes D:\\llm_exp: {path}") from exc
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_errors(instance: dict[str, Any], schema_path: Path) -> list[str]:
    validator = jsonschema.Draft7Validator(load_json(schema_path))
    result = []
    for error in sorted(validator.iter_errors(instance),
                        key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        result.append(f"{schema_path.name}:{location}: {error.message}")
    return result


def remote_heads() -> dict[str, str]:
    output = git(["ls-remote", "--heads", "origin"]).stdout
    result: dict[str, str] = {}
    for line in output.splitlines():
        commit, ref = line.split(maxsplit=1)
        result[ref.removeprefix("refs/heads/")] = commit
    return result


def remote_tags() -> dict[str, tuple[str, str | None]]:
    output = git(["ls-remote", "--tags", "origin"]).stdout
    raw: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in output.splitlines():
        commit, ref = line.split(maxsplit=1)
        if ref.endswith("^{}"):
            peeled[ref[len("refs/tags/"):-3]] = commit
        else:
            raw[ref.removeprefix("refs/tags/")] = commit
    return {tag: (object_id, peeled.get(tag)) for tag, object_id in raw.items()}


def required_remote_branches() -> list[str]:
    status = load_yaml(STATUS_PATH)
    index = load_yaml(INDEX_PATH)
    branches = {status["project"]["memory_branch"]}
    for experiment in index["experiments"]:
        if experiment["execution_state"] == "completed":
            branches.update(experiment["source_branches"])
    return sorted(branches)


def sync_required_refs() -> None:
    branches = required_remote_branches()
    refspecs = [
        f"+refs/heads/{branch}:refs/remotes/origin/{branch}"
        for branch in branches
    ]
    run(["git", "fetch", "--no-tags", "origin", *refspecs], cwd=ROOT, timeout=300)
    remote = remote_heads()
    problems = []
    for branch in branches:
        remote_tip = remote.get(branch)
        tracked_tip = ref_commit(f"refs/remotes/origin/{branch}")
        if remote_tip is None:
            problems.append(f"required branch is absent from origin: {branch}")
        elif tracked_tip != remote_tip:
            problems.append(f"explicit fetch did not synchronize: {branch}")
    if problems:
        raise ValidationFailure("; ".join(problems))


class ProjectValidator:
    def __init__(self, *, full: bool = False, require_clean: bool = True):
        self.full = full
        self.require_clean = require_clean
        self.errors: list[str] = []
        self.status = load_yaml(STATUS_PATH)
        self.index = load_yaml(INDEX_PATH)
        self.artifacts = load_yaml(ARTIFACT_PATH)
        self.remote_head_map: dict[str, str] | None = None
        self.remote_tag_map: dict[str, tuple[str, str | None]] | None = None

    def error(self, message: str) -> None:
        self.errors.append(message)

    def check(self) -> None:
        self.check_schemas()
        self.check_control_plane()
        self.check_contract()
        self.check_experiments()
        self.check_role_tags()
        self.check_artifacts()
        if self.full:
            self.check_remote_memory()

    def raise_if_failed(self) -> None:
        if self.errors:
            body = "\n".join(f"  - {item}" for item in self.errors)
            raise ValidationFailure(f"project-memory validation failed:\n{body}")

    def check_schemas(self) -> None:
        self.errors.extend(schema_errors(
            self.status, ROOT / "schemas" / "project-status.schema.json"))
        self.errors.extend(schema_errors(
            self.index, ROOT / "schemas" / "experiment-index.schema.json"))
        self.errors.extend(schema_errors(
            self.artifacts, ROOT / "schemas" / "formal-artifacts.schema.json"))

    def check_control_plane(self) -> None:
        project = self.status.get("project", {})
        toolchain = self.status.get("toolchain", {})
        governance = self.status.get("governance", {})
        if project.get("repository_url") != EXPECTED_REPOSITORY:
            self.error("PROJECT_STATUS repository_url is not the approved origin")
        if project.get("memory_branch") != EXPECTED_MEMORY_BRANCH:
            self.error("PROJECT_STATUS memory_branch changed")
        if Path(project.get("memory_worktree", "")).resolve() != ROOT:
            self.error("PROJECT_STATUS memory_worktree does not match this worktree")
        if toolchain.get("qairt_release") != EXPECTED_QAIRT:
            self.error("active QAIRT release is not 2.49.0.260730")
        if governance.get("maximum_running_experiments") != 1:
            self.error("maximum_running_experiments must remain 1")
        if governance.get("sync_policy") != "fetch_and_fast_forward_only":
            self.error("sync policy must be fetch_and_fast_forward_only")
        if governance.get("stale_fallback_allowed") is not False:
            self.error("stale fallback must remain disabled")
        try:
            branch = git_branch(ROOT)
            if branch != EXPECTED_MEMORY_BRANCH:
                self.error(f"memory worktree is on {branch}, expected {EXPECTED_MEMORY_BRANCH}")
            if canonical_git_url(git_origin(ROOT)) != canonical_git_url(EXPECTED_REPOSITORY):
                self.error("memory worktree origin is not the approved repository")
            configured_common = Path(project.get("git_common_dir", "")).resolve()
            if git_common_dir(ROOT) != configured_common:
                self.error("memory worktree Git common directory changed")
            if self.require_clean:
                dirty = git_dirty(ROOT)
                if dirty:
                    self.error(f"memory worktree is dirty: {dirty[:5]}")
        except ValidationFailure as exc:
            self.error(str(exc))
        forbidden = {".bin", ".mllm", ".log", ".html", ".so", ".a", ".o"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.name == ".git":
                continue
            relative = path.relative_to(ROOT)
            if path.suffix.lower() in forbidden:
                self.error(f"forbidden artifact in control plane: {relative}")
            if path.stat().st_size > CONTROL_FILE_LIMIT:
                self.error(f"control-plane file exceeds 5 MiB: {relative}")

    def check_contract(self) -> None:
        text = CONTRACT_PATH.read_text(encoding="utf-8")
        ids = [int(value) for value in re.findall(r"PC-([0-9]{3})", text)]
        if sorted(set(ids)) != list(range(1, 31)):
            self.error("PROJECT_CONTRACT must contain exactly PC-001 through PC-030")
        if len(ids) != 30:
            self.error("PROJECT_CONTRACT contains duplicate PC identifiers")
        if EXPECTED_QAIRT not in text:
            self.error("PROJECT_CONTRACT does not pin QAIRT 2.49.0.260730")
        if not (ROOT / "docs/exclusions/EXCLUDED-001.md").is_file():
            self.error("EXCLUDED-001 legacy boundary is missing")

    def check_experiments(self) -> None:
        experiments = self.index.get("experiments", [])
        expected_allowed = {
            "execution_state": EXECUTION_VALUES,
            "evidence_validity": EVIDENCE_VALUES,
            "local_gate": GATE_VALUES,
            "adoption_status": ADOPTION_VALUES,
        }
        if self.index.get("allowed_values") != expected_allowed:
            self.error("allowed_values differs from the four-axis contract")
        if not isinstance(experiments, list):
            self.error("experiments must be a list")
            return
        by_id: dict[str, dict[str, Any]] = {}
        slugs: set[str] = set()
        for experiment in experiments:
            exp_id = experiment.get("id")
            slug = experiment.get("slug")
            if exp_id in by_id:
                self.error(f"duplicate experiment id: {exp_id}")
            else:
                by_id[exp_id] = experiment
            if slug in slugs:
                self.error(f"duplicate experiment slug: {slug}")
            slugs.add(slug)
            record = ROOT / experiment.get("record", "")
            if not record.is_file():
                self.error(f"{exp_id} record missing: {record}")
            if (experiment.get("adoption_status") == "accepted"
                    and experiment.get("decided_by") != "user"):
                self.error(f"{exp_id} accepted without user decision")
            if experiment.get("execution_state") == "completed":
                if not experiment.get("source_branches") or not experiment.get("source_commits"):
                    self.error(f"{exp_id} completed without source branches and commits")
        for exp_id, experiment in by_id.items():
            parent = experiment.get("parent")
            if parent is not None:
                if parent not in by_id:
                    self.error(f"{exp_id} parent does not exist: {parent}")
                elif exp_id not in by_id[parent].get("variants", []):
                    self.error(f"{exp_id} is absent from {parent}.variants")
            for child in experiment.get("variants", []):
                if child not in by_id:
                    self.error(f"{exp_id} variant does not exist: {child}")
                elif by_id[child].get("parent") != exp_id:
                    self.error(f"{exp_id}/{child} parent-child mismatch")
        running = [item["id"] for item in experiments
                   if item.get("execution_state") == "running"]
        active = self.status.get("governance", {}).get("active_experiment")
        if len(running) > 1:
            self.error(f"more than one running experiment: {running}")
        expected_active = running[0] if len(running) == 1 else None
        if active != expected_active:
            self.error(f"active_experiment {active!r} does not match running set {running}")
        next_number = self.status.get("governance", {}).get("next_experiment_number")
        parent_numbers = [
            int(match.group(1)) for exp_id in by_id
            if (match := re.fullmatch(r"EXP-([0-9]{4})", exp_id))
        ]
        if not isinstance(next_number, int) or (
                parent_numbers and next_number <= max(parent_numbers)):
            self.error("next_experiment_number is not ahead of all parent experiments")
        for closed in self.status.get("closed_families", []):
            exp_id = closed.get("experiment")
            if exp_id not in by_id:
                self.error(f"closed family missing from index: {exp_id}")
            elif by_id[exp_id].get("adoption_status") != "rejected":
                self.error(f"closed family is not rejected: {exp_id}")
        self.check_source_references(experiments)

    def check_source_references(self, experiments: list[dict[str, Any]]) -> None:
        remote = None
        if self.full:
            try:
                remote = self.remote_head_map = remote_heads()
            except ValidationFailure as exc:
                self.error(f"remote branch query failed: {exc}")
                return
        for experiment in experiments:
            if experiment.get("execution_state") != "completed":
                continue
            exp_id = experiment["id"]
            branches = experiment.get("source_branches", [])
            refs: list[str] = []
            for branch in branches:
                local_ref = f"refs/heads/{branch}"
                remote_ref = f"refs/remotes/origin/{branch}"
                selected_ref = local_ref if ref_commit(local_ref) else remote_ref
                if not ref_commit(selected_ref):
                    self.error(f"{exp_id} source branch is unavailable locally: {branch}")
                else:
                    refs.append(selected_ref)
                if remote is not None:
                    tip = remote.get(branch)
                    if tip is None:
                        self.error(f"{exp_id} source branch is absent from origin: {branch}")
                    elif ref_commit(remote_ref) != tip:
                        self.error(
                            f"{exp_id} remote-tracking ref is stale for {branch}; fetch required"
                        )
            for commit in experiment.get("source_commits", []):
                full_commit = resolve_commit(commit)
                if full_commit is None:
                    self.error(f"{exp_id} source commit cannot be resolved: {commit}")
                    continue
                if refs and not any(is_ancestor(full_commit, ref) for ref in refs):
                    self.error(f"{exp_id} source commit is not on a declared branch: {commit}")
                if remote is not None:
                    remote_refs = [
                        f"refs/remotes/origin/{branch}" for branch in branches
                        if remote.get(branch) is not None
                    ]
                    if remote_refs and not any(
                            is_ancestor(full_commit, ref) for ref in remote_refs):
                        self.error(
                            f"{exp_id} source commit is not reachable from origin: {commit}"
                        )

    def expected_tags(self) -> dict[str, str]:
        roles = self.status.get("roles", {})
        return {
            roles["archived_reference"]["tag"]:
                roles["archived_reference"]["source_commit"],
            roles["performance_comparator"]["tag"]:
                roles["performance_comparator"]["artifact_source_commit"],
            roles["selected_w4a8_baseline"]["tag"]:
                roles["selected_w4a8_baseline"]["artifact_source_commit"],
        }

    def check_role_tags(self) -> None:
        for tag, target in self.expected_tags().items():
            tag_type = git(["cat-file", "-t", f"refs/tags/{tag}"], check=False)
            if tag_type.returncode != 0:
                self.error(f"required annotated tag is missing: {tag}")
                continue
            if tag_type.stdout.strip() != "tag":
                self.error(f"required tag is not annotated: {tag}")
            actual = ref_commit(f"refs/tags/{tag}")
            expected = resolve_commit(target)
            if actual != expected:
                self.error(f"tag target mismatch: {tag}: {actual} != {expected}")
        if self.full:
            try:
                self.remote_tag_map = remote_tags()
            except ValidationFailure as exc:
                self.error(f"remote tag query failed: {exc}")
                return
            for tag, target in self.expected_tags().items():
                remote = self.remote_tag_map.get(tag)
                expected = resolve_commit(target)
                if remote is None:
                    self.error(f"required tag is absent from origin: {tag}")
                elif remote[1] is None:
                    self.error(f"origin tag is not annotated: {tag}")
                elif remote[1] != expected:
                    self.error(f"origin tag target mismatch: {tag}")

    def check_artifacts(self) -> None:
        artifacts = self.artifacts.get("artifacts", [])
        found = {role: set() for role in REQUIRED_ARTIFACT_ROLES}
        seen: set[tuple[str, str]] = set()
        for artifact in artifacts:
            artifact_role = artifact.get("artifact_role")
            roles = artifact.get("roles", [])
            for role in roles:
                key = (role, artifact_role)
                if key in seen:
                    self.error(f"duplicate formal artifact role: {role}/{artifact_role}")
                seen.add(key)
                if role in found:
                    found[role].add(artifact_role)
            try:
                path = windows_to_wsl(
                    artifact.get("root_windows", ""), artifact.get("relative_path", ""))
            except ValidationFailure as exc:
                self.error(str(exc))
                continue
            expected_area = (
                Path("/mnt/d/llm_exp/results")
                if artifact_role in RESULT_ARTIFACT_ROLES
                else Path("/mnt/d/llm_exp/models")
            ).resolve()
            try:
                path.relative_to(expected_area)
            except ValueError:
                self.error(f"{artifact_role} is outside {expected_area}: {path}")
            if artifact.get("required") and not path.is_file():
                self.error(f"required artifact is missing: {path}")
                continue
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size != artifact.get("size_bytes"):
                self.error(
                    f"artifact size mismatch: {path}: {size} != {artifact.get('size_bytes')}"
                )
                continue
            if self.full or size <= QUICK_HASH_LIMIT:
                if sha256_file(path) != artifact.get("sha256"):
                    self.error(f"artifact sha256 mismatch: {path}")
        for role, required_roles in REQUIRED_ARTIFACT_ROLES.items():
            missing = required_roles - found[role]
            if missing:
                self.error(f"{role} formal artifact profile missing: {sorted(missing)}")

    def check_remote_memory(self) -> None:
        if self.remote_head_map is None:
            try:
                self.remote_head_map = remote_heads()
            except ValidationFailure as exc:
                self.error(f"remote branch query failed: {exc}")
                return
        remote = self.remote_head_map.get(EXPECTED_MEMORY_BRANCH)
        if remote is None:
            self.error("project-memory branch is absent from origin")
            return
        try:
            local = git_head(ROOT)
        except ValidationFailure as exc:
            self.error(str(exc))
            return
        tracked = ref_commit(f"refs/remotes/origin/{EXPECTED_MEMORY_BRANCH}")
        if tracked != remote:
            self.error("project-memory remote-tracking ref is stale; fetch required")
        if local != remote:
            self.error(f"project-memory local HEAD {local} differs from origin {remote}")


def validate(*, full: bool = False,
             require_clean: bool = True) -> ProjectValidator:
    validator = ProjectValidator(full=full, require_clean=require_clean)
    validator.check()
    validator.raise_if_failed()
    return validator


def find_experiment(index: dict[str, Any], exp_id: str) -> dict[str, Any]:
    for experiment in index["experiments"]:
        if experiment["id"] == exp_id:
            return experiment
    raise ValidationFailure(f"unknown experiment: {exp_id}")


def validate_source_worktree(source: Path, *,
                             experiment: dict[str, Any] | None = None,
                             require_clean: bool = True) -> dict[str, str]:
    source = source.resolve()
    if not source.is_dir():
        raise ValidationFailure(f"source worktree does not exist: {source}")
    if source == ROOT:
        raise ValidationFailure("project-memory worktree cannot be a source worktree")
    status = load_yaml(STATUS_PATH)
    expected_common = Path(status["project"]["git_common_dir"]).resolve()
    actual_common = git_common_dir(source)
    if actual_common != expected_common:
        raise ValidationFailure(
            f"source worktree belongs to {actual_common}, expected {expected_common}"
        )
    if canonical_git_url(git_origin(source)) != canonical_git_url(EXPECTED_REPOSITORY):
        raise ValidationFailure("source worktree origin differs from the approved repository")
    if require_clean:
        dirty = git_dirty(source)
        if dirty:
            raise ValidationFailure(f"source worktree is dirty: {dirty[:5]}")
    branch = git_branch(source)
    head = git_head(source)
    clean_root = resolve_commit(EXPECTED_CLEAN_ROOM_ROOT)
    if clean_root is None or not is_ancestor(clean_root, head):
        raise ValidationFailure("source HEAD is outside the clean-room W4A16-derived lineage")
    if experiment and experiment.get("execution_state") == "running":
        runtime = experiment.get("runtime") or {}
        if Path(runtime.get("source_worktree", "")).resolve() != source:
            raise ValidationFailure("running experiment worktree does not match its runtime lock")
        if runtime.get("source_branch") != branch:
            raise ValidationFailure("running experiment branch does not match its runtime lock")
    return {"path": str(source), "branch": branch, "head": head}


def print_brief(source_worktree: Path | None = None) -> None:
    status = load_yaml(STATUS_PATH)
    index = load_yaml(INDEX_PATH)
    roles = status["roles"]
    active = status["governance"]["active_experiment"]
    print("PROJECT_MEMORY=verified")
    print(f"QAIRT={status['toolchain']['qairt_release']}")
    print(
        f"COMPARATOR={roles['performance_comparator']['tag']} "
        f"prefill={roles['performance_comparator']['prefill_tokens_per_second']:.3f} "
        f"decode={roles['performance_comparator']['decode_tokens_per_second']:.3f}"
    )
    print(
        f"BASELINE={roles['selected_w4a8_baseline']['tag']} "
        f"prefill={roles['selected_w4a8_baseline']['prefill_tokens_per_second']:.3f} "
        f"decode={roles['selected_w4a8_baseline']['decode_tokens_per_second']:.3f}"
    )
    print(f"ACTIVE_EXPERIMENT={active or 'none'}")
    rejected = [
        item["id"] for item in index["experiments"]
        if item["parent"] is None and item["adoption_status"] == "rejected"
    ]
    print(f"REJECTED_FAMILIES={','.join(rejected)}")
    if source_worktree is not None:
        info = validate_source_worktree(source_worktree, require_clean=False)
        print(f"SOURCE_WORKTREE={info['path']}")
        print(f"SOURCE_BRANCH={info['branch']}")
        print(f"SOURCE_HEAD={info['head']}")


def preflight(source_worktree: Path, exp_id: str | None) -> None:
    validate(full=False, require_clean=True)
    status = load_yaml(STATUS_PATH)
    index = load_yaml(INDEX_PATH)
    active = status["governance"]["active_experiment"]
    experiment = None
    if exp_id is not None:
        experiment = find_experiment(index, exp_id)
        state = experiment["execution_state"]
        if state not in {"approved", "running"}:
            raise ValidationFailure(f"{exp_id} is {state}, not approved or running")
        if state == "running" and active != exp_id:
            raise ValidationFailure(f"{exp_id} is running but does not own the active lock")
        if state == "approved" and active is not None:
            raise ValidationFailure(f"another experiment owns the active lock: {active}")
    elif active is not None:
        experiment = find_experiment(index, active)
    else:
        raise ValidationFailure("stateful preflight requires an approved Experiment ID")
    info = validate_source_worktree(
        source_worktree, experiment=experiment, require_clean=True)
    if experiment["execution_state"] == "approved":
        digits = experiment["id"].split("-")[1]
        expected = f"codex/exp-{digits}-{experiment['slug']}"
        print("PREFLIGHT=approved-before-branch")
        print(f"EXPECTED_BRANCH={expected}")
    else:
        print("PREFLIGHT=running")
    print(f"EXPERIMENT={experiment['id']}")
    print(f"SOURCE_BRANCH={info['branch']}")
    print(f"SOURCE_HEAD={info['head']}")


def begin_mutation() -> tuple[dict[str, Any], dict[str, Any]]:
    validate(full=False, require_clean=True)
    return load_yaml(STATUS_PATH), load_yaml(INDEX_PATH)


def command_propose(args: argparse.Namespace) -> None:
    status, index = begin_mutation()
    number = status["governance"]["next_experiment_number"]
    exp_id = f"EXP-{number:04d}"
    if any(item["slug"] == args.slug for item in index["experiments"]):
        raise ValidationFailure(f"experiment slug already exists: {args.slug}")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.slug):
        raise ValidationFailure("slug must be lowercase kebab-case")
    record = f"docs/experiments/{exp_id}.md"
    experiment = {
        "id": exp_id, "slug": args.slug, "title": args.title,
        "parent": args.parent, "variants": [],
        "execution_state": "proposed", "evidence_validity": "not_checked",
        "local_gate": "not_run", "adoption_status": "pending",
        "evidence_profile": args.evidence_profile,
        "source_branches": [f"codex/exp-{number:04d}-{args.slug}"],
        "source_commits": [], "record": record,
        "distinct_from": args.distinct_from or [], "hard_gate": args.gate,
        "reopen_condition": args.reopen_condition,
        "decided_by": "pending", "decided_on": dt.date.today().isoformat(),
    }
    if args.parent:
        parent = find_experiment(index, args.parent)
        if parent["adoption_status"] == "rejected":
            raise ValidationFailure(
                "rejected family reopening requires an approved contract amendment first"
            )
        parent["variants"].append(exp_id)
    index["experiments"].append(experiment)
    status["governance"]["next_experiment_number"] = number + 1
    status["project"]["updated_on"] = dt.date.today().isoformat()
    narrative = (
        f"# {exp_id} — {args.title}\n\n"
        f"## Hypothesis\n\n{args.hypothesis}\n\n"
        f"## Primary physical variable\n\n{args.variable}\n\n"
        f"## Hard gate\n\n{args.gate}\n\n"
        f"## Evidence profile\n\n{args.evidence_profile}\n\n"
        "## Outcome\n\nPending.\n"
    )
    atomic_text(ROOT / record, narrative)
    atomic_yaml(INDEX_PATH, index)
    atomic_yaml(STATUS_PATH, status)
    print(f"PROPOSED={exp_id}")
    print("NEXT=review, commit, push, then obtain explicit user approval")


def command_approve(args: argparse.Namespace) -> None:
    status, index = begin_mutation()
    experiment = find_experiment(index, args.experiment)
    if experiment["execution_state"] != "proposed":
        raise ValidationFailure(f"{args.experiment} is not proposed")
    if not args.approval_ref.strip():
        raise ValidationFailure("approval_ref must identify the explicit user approval")
    experiment["execution_state"] = "approved"
    experiment["approval_ref"] = args.approval_ref.strip()
    experiment["decided_by"] = "user"
    experiment["decided_on"] = dt.date.today().isoformat()
    status["project"]["updated_on"] = dt.date.today().isoformat()
    atomic_yaml(INDEX_PATH, index)
    atomic_yaml(STATUS_PATH, status)
    print(f"APPROVED={args.experiment}")
    print("NEXT=commit and push project-memory before source preflight")


def command_start(args: argparse.Namespace) -> None:
    status, index = begin_mutation()
    if status["governance"]["active_experiment"] is not None:
        raise ValidationFailure(
            f"active lock is held by {status['governance']['active_experiment']}"
        )
    experiment = find_experiment(index, args.experiment)
    if experiment["execution_state"] != "approved":
        raise ValidationFailure(f"{args.experiment} is not approved")
    info = validate_source_worktree(args.source_worktree, require_clean=True)
    expected = experiment["source_branches"][0]
    if info["branch"] != expected:
        raise ValidationFailure(
            f"source branch is {info['branch']}, expected {expected}"
        )
    experiment["execution_state"] = "running"
    experiment["runtime"] = {
        "source_worktree": info["path"], "source_branch": info["branch"],
        "started_on": dt.date.today().isoformat(),
    }
    status["governance"]["active_experiment"] = args.experiment
    status["project"]["updated_on"] = dt.date.today().isoformat()
    atomic_yaml(INDEX_PATH, index)
    atomic_yaml(STATUS_PATH, status)
    print(f"RUNNING={args.experiment}")
    print("NEXT=commit and push project-memory; then run preflight again")


def command_close(args: argparse.Namespace) -> None:
    status, index = begin_mutation()
    experiment = find_experiment(index, args.experiment)
    if experiment["execution_state"] != "running":
        raise ValidationFailure(f"{args.experiment} is not running")
    if status["governance"]["active_experiment"] != args.experiment:
        raise ValidationFailure("experiment does not own the active lock")
    info = validate_source_worktree(args.source_worktree, experiment=experiment)
    branch = experiment["runtime"]["source_branch"]
    remote = remote_heads()
    if branch not in remote:
        raise ValidationFailure(f"source branch is not on origin: {branch}")
    remote_ref = f"refs/remotes/origin/{branch}"
    if ref_commit(remote_ref) != remote[branch]:
        raise ValidationFailure(f"remote-tracking ref is stale for {branch}; fetch first")
    head = info["head"]
    if not is_ancestor(head, remote_ref):
        raise ValidationFailure("source HEAD is not reachable from the remote branch")
    experiment["execution_state"] = "completed"
    experiment["evidence_validity"] = args.evidence_validity
    experiment["local_gate"] = args.local_gate
    experiment["source_commits"] = list(dict.fromkeys(
        [*experiment["source_commits"], head]))
    experiment["evidence"] = args.evidence or []
    status["governance"]["active_experiment"] = None
    status["project"]["updated_on"] = dt.date.today().isoformat()
    atomic_yaml(INDEX_PATH, index)
    atomic_yaml(STATUS_PATH, status)
    print(f"COMPLETED={args.experiment}")
    print("ADOPTION=pending")


def command_decide(args: argparse.Namespace) -> None:
    status, index = begin_mutation()
    experiment = find_experiment(index, args.experiment)
    if experiment["execution_state"] != "completed":
        raise ValidationFailure("only completed experiments can receive a decision")
    if args.status == "accepted" and args.decided_by != "user":
        raise ValidationFailure("accepted adoption requires decided_by=user")
    if not args.decision_ref.strip():
        raise ValidationFailure("decision_ref must identify the explicit decision")
    experiment["adoption_status"] = args.status
    experiment["decided_by"] = args.decided_by
    experiment["decided_on"] = dt.date.today().isoformat()
    experiment["decision_ref"] = args.decision_ref.strip()
    status["project"]["updated_on"] = dt.date.today().isoformat()
    atomic_yaml(INDEX_PATH, index)
    atomic_yaml(STATUS_PATH, status)
    print(f"DECIDED={args.experiment}:{args.status}")
    if args.status == "accepted":
        print("NOTE=baseline promotion is a separate user-authorized status edit")


def command_abort(args: argparse.Namespace) -> None:
    status, index = begin_mutation()
    experiment = find_experiment(index, args.experiment)
    if experiment["execution_state"] not in {"proposed", "approved", "running"}:
        raise ValidationFailure(f"{args.experiment} cannot be aborted now")
    if experiment["execution_state"] == "running":
        if status["governance"]["active_experiment"] != args.experiment:
            raise ValidationFailure("running experiment does not own the active lock")
        status["governance"]["active_experiment"] = None
    experiment["execution_state"] = "aborted"
    experiment["abort_reason"] = args.reason
    experiment["adoption_status"] = "not_applicable"
    status["project"]["updated_on"] = dt.date.today().isoformat()
    atomic_yaml(INDEX_PATH, index)
    atomic_yaml(STATUS_PATH, status)
    print(f"ABORTED={args.experiment}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validate")
    validation.add_argument("--full", action="store_true")
    subparsers.add_parser("sync-refs")
    brief = subparsers.add_parser("brief")
    brief.add_argument("--source-worktree", type=Path)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--source-worktree", type=Path, required=True)
    pre.add_argument("--experiment")
    propose = subparsers.add_parser("propose")
    propose.add_argument("--slug", required=True)
    propose.add_argument("--title", required=True)
    propose.add_argument("--hypothesis", required=True)
    propose.add_argument("--variable", required=True)
    propose.add_argument("--gate", required=True)
    propose.add_argument(
        "--evidence-profile", required=True,
        choices=["micrograph-performance", "full-model-performance",
                 "correctness-only", "diagnostic"])
    propose.add_argument("--parent")
    propose.add_argument("--distinct-from", action="append")
    propose.add_argument("--reopen-condition", default="not_applicable")
    approve = subparsers.add_parser("approve")
    approve.add_argument("experiment")
    approve.add_argument("--approval-ref", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("experiment")
    start.add_argument("--source-worktree", type=Path, required=True)
    close = subparsers.add_parser("close")
    close.add_argument("experiment")
    close.add_argument("--source-worktree", type=Path, required=True)
    close.add_argument("--evidence-validity", choices=EVIDENCE_VALUES, required=True)
    close.add_argument("--local-gate", choices=GATE_VALUES, required=True)
    close.add_argument("--evidence", action="append")
    decide = subparsers.add_parser("decide")
    decide.add_argument("experiment")
    decide.add_argument(
        "--status", choices=["accepted", "rejected", "superseded", "not_applicable"],
        required=True)
    decide.add_argument("--decided-by", choices=["user", "codex"], required=True)
    decide.add_argument("--decision-ref", required=True)
    abort = subparsers.add_parser("abort")
    abort.add_argument("experiment")
    abort.add_argument("--reason", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "validate":
            if args.full:
                sync_required_refs()
            validate(full=args.full, require_clean=True)
            print(f"VALIDATION=pass mode={'full' if args.full else 'quick'}")
        elif args.command == "sync-refs":
            validate(full=False, require_clean=True)
            sync_required_refs()
            print("REFERENCE_SYNC=pass")
        elif args.command == "brief":
            validate(full=False, require_clean=True)
            print_brief(args.source_worktree)
        elif args.command == "preflight":
            preflight(args.source_worktree, args.experiment)
        elif args.command == "propose":
            command_propose(args)
        elif args.command == "approve":
            command_approve(args)
        elif args.command == "start":
            command_start(args)
        elif args.command == "close":
            command_close(args)
        elif args.command == "decide":
            command_decide(args)
        elif args.command == "abort":
            command_abort(args)
        else:
            raise AssertionError(args.command)
        return 0
    except (ValidationFailure, OSError, yaml.YAMLError,
            json.JSONDecodeError) as exc:
        print(f"HARD_STOP={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
