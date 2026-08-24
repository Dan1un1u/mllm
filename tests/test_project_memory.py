from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "project_memory.py"
SPEC = importlib.util.spec_from_file_location("project_memory", MODULE_PATH)
assert SPEC and SPEC.loader
pm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pm)


class PathMappingTests(unittest.TestCase):
    def test_windows_d_path_maps_inside_llm_exp(self) -> None:
        path = pm.windows_to_wsl(
            r"D:\llm_exp\models\example", "context.bin"
        )
        self.assertEqual(path, Path("/mnt/d/llm_exp/models/example/context.bin"))

    def test_windows_path_escape_is_rejected(self) -> None:
        with self.assertRaises(pm.ValidationFailure):
            pm.windows_to_wsl(
                r"D:\llm_exp\models\example", "../outside.bin"
            )

    def test_non_d_drive_is_rejected(self) -> None:
        with self.assertRaises(pm.ValidationFailure):
            pm.windows_to_wsl(
                r"C:\llm_exp\models\example", "context.bin"
            )


class IndexInvariantTests(unittest.TestCase):
    def validator(self):
        validator = pm.ProjectValidator(full=False, require_clean=False)
        validator.index = copy.deepcopy(validator.index)
        validator.status = copy.deepcopy(validator.status)
        validator.check_source_references = lambda experiments: None
        return validator

    def test_duplicate_id_is_rejected(self) -> None:
        validator = self.validator()
        validator.index["experiments"].append(
            copy.deepcopy(validator.index["experiments"][0])
        )
        validator.check_experiments()
        self.assertTrue(any("duplicate experiment id" in item
                            for item in validator.errors))

    def test_multiple_running_experiments_are_rejected(self) -> None:
        validator = self.validator()
        for experiment in validator.index["experiments"][:2]:
            experiment["execution_state"] = "running"
        validator.status["governance"]["active_experiment"] = (
            validator.index["experiments"][0]["id"]
        )
        validator.check_experiments()
        self.assertTrue(any("more than one running experiment" in item
                            for item in validator.errors))

    def test_parent_child_mismatch_is_rejected(self) -> None:
        validator = self.validator()
        child = next(item for item in validator.index["experiments"]
                     if item["id"] == "EXP-0009-A")
        child["parent"] = "EXP-0008"
        validator.check_experiments()
        self.assertTrue(any("parent-child mismatch" in item
                            or "absent from" in item
                            for item in validator.errors))

    def test_accepted_requires_user(self) -> None:
        validator = self.validator()
        accepted = next(item for item in validator.index["experiments"]
                        if item["adoption_status"] == "accepted")
        accepted["decided_by"] = "codex"
        validator.check_experiments()
        self.assertTrue(any("accepted without user decision" in item
                            for item in validator.errors))


class SchemaTests(unittest.TestCase):
    def test_missing_required_experiment_field_is_rejected(self) -> None:
        index = pm.load_yaml(pm.INDEX_PATH)
        broken = copy.deepcopy(index)
        del broken["experiments"][0]["local_gate"]
        errors = pm.schema_errors(
            broken, pm.ROOT / "schemas" / "experiment-index.schema.json"
        )
        self.assertTrue(any("local_gate" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
