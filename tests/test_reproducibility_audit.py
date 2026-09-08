"""Focused stdlib tests: python -m unittest discover -s tests -p test_reproducibility_audit.py."""
import importlib.util
import ast
import re
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "reproducibility/audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_tuning_storage_comes_from_environment_without_importing_scripts(self):
        root = Path(__file__).resolve().parents[1]
        for seed in range(5):
            script = root / "training/molclr/data/perovskite-resplit" / f"split_seed_{seed}" / "hyper_param_tune.py"
            tree = ast.parse(script.read_text(encoding="utf-8"))
            uses_environment = any(
                    isinstance(expression, ast.Subscript)
                    and
                    isinstance(expression.value, ast.Attribute)
                    and isinstance(expression.value.value, ast.Name)
                    and expression.value.value.id == "os"
                    and expression.value.attr == "environ"
                    and isinstance(expression.slice, ast.Constant)
                    and expression.slice.value == "OPTUNA_STORAGE_URL"
                for expression in ast.walk(tree))
            self.assertTrue(uses_environment, f"Seed {seed}: storage must use OPTUNA_STORAGE_URL")
            has_embedded_credentials = any(
                isinstance(node, ast.Constant) and isinstance(node.value, str)
                and re.search(r"(?:postgres(?:ql)?|mysql|mariadb|mongodb|redis)(?:\+\w+)?://[^@\s]+@", node.value)
                for node in ast.walk(tree))
            self.assertFalse(has_embedded_credentials, f"Seed {seed}: embedded database credentials are not allowed")

    def test_cleaning_keeps_duplicate_observations_in_order(self):
        retained = {"SMILES": "B", "TARGET": "1.00"}
        rows = [retained, {"SMILES": "A"}, retained.copy(), {"SMILES": "C"}]
        self.assertEqual(audit.cleaned_test([{"SMILES": "A"}], rows), [retained, retained, {"SMILES": "C"}])

    def test_folds_reject_invalid_indices_and_coverage(self):
        good = {"train_indices": [[1], [0]], "valid_indices": [[0], [1]]}
        self.assertEqual(len(audit.validate_folds(good, 2, 2)), 2)
        for bad in ([2], [True], [0, 0], [], [0.0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                audit.validate_folds({"train_indices": [[1], [0]], "valid_indices": [bad, [1]]}, 2, 2)
        with self.assertRaises(ValueError):
            audit.validate_folds({"train_indices": [[1], [1]], "valid_indices": [[0], [0]]}, 2, 2)

    def test_metrics_and_constant_target(self):
        result = audit.metrics([{"true": "1", "pred": "2"}, {"true": "3", "pred": "3"}])
        self.assertAlmostEqual(result["rmse"], math.sqrt(0.5))
        self.assertEqual(result["mae"], 0.5)
        self.assertEqual(result["r2"], 0.5)
        self.assertEqual(audit.metrics([{"true": "1", "pred": "2"}])["r2"], "")
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(ValueError):
                audit.metrics([{"true": value, "pred": "1"}])
            with self.assertRaises(ValueError):
                audit.metrics([{"true": "1", "pred": value}])

    def test_real_inventory_deterministic_and_snapshot_check(self):
        first = audit.build_outputs()
        self.assertEqual(first, audit.build_outputs())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, data in first.items():
                (output / name).write_bytes(data)
            self.assertEqual(audit.check_outputs(output, first), [])
            (output / "samples.csv").write_bytes(first["samples.csv"].replace(b"\n", b"\r\n"))
            self.assertEqual(audit.check_outputs(output, first), [])
            (output / "summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(audit.check_outputs(output, first), ["summary.json"])


if __name__ == "__main__":
    unittest.main()
