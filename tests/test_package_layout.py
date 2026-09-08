"""Check that the shared harness works through the installable package layout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"


class PackageLayoutTest(unittest.TestCase):
    def test_harness_public_api_imports_without_site_packages_or_repository_cwd(self):
        # -I -S removes PYTHONPATH, the current directory and installed ML libraries.
        script = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import shopping_grpo.harness as harness
assert 'shopping_grpo.harness.sft_pipeline' not in sys.modules
assert harness.EpisodeRequest(task_id=7).task_id == 7
assert harness.HarnessConfig(max_steps=35).max_steps == 35
assert callable(harness.EpisodeRunner)
assert callable(harness.evaluate_trajectory)
assert not {'torch', 'transformers', 'verl', 'fcntl'} & set(sys.modules)
print(json.dumps({'origin': str(Path(harness.__file__).resolve())}))
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", script, str(SOURCE)],
                cwd=directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
        origin = Path(json.loads(result.stdout)["origin"])
        self.assertTrue(origin.is_relative_to(SOURCE.resolve()))

    def test_every_lazy_export_module_is_available_inside_the_source_package(self):
        import shopping_grpo.harness as harness

        for public_name, (module_name, _) in harness._EXPORTS.items():
            with self.subTest(export=public_name):
                spec = importlib.util.find_spec(module_name)
                self.assertIsNotNone(spec, module_name)
                self.assertIsNotNone(spec.origin, module_name)
                self.assertTrue(Path(spec.origin).resolve().is_relative_to(SOURCE.resolve()))

    def test_grpo_bridge_resolves_default_registry_without_external_training_runtime(self):
        from shopping_grpo.harness import SHOPPING_TOOL_REGISTRY, VerlHarnessBridge

        tools = VerlHarnessBridge().to_verl_tool_config()["tools"]
        actual_names = {entry["tool_schema"]["function"]["name"] for entry in tools}
        expected_names = {schema["function"]["name"] for schema in SHOPPING_TOOL_REGISTRY.schemas}
        self.assertEqual(actual_names, expected_names)
        self.assertTrue(actual_names)

    def test_default_sft_adapter_loads_and_excludes_canonical_evaluation_tasks(self):
        from shopping_grpo.harness.stage_sft import (
            SFTDataLeakError,
            SFTStageAdapter,
            build_sft_row,
        )

        canonical_path = REPOSITORY / "data" / "evaluation" / "tasks.jsonl"
        canonical_ids = {
            int(json.loads(line)["task_id"])
            for line in canonical_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        adapter = SFTStageAdapter()
        self.assertEqual(adapter.held_out_task_ids, canonical_ids)
        self.assertTrue(canonical_ids)
        with self.assertRaises(SFTDataLeakError):
            build_sft_row({"task_id": min(canonical_ids)})


if __name__ == "__main__":
    unittest.main()
