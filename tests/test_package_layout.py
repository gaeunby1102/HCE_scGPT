"""Verify imports, resources, and output paths after installation."""

import ast
from importlib import resources, util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import HCE
from HCE import HCEPerturbationPredictor, HierarchicalPerturbationLoss, OntologyDAG
from HCE.ontology import load_ontology_from_json


class PackageLayoutTests(unittest.TestCase):
    def test_public_core_imports_are_preserved(self):
        self.assertEqual(HCEPerturbationPredictor.__module__, "HCE.model")
        self.assertEqual(HierarchicalPerturbationLoss.__module__, "HCE.loss")
        self.assertEqual(OntologyDAG.__module__, "HCE.ontology")

    def test_all_internal_import_targets_resolve(self):
        for source in Path(HCE.__file__).parent.rglob("*.py"):
            tree = ast.parse(source.read_text())
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names if alias.name.startswith("HCE."))
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("HCE."):
                    modules.append(node.module)
            for module in modules:
                with self.subTest(source=source.name, module=module):
                    self.assertIsNotNone(util.find_spec(module))

    def test_experiment_entry_points_are_packaged(self):
        for module in [
            "HCE.experiments.k562.demo", "HCE.experiments.k562.benchmark_ood",
            "HCE.experiments.k562.benchmark_full_go",
            "HCE.experiments.norman.gears_norman_baseline",
            "HCE.experiments.norman.gears_norman_hce",
            "HCE.experiments.norman.scgpt_norman_hce",
            "HCE.experiments.norman.scgpt_norman_finetune",
            "HCE.experiments.norman.scgpt_gears_hce",
            "HCE.experiments.norman.eval_by_split",
            "HCE.experiments.norman.interaction_analysis",
            "HCE.experiments.norman.deep_interaction_analysis",
            "HCE.experiments.brain.brain_insilico_perturb",
            "HCE.experiments.brain.step1_finetune_hce",
            "HCE.experiments.brain.step2_hce_jacobian",
            "HCE.experiments.brain.step3_visualize",
        ]:
            with self.subTest(module=module):
                self.assertIsNotNone(util.find_spec(module))

    def test_hallmark_resource_loads_without_network(self):
        resource = resources.files("HCE.ontologies").joinpath("hallmark_ontology.json")
        with resources.as_file(resource) as path:
            dag = load_ontology_from_json(str(path))
        self.assertGreater(len(dag), 0)
        self.assertEqual(len(dag.topological_sort()), len(dag))

    def test_output_paths_and_environment_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {key: value for key, value in os.environ.items() if not key.startswith("HCE_")}
            command = [sys.executable, "-c",
                       "import json; from HCE import config; "
                       "print(json.dumps([config.RESULTS_ROOT, config.JACOBIAN_RESULTS]))"]
            package = Path(HCE.__file__).resolve().parent
            source_root = package.parents[1]
            default_root = (source_root if (source_root / "pyproject.toml").is_file()
                            and (source_root / "src/HCE").is_dir() else Path(directory).resolve())
            result = subprocess.check_output(command, env=env, cwd=directory, text=True)
            self.assertEqual(json.loads(result), [str(default_root / "results"),
                                                 str(default_root / "jacobian/results")])
            env.update(HCE_RESULTS_ROOT=str(Path(directory) / "custom results"),
                       HCE_JACOBIAN_RESULTS=str(Path(directory) / "custom brain"))
            result = subprocess.check_output(command, env=env, cwd=directory, text=True)
            self.assertEqual(json.loads(result), [env["HCE_RESULTS_ROOT"], env["HCE_JACOBIAN_RESULTS"]])


if __name__ == "__main__":
    unittest.main()
