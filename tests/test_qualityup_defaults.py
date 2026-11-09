"""qualityup defaults: the LUT4 late-MLP plan and its generator, and the body-only drafter preferred by default."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hf_release  # noqa: E402
import make_lut4_plan as M  # noqa: E402


class PlanTests(unittest.TestCase):
    def test_shipped_plan_is_release_02_with_layers_24_to_63_mlp_at_lut4(self):
        base = json.loads((ROOT / "configs/quantization/mix25in_vq3pA.json").read_text())
        plan = json.loads((ROOT / "configs/quantization/mix25in_vq3pA_lut4L24-63.json").read_text())
        self.assertEqual(plan.keys(), base.keys())
        self.assertEqual(plan, M.make(base, range(24, 64), ["gate", "up", "down"]))
        for key, fmt in plan.items():
            layer, kind = key.split(".")
            late_mlp = kind in ("gate", "up", "down") and int(layer) >= 24
            self.assertEqual(fmt, M.LUT4 if late_mlp else base[key], key)

    def test_layer_spec(self):
        self.assertEqual(M.layers("32-34,60"), [32, 33, 34, 60])
        with self.assertRaises(SystemExit):
            M.layers("60-64")


class DrafterDefaultTests(unittest.TestCase):
    def test_body_only_drafter_preferred_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            build = bundle / "coreai"
            build.mkdir()
            legacy = bundle / "drafter" / hf_release.DEFAULT_DRAFTER
            legacy.mkdir(parents=True)
            self.assertEqual(hf_release.drafter_paths(build), (legacy, legacy.parent))
            body = bundle / "drafter" / hf_release.BODY_ONLY_DRAFTER
            body.mkdir(parents=True)
            self.assertEqual(hf_release.drafter_paths(build), (body, body.parent))
            self.assertEqual(hf_release.drafter_paths(build, legacy)[0], legacy.resolve())   # --draft still wins


if __name__ == "__main__":
    unittest.main()
