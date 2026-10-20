"""Regression checks: damaged data must not retain a successful evidence verdict."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("q38_check_evidence", HERE / "check_evidence.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class EvidenceCorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = CHECK.read_json(ROOT, CHECK.EVIDENCE)
        CHECK.validate_record(cls.original, ROOT)

    def test_false_match_flags_do_not_hide_changed_outputs_or_inputs(self):
        for key, value in (("generated_token_IDs", 0), ("input_token_IDs", 0),
                           ("last_position_logit_SHA256", "0" * 64)):
            with self.subTest(field=key):
                data = copy.deepcopy(self.original)
                data["rounds"][2]["cases"][0][key][0] = value
                with self.assertRaisesRegex(CHECK.EvidenceError, "Reference mismatch"):
                    CHECK.validate_record(data, ROOT)

    def test_inflated_rate_or_incomplete_timing_is_rejected(self):
        data = copy.deepcopy(self.original)
        data["derived_summary"][2]["pooled_decode_tokens_per_second"] *= 2
        with self.assertRaisesRegex(CHECK.EvidenceError, "Arithmetic mismatch"):
            CHECK.validate_record(data, ROOT)
        data = copy.deepcopy(self.original)
        data["rounds"][2]["cases"][0]["step_seconds"].pop()
        with self.assertRaisesRegex(CHECK.EvidenceError, "count mismatch"):
            CHECK.validate_record(data, ROOT)

    def test_operator_and_quality_pass_claims_are_not_trusted(self):
        data = copy.deepcopy(self.original)
        data["operator_checks"]["cases"][0]["exact"] = False
        with self.assertRaisesRegex(CHECK.EvidenceError, "did not pass"):
            CHECK.validate_record(data, ROOT)
        data = copy.deepcopy(self.original)
        data["quality_accepted"] = True
        with self.assertRaisesRegex(CHECK.EvidenceError, "overall pass claim"):
            CHECK.validate_record(data, ROOT)

    def test_changed_source_and_escaping_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(ROOT / ".gitattributes", root / ".gitattributes")
            names = CHECK.SOURCE_NAMES | {"inference/source-files.json"}
            for name in names:
                target = root / CHECK.COURSE / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / CHECK.COURSE / name, target)
            target = root / CHECK.COURSE / "inference/benchmark.py"
            target.write_bytes(target.read_bytes() + b"\n# changed source\n")
            with self.assertRaisesRegex(CHECK.EvidenceError, "Source SHA256 mismatch"):
                CHECK.check_sources(root, self.original)
            with self.assertRaisesRegex(CHECK.EvidenceError, "Unsafe public path"):
                CHECK.safe_file(root, "../outside.py")


if __name__ == "__main__":
    unittest.main()
