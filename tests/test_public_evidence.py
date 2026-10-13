"""Exercise the offline evidence checker against isolated damaged checkouts."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/check_public_evidence.py"
MANIFEST = "docs/public-evidence-manifest.json"


class PublicEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        manifest = json.loads((REPO / MANIFEST).read_text(encoding="utf-8"))
        files = {MANIFEST, *manifest["file_sha256"]}
        for entry in manifest["source_manifests"]:
            data = json.loads((REPO / entry["path"]).read_text(encoding="utf-8"))
            files.update((Path(entry["file_prefix"]) / name).as_posix()
                         for name in data["file_sha256"])
        for name in files:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)

    def run_checker(self):
        # Isolated mode avoids local PYTHONPATH; no Torch or model imports are needed.
        return subprocess.run([sys.executable, "-I", str(SCRIPT), "--root", str(self.root)],
                              capture_output=True, text=True, timeout=20)

    def test_public_snapshot(self):
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("244/256; 12 differ", result.stdout)
        self.assertIn("does not reproduce training", result.stdout)

    def test_modified_evidence_is_rejected(self):
        path = self.root / "docs/experiments/2026-10-10-1145-HTTP-observed-ABBA.json"
        path.write_bytes(path.read_bytes() + b" ")
        result = self.run_checker()
        self.assertEqual(result.returncode, 1)
        self.assertIn("SHA256 mismatch", result.stderr)
        self.assertNotIn("PASS:", result.stdout)

    def test_missing_source_is_rejected(self):
        (self.root / "quantization.py").unlink()
        result = self.run_checker()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing public file: quantization.py", result.stderr)

    def test_windows_line_endings_are_equivalent(self):
        path = self.root / "quantization.py"
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inconsistent_summary_arithmetic_is_rejected_even_with_matching_hash(self):
        # Update only this isolated fixture's digest, so this reaches the arithmetic check.
        manifest = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        name = manifest["records"]["http"]
        path = self.root / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["runs"][0]["client_wall_seconds"] *= 2
        content = (json.dumps(data) + "\n").encode()
        path.write_bytes(content)
        manifest["file_sha256"][name] = hashlib.sha256(content).hexdigest()
        (self.root / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        result = self.run_checker()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Summary arithmetic mismatch: original", result.stderr)


if __name__ == "__main__":
    unittest.main()
