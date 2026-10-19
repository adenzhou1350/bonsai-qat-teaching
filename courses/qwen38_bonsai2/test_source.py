"""Standard-library source-integrity controls; no PyTorch or model downloads."""
import hashlib
import tempfile
from pathlib import Path
import unittest
from verify_source import check_file


class SourceChecks(unittest.TestCase):
    def test_LFS_and_Git_blob_and_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fixture"
            content = b"authored source fixture\n"
            p.write_bytes(content)
            for identity in ({"lfs_sha256": hashlib.sha256(content).hexdigest()},
                             {"git_blob": hashlib.sha1(("blob " + str(len(content)) + "\0").encode() + content).hexdigest()}):
                record = dict(identity, size=len(content))
                self.assertTrue(check_file(p, record)["primary_Hub_LFS_or_Git_blob_verified"])
                p.write_bytes(b"X" + content[1:])
                with self.assertRaises(ValueError):
                    check_file(p, record)
                p.write_bytes(content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
