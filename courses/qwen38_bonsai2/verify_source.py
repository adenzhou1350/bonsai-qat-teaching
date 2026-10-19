"""Verify local weights against pinned public Hub LFS/Git object identities."""
import argparse
import hashlib
import json
from pathlib import Path


def check_file(path, record):
    path = Path(path)
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Source size mismatch: {path.name}")
    sha256 = hashlib.sha256()
    git = hashlib.sha1(("blob " + str(record["size"]) + "\0").encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha256.update(chunk)
            git.update(chunk)
    expected = record.get("lfs_sha256")
    valid = sha256.hexdigest() == expected if expected else git.hexdigest() == record["git_blob"]
    if not valid:
        raise ValueError(f"Source object mismatch: {path.name}")
    return {"name": path.name, "bytes": record["size"], "SHA256": sha256.hexdigest(),
            "primary_Hub_LFS_or_Git_blob_verified": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Choose a new verification output")
    spec = json.loads(Path(__file__).with_name("source-files.json").read_text(encoding="utf-8"))
    rows = []
    for item in spec["files"]:
        name = item["name"]
        if Path(name).name != name:
            raise ValueError("Expected a flat source-file name")
        path = (a.model / name).resolve()
        if not path.is_relative_to(a.model.resolve()):
            raise ValueError("Source file resolves outside the model directory")
        rows.append(check_file(path, item))
    result = {"revision": spec["revision"], "completed_files": rows, "passed": True,
              "requested_files": len(rows), "requested_bytes": sum(r["bytes"] for r in rows)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(rows)} pinned original files; no model or CUDA execution")


if __name__ == "__main__":
    main()
