"""Portable launcher for the unchanged operator controls."""
import argparse
import json
import time
from pathlib import Path
from common import save_new, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve earlier evidence")
    verify_sources()
    from test_candidate import run
    start = time.monotonic()
    result = run(args.device)
    result["seconds"] = time.monotonic() - start
    save_new(args.output, result)
    print(json.dumps({
        "passed": result["passed"], "cases": len(result["cases"]),
        "seconds": result["seconds"], "GPU_used": args.device == "cuda",
    }))


if __name__ == "__main__":
    main()
