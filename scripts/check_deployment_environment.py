"""Report recorded inference dependencies and NVML health without loading weights."""
import argparse
import importlib.metadata
import json
import platform
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('35b', '122b', '27b'), required=True)
    a = p.parse_args()
    expected = {'torch': '2.11.0', 'transformers': '5.12.1', 'triton': '3.6.0',
                'safetensors': '0.8.0', 'numpy': '2.2.6'}
    if a.model != '27b': expected['causal-conv1d'] = '1.7.0'
    packages = {}
    for name, version in expected.items():
        try: actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: actual = None
        packages[name] = {'recorded': version, 'installed': actual,
                          'version_matches': actual is not None and actual.split('+')[0] == version}
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
        nvml = {'exit': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        nvml = {'exit': None, 'error_type': type(e).__name__}
    v = {'model': a.model, 'system': platform.system(), 'python': platform.python_version(),
         'packages': packages, 'NVML': nvml, 'weights_loaded': False, 'CUDA_initialized_by_checker': False,
         'ready_for_GPU_admission_check': platform.system() == 'Linux' and nvml['exit'] == 0 and all(x['version_matches'] for x in packages.values()),
         'limitation': 'Version/NVML checks do not reserve a GPU, prove extension ABI, or establish exact numerical behavior.'}
    print(json.dumps(v, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
