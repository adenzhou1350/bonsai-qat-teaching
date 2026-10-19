"""Linux local-file FP32 gradients; keep the optimizer and master precision unchanged."""
import json
import mmap
import os
import shutil
from pathlib import Path
import torch

def memory_admission(elements, largest_elements):
    root = Path('/sys/fs/cgroup')
    maximum = (root / 'memory.max').read_text().strip()
    stats = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
    existing = int(stats['anon']) + int(stats.get('shmem', 0))
    # Masters remain anonymous. Gradients are regular local-file pages, reclaimable
    # by the kernel. Budget four largest FP32 work tensors plus36GiB of reserve.
    needed = elements * 4 + largest_elements * 16 + 36 * 2**30
    if maximum == 'max' or existing + needed >= int(maximum):
        raise RuntimeError('Known cgroup budget must admit masters, largest work tensors and reserves')
    return {'cgroup_limit_bytes': int(maximum), 'existing_anon_and_shmem_bytes': existing,
            'planned_anonymous_master_scratch_and_reserve_bytes': needed,
            'FP32_gradient_local_file_bytes': elements * 4,
            'optimizer': 'Adafactor_beta1None_relativeFalse_scaleFalse',
            'gradient_storage': 'shared_regular_local_file_mmap_FP32'}

class MappedGradients:
    """Attach preallocated contiguous gradients before backward; never update during backward."""
    def __init__(self, named, directory, reserve_bytes=20 * 2**30):
        self.named = list(named)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        if not self.named or len({name for name, _ in self.named}) != len(self.named):
            raise ValueError('Expected unique nonempty parameter names')
        required = sum(p.numel() * 4 for _, p in self.named)
        if shutil.disk_usage(self.directory).free < required + reserve_bytes:
            raise RuntimeError('Local disk must admit all gradient files and reserve')
        if not hasattr(mmap, 'MADV_REMOVE'):
            raise RuntimeError('Linux shared-file MADV_REMOVE support is required')
        self.entries = []
        self.handles = []
        self.touched = set()
        self.closed = False
        self.total_bytes = required
        for index, (name, param) in enumerate(self.named):
            if not param.is_leaf or not param.requires_grad or param.device.type != 'cpu' or param.dtype != torch.float32 or not param.is_contiguous() or param.numel() == 0 or param.grad is not None:
                raise ValueError('Expected contiguous CPU FP32 leaf masters without existing gradients')
            file = self.directory / f'{index:03d}.f32'
            with file.open('xb') as stream:
                stream.truncate(param.numel() * 4)
            with file.open('r+b') as stream:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE)
            tensor = torch.frombuffer(mapping, dtype=torch.float32).reshape(param.shape)
            param.grad = tensor
            entry = {'name': name, 'parameter': param, 'tensor': tensor, 'mapping': mapping, 'pointer': tensor.data_ptr(), 'file': file}
            self.entries.append(entry)
            self.handles.append(param.register_post_accumulate_grad_hook(self._hook(entry)))
        self.zero_grad()
        (self.directory / 'gradient-layout.json').write_text(json.dumps({'format': 'qwen38-local-file-FP32-gradients-v1', 'matrices': len(self.entries), 'logical_file_bytes': required, 'scratch_only_not_a_checkpoint': True}, indent=2))

    def _hook(self, entry):
        def accumulated(param):
            if param.grad is None or param.grad.data_ptr() != entry['pointer']:
                raise RuntimeError('Autograd replaced the preallocated file gradient')
            self.touched.add(entry['name'])
        return accumulated

    def assert_complete(self):
        if self.closed or self.touched != {name for name, _ in self.named}:
            raise RuntimeError('Every master must actually receive a backward accumulation')
        for entry in self.entries:
            if entry['parameter'].grad is None or entry['parameter'].grad.data_ptr() != entry['pointer']:
                raise RuntimeError('Gradient storage identity changed')

    def zero_grad(self):
        if self.closed:
            raise RuntimeError('Gradient store is closed')
        for entry in self.entries:
            param = entry['parameter']
            if param.grad is None or param.grad.data_ptr() != entry['pointer']:
                raise RuntimeError('Use MappedGradients.zero_grad instead of optimizer.zero_grad')
            # Hole-punch only our newly created gradient scratch files. Subsequent
            # reads fault in zero pages; this avoids writing107GB of zero values.
            entry['mapping'].madvise(mmap.MADV_REMOVE)
        self.touched.clear()

    def close(self):
        if self.closed:
            return
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for entry in self.entries:
            entry['parameter'].grad = None
            entry.pop('tensor')
            entry['mapping'].close()
        self.entries.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
