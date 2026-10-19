"""Bound pinned host storage to64MiB; masters and accumulated gradients stay pageable."""
import threading
import torch
from offload import StagedTernary
from rotation import rotate
from ternary import STE,quantize,unpack_grid

_lock=threading.Lock()
_buffer=None
BUFFER_ELEMENTS=16*1024*1024

def buffer():
    global _buffer
    if _buffer is None:
        # A cache first touched during inference must remain mutable for training.
        with torch.inference_mode(False),torch.no_grad():
            _buffer=torch.empty(BUFFER_ELEMENTS,dtype=torch.float32,pin_memory=True)
    return _buffer

class BufferedCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,master,device):
        if master.device.type!='cpu' or master.dtype!=torch.float32 or not master.is_contiguous() or device.type!='cuda':
            raise ValueError('Expected contiguous CPU FP32 master and one admitted CUDA device')
        gpu=torch.empty(master.shape,dtype=torch.float32,device=device)
        source=master.view(-1);target=gpu.view(-1)
        with _lock:
            pinned=buffer()
            for low in range(0,source.numel(),BUFFER_ELEMENTS):
                count=min(BUFFER_ELEMENTS,source.numel()-low)
                pinned[:count].copy_(source[low:low+count])
                target[low:low+count].copy_(pinned[:count],non_blocking=True)
                torch.cuda.current_stream(device).synchronize()
        return gpu

    @staticmethod
    def backward(ctx,gradient):
        host=torch.empty(gradient.shape,dtype=torch.float32,device='cpu')
        source=gradient.contiguous().view(-1);target=host.view(-1)
        with _lock:
            pinned=buffer()
            for low in range(0,source.numel(),BUFFER_ELEMENTS):
                count=min(BUFFER_ELEMENTS,source.numel()-low)
                pinned[:count].copy_(source[low:low+count],non_blocking=True)
                torch.cuda.current_stream(gradient.device).synchronize()
                target[low:low+count].copy_(pinned[:count])
        return host,None

class BufferedStagedTernary(StagedTernary):
    def forward(self,x):
        master=BufferedCopy.apply(self.master,x.device)
        code,scale=quantize(master,self.group)
        grid=unpack_grid(code,scale,self.group).to(self.compute_dtype)
        weight=STE.apply(master,grid)
        if self.embedding:
            return rotate(torch.nn.functional.embedding(x,weight),self.signs,self.block,inverse=True)
        return torch.nn.functional.linear(rotate(x,self.signs,self.block),weight)
