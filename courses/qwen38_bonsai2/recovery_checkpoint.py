"""Stream lossless FP32 masters to gzip; save optimizer/RNG separately and commit a manifest."""
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
import torch
import numpy as np

CHUNK=8*1024*1024

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while value:=stream.read(CHUNK):h.update(value)
    return h.hexdigest()

class BudgetWriter:
    def __init__(self,raw,directory,total,maximum,reserve):
        self.raw,self.directory,self.total,self.maximum,self.reserve=raw,directory,total,maximum,reserve
    def write(self,data):
        if self.total[0]+len(data)>self.maximum or shutil.disk_usage(self.directory).free-len(data)<self.reserve:
            raise RuntimeError('Checkpoint storage cap or free-space reserve reached; preserve partial files')
        n=self.raw.write(data);self.total[0]+=n;return n
    def flush(self):self.raw.flush()

def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def masters(model):
    result={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not result or any(p.device.type!='cpu' or p.dtype!=torch.float32 or not p.is_contiguous() for p in result.values()):
        raise ValueError('Checkpoint expects contiguous CPU FP32 trainable masters')
    return result

@torch.no_grad()
def save_checkpoint(model,optimizer,directory,metadata,maximum_bytes=80_000_000_000,reserve_bytes=12*2**30,base_factory=None):
    if sys.byteorder!='little':raise ValueError('The checkpoint byte format is little endian')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    total=[0];rows=[]
    for index,(name,param) in enumerate(masters(model).items()):
        path=directory/f'{index:03d}.f32.gz';partial=path.with_suffix(path.suffix+'.partial')
        view=memoryview(param.detach().numpy()).cast('B');raw_hash=hashlib.sha256()
        base=base_factory(name,param) if base_factory is not None else None
        if base is not None:
            if base.device.type!='cpu' or base.dtype!=torch.float32 or base.shape!=param.shape or not base.is_contiguous():
                raise ValueError('Original master basis differs from the checkpoint tensor')
            base_view=memoryview(base.detach().numpy()).cast('B');base_hash=hashlib.sha256()
        with partial.open('xb') as raw:
            guard=BudgetWriter(raw,directory,total,maximum_bytes,reserve_bytes)
            with gzip.GzipFile(filename='',fileobj=guard,mode='wb',compresslevel=1,mtime=0) as compressed:
                for low in range(0,len(view),CHUNK):
                    piece=view[low:low+CHUNK];raw_hash.update(piece)
                    if base is None:compressed.write(piece)
                    else:
                        base_hash.update(base_view[low:low+CHUNK])
                        compressed.write(np.bitwise_xor(np.frombuffer(piece,dtype=np.uint8),np.frombuffer(base_view[low:low+CHUNK],dtype=np.uint8)))
            raw.flush();os.fsync(raw.fileno())
        partial.rename(path)
        rows.append({'name':name,'file':path.name,'shape':list(param.shape),'dtype':'float32',
                     'encoding':'raw' if base is None else 'xor-with-original-FP32-master',
                     'base_raw_SHA256':None if base is None else base_hash.hexdigest(),
                     'raw_bytes':len(view),'raw_SHA256':raw_hash.hexdigest(),'SHA256':sha(path),'file_bytes':path.stat().st_size})
        if base is not None:del base_view,base
    state={'optimizer':optimizer.state_dict(),'torch_RNG':torch.get_rng_state(),
           'python_RNG':random.getstate(),'CUDA_RNG':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    path=directory/'state.pt';partial=directory/'state.pt.partial'
    with partial.open('xb') as raw:
        torch.save(state,BudgetWriter(raw,directory,total,maximum_bytes,reserve_bytes));raw.flush();os.fsync(raw.fileno())
    partial.rename(path)
    manifest={'format':'qwen38-CPU-master-gzip-v2','complete':True,'byteorder':'little','masters':rows,
              'optimizer_type':type(optimizer).__name__,'state_file':'state.pt','state_SHA256':sha(path),
              'metadata':metadata,'written_tensor_and_state_bytes':total[0]}
    temporary=directory/'manifest.json.partial'
    with temporary.open('x') as stream:
        json.dump(manifest,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.rename(directory/'manifest.json');sync_directory(directory)
    return manifest

@torch.no_grad()
def load_checkpoint(model,optimizer,directory,expected_metadata):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('format')!='qwen38-CPU-master-gzip-v2' or not manifest.get('complete') or manifest.get('byteorder')!=sys.byteorder:
        raise ValueError('Invalid or incomplete checkpoint format')
    if manifest['metadata']!=expected_metadata or manifest['optimizer_type']!=type(optimizer).__name__:
        raise ValueError('Checkpoint data/source/settings identity mismatch')
    named=masters(model)
    if list(named)!=[r['name'] for r in manifest['masters']]:raise ValueError('Master order/name mismatch')
    for row in manifest['masters']:
        path=directory/row['file']
        if path.parent!=directory or path.name!=row['file'] or sha(path)!=row['SHA256']:
            raise ValueError('Checkpoint tensor file identity mismatch')
        if list(named[row['name']].shape)!=row['shape']:raise ValueError('Master shape mismatch')
        if row['encoding'] not in ('raw','xor-with-original-FP32-master'):raise ValueError('Unknown master encoding')
        if row['encoding']=='xor-with-original-FP32-master':
            view=memoryview(named[row['name']].detach().numpy()).cast('B')
            if hashlib.sha256(view).hexdigest()!=row['base_raw_SHA256']:
                raise ValueError('Reconstruct the exact original rotated master before applying the lossless XOR')
    state_path=directory/manifest['state_file']
    if state_path.parent!=directory or state_path.name!=manifest['state_file'] or sha(state_path)!=manifest['state_SHA256']:
        raise ValueError('Optimizer/RNG identity mismatch')
    # Validate every compressed object before changing any model parameter.
    for row in manifest['masters']:
        param=named[row['name']];view=memoryview(param.detach().numpy()).cast('B');h=hashlib.sha256()
        if len(view)!=row['raw_bytes']:raise ValueError('Raw tensor byte count mismatch')
        with gzip.open(directory/row['file'],'rb') as stream:
            low=0
            while low<len(view):
                if row['encoding']=='raw':size=stream.readinto(view[low:low+CHUNK])
                else:
                    delta=stream.read(CHUNK);size=len(delta)
                    if size:
                        target=np.frombuffer(view[low:low+size],dtype=np.uint8)
                        np.bitwise_xor(target,np.frombuffer(delta,dtype=np.uint8),out=target)
                if not size:raise ValueError('Truncated FP32 master')
                h.update(view[low:low+size]);low+=size
            if stream.read(1):raise ValueError('Unexpected trailing tensor bytes')
        if h.hexdigest()!=row['raw_SHA256']:raise ValueError('Decoded master identity mismatch')
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_RNG']);random.setstate(state['python_RNG'])
    if state['CUDA_RNG'] is not None:
        if len(state['CUDA_RNG'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(state['CUDA_RNG'])
    return manifest
