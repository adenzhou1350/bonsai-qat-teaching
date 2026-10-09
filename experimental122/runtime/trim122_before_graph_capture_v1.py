"""Release unused prefill allocator blocks before decode snapshots and capture."""
import ast,hashlib,inspect,textwrap,time
import torch
from gsq_graph_decoder_v3_cache_snapshot import GraphDecoder

EVENTS=[]

def trim_before_capture():
 assert not torch.is_grad_enabled()
 torch.cuda.synchronize();began=time.perf_counter();allocated=torch.cuda.memory_allocated()
 before=torch.cuda.memory_reserved();free,total=torch.cuda.mem_get_info()
 torch.cuda.empty_cache()
 after=torch.cuda.memory_reserved();free_after,_=torch.cuda.mem_get_info()
 assert torch.cuda.memory_allocated()==allocated and after<=before
 EVENTS.append({'allocated_bytes_unchanged':allocated,'reserved_before_bytes':before,'reserved_after_bytes':after,'device_used_before_bytes':total-free,'device_used_after_bytes':total-free_after,'seconds':time.perf_counter()-began})

def bind_trim_before_capture():
 original=GraphDecoder.prepare;source=inspect.getsource(original)
 tree=ast.parse(textwrap.dedent(source));body=tree.body[0].body
 at=[i for i,n in enumerate(body) if isinstance(n,ast.Delete) and [ast.unparse(v) for v in n.targets]==['output']]
 assert len(at)==1
 assert any(isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='snapshots' for v in n.targets) for n in body[at[0]+1:])
 body.insert(at[0]+1,ast.parse('trim_before_capture()').body[0]);ast.fix_missing_locations(tree)
 namespace=dict(original.__globals__);namespace['trim_before_capture']=trim_before_capture
 exec(compile(tree,__file__,'exec'),namespace);GraphDecoder.prepare=namespace['prepare']
 return hashlib.sha256(source.strip().encode()).hexdigest()
