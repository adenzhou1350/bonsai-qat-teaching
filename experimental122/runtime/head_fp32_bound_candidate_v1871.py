"""Private experiment: preserve packed BF16 operands, return unrounded FP32 head logits."""
import hashlib,inspect,json,os,types
from pathlib import Path
import torch
from primary_gsq122_dense4_runtime_v1 import PackedAffine4NativeChunkedHead
from affine4_packed_runtime import _decode
_original=PackedAffine4NativeChunkedHead.forward
_observed={}
_binding_proof={}
_installed=False
def forward_fp32(self,x):
 import triton
 assert x.dtype==torch.bfloat16 and x.is_cuda and not self.has_bias
 shape=x.shape[:-1];x=x.reshape(-1,self.in_features).contiguous()
 output=torch.empty((len(x),self.out_features),device=x.device,dtype=torch.float32)
 for start in range(0,self.out_features,2048):
  stop=min(start+2048,self.out_features);n=stop-start;k=self.in_features
  weight=torch.empty((n,k),device=x.device,dtype=torch.bfloat16)
  _decode[(triton.cdiv(n,4),triton.cdiv(k,256))](self.codes[start:stop],self.scales[start:stop],self.zeros[start:stop],weight,n,k,4,256,enable_fp_fusion=False,num_warps=4)
  if 2<=len(x)<=4:
   for i in range(len(x)):output[i:i+1,start:stop]=torch.mm(x[i:i+1],weight.t(),out_dtype=torch.float32)
  else:output[:,start:stop]=torch.mm(x,weight.t(),out_dtype=torch.float32)
  del weight
 result=output.reshape(*shape,self.out_features)
 assert result.dtype==torch.float32 and result.is_cuda
 path=os.environ.get('BONSAI_HEAD_CALL_PROOF')
 if path and not torch.cuda.is_current_stream_capturing():
  key=str(len(x))
  if key not in _observed:
   _observed[key]={'input_rows':len(x),'input_dtype':str(x.dtype),'actual_output_dtype':str(result.dtype),'actual_output_is_CUDA':result.is_cuda,'candidate_function_executed':True,'current_outer_forward_name':getattr(self.forward,'__func__',None).__name__,'shape':list(result.shape)}
   target=Path(path);tmp=target.with_suffix('.partial');tmp.write_text(json.dumps({'actual_GPU_calls':_observed,'binding_proof':_binding_proof},indent=2));tmp.replace(target)
 return result
def install():
 global _installed
 if _installed:return
 assert hashlib.sha256(Path(inspect.getfile(PackedAffine4NativeChunkedHead)).read_bytes()).hexdigest()=='2ef293421af594782bb2fa86b5a4e5be930bad0138b24109365b6c64e223fbfa'
 import bonsai_native122_vllm_request_prefill_v1662 as engine
 assert hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()=='934ddd3acbbf05c96fab62199d7ac6604c5f2d71a1effec0c056feab0743a7e2'
 original_bind=engine.bind_vllm_small_decode
 def bind(model):
  counts=original_bind(model)
  heads=[m for m in model.modules() if isinstance(m,PackedAffine4NativeChunkedHead)]
  assert len(heads)==1
  head=heads[0]
  _binding_proof.update({'after_original_small_decode_binding':True,'heads':1,'previous_instance_function':head.forward.__func__.__name__,'original_body_retained_on_disk':True,'candidate_function':'forward_fp32','persistent_weights_unchanged':True})
  head.forward=types.MethodType(forward_fp32,head)
  return counts
 engine.bind_vllm_small_decode=bind
 assert PackedAffine4NativeChunkedHead.forward is _original
 PackedAffine4NativeChunkedHead.forward=forward_fp32
 _installed=True
