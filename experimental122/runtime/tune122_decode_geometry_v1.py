"""Change only launch geometry for three measured shapes; original math retained."""
def install():
 import hashlib,inspect
 import torch,triton
 import affine4_packed_runtime as affine
 import gsq_ternary_packed_runtime as ternary
 expected={affine:'5c458dd0a844aace47bfc29db524667bd46440038dfbdf501a5b92ffe2faeb02',ternary:'000219642e51d87168ab8f50891f9b0cbde93143cb4c5b3530abe086b599955f'}
 for module,h in expected.items():
  assert hashlib.sha256(open(inspect.getfile(module),'rb').read().replace(b'\r\n',b'\n')).hexdigest()==h
 assert not getattr(affine.PackedAffine4Linear,'_geometry_tuned',False)
 original_affine=affine.PackedAffine4Linear.dequant;original_ternary=ternary.TritBank.weights
 def dequant(self):
  n,k=self.out_features,self.in_features
  if (n,k)!=(16384,3072):return original_affine(self)
  weight=torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
  affine._decode[(triton.cdiv(n,4),triton.cdiv(k,512))](self.codes,self.scales,self.zeros,weight,n,k,4,512,enable_fp_fusion=False,num_warps=8)
  return weight
 def weights(self,ids):
  e,n,k=self.shape
  config={(2048,3072):(8,512,8),(3072,1024):(8,256,4)}.get((n,k))
  if len(ids)!=8 or config is None:return original_ternary(self,ids)
  bn,bk,w=config;weight=torch.empty((len(ids),n,k),device='cuda',dtype=torch.bfloat16)
  ternary._decode[(triton.cdiv(n,bn),triton.cdiv(k,bk),len(ids))](self.codes,self.scales,ids.contiguous(),weight,n,k,triton.cdiv(k,5),bn,bk,num_warps=w,enable_fp_fusion=False)
  return weight
 affine.PackedAffine4Linear.dequant=dequant;ternary.TritBank.weights=weights
 affine.PackedAffine4Linear._geometry_tuned=True
