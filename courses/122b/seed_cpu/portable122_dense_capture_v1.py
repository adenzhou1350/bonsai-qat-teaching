"""Source-only GPTQ4 capture with independent FP32 and FP64 Hessian controls."""
import torch
from affine4_cpu import pack,unpack
from portable122_gptq4_v1 import GPTQWeightQuantizer

class Dense4Calibration:
    def __init__(self,layer):
        self.chosen={n:m for n,m in layer.named_modules() if isinstance(m,torch.nn.Linear) and m.weight.ndim==2 and m.weight.shape[1]%128==0 and m.weight.dtype==torch.bfloat16 and n not in ('mlp.gate','mlp.shared_expert_gate')}
        assert self.chosen
        self.fitters={n:GPTQWeightQuantizer(m.weight,nbits=4,groupsize=128,blocksize=128,percdamp=.01) for n,m in self.chosen.items()}
        self.grams={n:torch.zeros_like(f.H) for n,f in self.fitters.items()}
        self.sampled={n:torch.arange(0,f.columns,max(1,f.columns//16)) for n,f in self.fitters.items()}
        self.gram64={n:torch.zeros(len(ids),len(ids),dtype=torch.float64) for n,ids in self.sampled.items()}
        self.counts={n:0 for n in self.chosen};self.tokens={n:0 for n in self.chosen};self.enabled=False
        self.handles=[m.register_forward_hook(self.hook(n)) for n,m in self.chosen.items()]
    def hook(self,name):
        def capture(module,args,output):
            if not self.enabled:return
            x=args[0].detach().reshape(-1,module.in_features)
            with torch.autocast('cpu',enabled=False):
                self.fitters[name].add_batch(x[None]);self.grams[name].add_(2*x.float().t().matmul(x.float()))
                sample=x[:,self.sampled[name]].double();self.gram64[name].add_(2*sample.t().matmul(sample))
            self.counts[name]+=1;self.tokens[name]+=len(x)
        return capture
    def fit(self,on_event=lambda row:None):
        self.enabled=False
        for h in self.handles:h.remove()
        states={};metrics=[]
        for name,f in self.fitters.items():
            assert self.counts[name]==512 and self.tokens[name]==262144 and f.nsamples==512
            ref=self.grams[name]/512;relative=float((f.H-ref).norm()/ref.norm().clamp_min(1e-12));assert relative<3e-6
            idx=self.sampled[name];ref64=self.gram64[name]/512;relative64=float((f.H[idx[:,None],idx[None,:]].double()-ref64).norm()/ref64.norm().clamp_min(1e-12));assert relative64<3e-6
            codes,scales,zeros=f.quantize();state=pack(codes,scales,zeros,list(self.chosen[name].weight.shape))
            assert torch.equal(unpack(state),f.dequantize(codes,scales,zeros).reshape(state['shape']))
            assert torch.isfinite(unpack(state)).all();states[name]=state
            row={'module':name,'shape':state['shape'],'captured_train_blocks':512,'train_input_rows':262144,'hessian_relative_l2':relative,'hessian_fp64_subset_relative_l2':relative64,'codes_and_stored_hard_bitwise_equal':True}
            metrics.append(row);on_event(row)
        return states,metrics
