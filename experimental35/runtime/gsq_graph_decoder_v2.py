"""Reusable single-request graph decoder with fully GPU-resident static cache."""
import torch
from transformers import StaticCache

class GraphDecoder:
    def __init__(self,model,prototype,max_cache_len=8192):
        self.model=model;self.max_cache_len=max_cache_len;self.cache=StaticCache(model.config,max_cache_len=max_cache_len)
        self.token=torch.zeros((1,1),device='cuda',dtype=torch.long);self.position=torch.zeros((4,1,1),device='cuda',dtype=torch.long)
        self.prepare(prototype)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for i in range(3):self.step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.step()
    def step(self):
        with torch.autocast('cuda',dtype=torch.bfloat16):output=self.model(input_ids=self.token,past_key_values=self.cache,use_cache=True,position_ids=self.position,logits_to_keep=1)
        self.token.copy_(output.logits[:,-1].argmax(-1,keepdim=True));self.position.add_(1)
    def prepare(self,ids):
        assert ids.ndim==2 and ids.shape[0]==1
        assert ids.shape[1]<self.max_cache_len
        self.cache.reset()
        with torch.autocast('cuda',dtype=torch.bfloat16):output=self.model(input_ids=ids,past_key_values=self.cache,use_cache=True,position_ids=torch.arange(ids.shape[1],device='cuda')[None,None].expand(4,1,-1),logits_to_keep=1)
        self.token.copy_(output.logits[:,-1].argmax(-1,keepdim=True));self.position.fill_(ids.shape[1])
    def generate_ids(self,ids,cap,eos):
        assert ids.shape[1]+cap<=self.max_cache_len
        self.prepare(ids);result=[]
        for i in range(cap):
            if i:self.graph.replay()
            value=int(self.token.item());result.append(value)
            if value in eos:break
        return result
