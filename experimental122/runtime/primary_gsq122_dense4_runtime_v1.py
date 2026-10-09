"""Inference dependency extracted from validated experimental runtime."""

from pathlib import Path

import torch

from gemq_dense_runtime import sha

from affine4_packed_runtime import PackedAffine4Linear, _decode

def bank_rows(directory,report):
    rows={};dense=[]
    for row in report['matrices']:
        assert sha(Path(directory)/row['file'])==row['sha256']
        if row['packing_format']=='own-gsq-five-trits-expert-bank-v1':
            assert row['attr'] in ('gate_up_proj','down_proj') and row['module'].endswith('.mlp.experts')
            li=int(row['module'].split('.')[1]);assert row['attr'] not in rows.setdefault(li,{})
            rows[li][row['attr']]=row
        else:
            assert row['packing_format']=='own-affine-dense4-v1' and row['attr']=='weight';dense.append(row)
    assert set(rows)==set(range(48)) and all(set(v)=={'gate_up_proj','down_proj'} for v in rows.values())
    assert sum(v['module']=='lm_head' for v in dense)==1
    return rows,dense

class PackedAffine4NativeChunkedHead(PackedAffine4Linear):
    def forward(self,x):
        import triton
        shape=x.shape[:-1];x=x.to(torch.bfloat16).reshape(-1,self.in_features).contiguous()
        output=torch.empty((len(x),self.out_features),device=x.device,dtype=torch.bfloat16)
        assert not self.has_bias
        for start in range(0,self.out_features,2048):
            stop=min(start+2048,self.out_features);n=stop-start;k=self.in_features
            weight=torch.empty((n,k),device=x.device,dtype=torch.bfloat16)
            _decode[(triton.cdiv(n,4),triton.cdiv(k,256))](self.codes[start:stop],self.scales[start:stop],self.zeros[start:stop],weight,n,k,4,256,enable_fp_fusion=False,num_warps=4)
            output[:,start:stop]=torch.nn.functional.linear(x,weight);del weight
        return output.reshape(*shape,self.out_features)
