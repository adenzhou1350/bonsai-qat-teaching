"""Precompute original BF16 normalized states; no teacher weights during QAT."""
import argparse,hashlib,json
from pathlib import Path
import torch
from model import TextModel

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--tokens',required=True)
    p.add_argument('--output',required=True); p.add_argument('--devices',default='0,1'); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    tokens=torch.load(a.tokens,map_location='cpu',weights_only=True)
    assert set(tokens)>={'train','val'} and tokens['train'].shape[1]==tokens['val'].shape[1]
    model=TextModel(a.model,[int(x) for x in a.devices.split(',')],False)
    metadata={'model_index_sha256':sha(Path(a.model)/'model.safetensors.index.json'),
              'tokens_sha256':sha(a.tokens),'seq_len':tokens['train'].shape[1]-1,'hidden_size':model.embedding.shape[1],'splits':{}}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for split in ('train','val'):
            path=out/f'{split}.bin'; shape=(len(tokens[split]),metadata['seq_len'],metadata['hidden_size'])
            with path.open('wb') as writer:
                for i,ids in enumerate(tokens[split]):
                    states=model(ids[:-1][None]).detach().cpu().contiguous()
                    assert states.dtype==torch.bfloat16 and torch.isfinite(states).all()
                    writer.write(states.view(torch.uint16).numpy().tobytes())
                    print(f'{split} {i+1}/{len(tokens[split])}',flush=True)
            metadata['splits'][split]={'shape':shape,'sha256':sha(path)}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
