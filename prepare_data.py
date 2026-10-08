"""Tokenize local JSONL {text: ...}; validation uses separate documents."""
import argparse,json,random
from pathlib import Path
import torch
from transformers import AutoTokenizer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--jsonl',required=True)
    p.add_argument('--output',required=True); p.add_argument('--seq-len',type=int,default=4096)
    p.add_argument('--train-blocks',type=int,default=4096); p.add_argument('--val-blocks',type=int,default=128)
    a=p.parse_args(); tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    documents=[json.loads(line)['text'] for line in Path(a.jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    documents=list(dict.fromkeys(documents)); random.Random(9117).shuffle(documents)
    cut=max(1,len(documents)//32); splits={'val':documents[:cut],'train':documents[cut:]}; output={}
    for name,rows in splits.items():
        count=a.val_blocks if name=='val' else a.train_blocks; buffer=[]; blocks=[]
        for text in rows:
            buffer.extend(tok.encode(text,add_special_tokens=False)+[tok.eos_token_id])
            while len(buffer)>=a.seq_len and len(blocks)<count:
                blocks.append(buffer[:a.seq_len]); del buffer[:a.seq_len]
            if len(blocks)==count: break
        assert len(blocks)==count, f'Not enough {name} text: {len(blocks)}/{count} blocks'
        output[name]=torch.tensor(blocks,dtype=torch.long)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save(output,a.output)
if __name__=='__main__': main()
