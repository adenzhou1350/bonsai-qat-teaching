"""Prepare a small, pinned WikiText TRAIN/VAL pilot; the test split is never read."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
FILES = {
    'train': (6357543, 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7'),
    'validation': (657209, '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c'),
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--train-rows',type=int,default=32)
    p.add_argument('--val-rows',type=int,default=8)
    p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--seed',type=int,default=381027)
    p.add_argument('--endpoint',choices=('https://huggingface.co','https://hf-mirror.com'),default='https://huggingface.co')
    p.add_argument('--local-only',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not 1<=a.train_rows<=256 or not 1<=a.val_rows<=32 or not 32<=a.seq_len<=512:
        raise ValueError('Use a new output and bounded pilot sizes')
    a.data_root.mkdir(parents=True,exist_ok=True)
    paths={}
    for split,(size,expected) in FILES.items():
        path=a.data_root/(split+'.parquet')
        if not path.exists():
            if a.local_only:raise FileNotFoundError('Required pinned dataset file is absent')
            partial=path.with_suffix('.partial')
            if partial.exists():raise FileExistsError('Inspect the preserved partial download')
            url=f'{a.endpoint}/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','10','--max-time','120','--output',str(partial),url],check=True,timeout=130)
            if partial.stat().st_size!=size or digest(partial)!=expected:
                raise ValueError('Dataset object differs from the pinned primary Hub hash')
            partial.rename(path)
        if path.stat().st_size!=size or digest(path)!=expected:
            raise ValueError('Dataset object changed')
        paths[split]=path
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    selected={};seen=set()
    for split,count in (('train',a.train_rows),('validation',a.val_rows)):
        texts=pq.read_table(paths[split],columns=['text'],use_threads=False)['text'].to_pylist()
        indices=list(range(len(texts)));random.Random(f'{a.seed}:{split}').shuffle(indices)
        rows=[]
        for index in indices:
            text=texts[index]
            if len(text)<a.seq_len:continue
            tokens=tokenizer.encode(text,add_special_tokens=False)
            if len(tokens)<a.seq_len:continue
            ids=tokens[:a.seq_len]
            identity=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
            if identity in seen:continue
            seen.add(identity)
            rows.append({'source_row':index,'input_ids':ids,'input_IDs_SHA256':identity})
            if len(rows)==count:break
        if len(rows)!=count:raise ValueError('Insufficient nonduplicate full paragraphs')
        selected[split]=rows
    result={'format':'qwen38-wikitext-pilot-v1','dataset':'Salesforce/wikitext','config':'wikitext-2-raw-v1',
            'revision':REVISION,'source_files':{s:{'bytes':n,'SHA256':h} for s,(n,h) in FILES.items()},
            'source_card':f'https://huggingface.co/datasets/Salesforce/wikitext/blob/{REVISION}/README.md',
            'selection':'seed-shuffled source paragraph rows; first full token window; exact input-ID dedup across TRAIN and VAL',
            'seed':a.seed,'seq_len':a.seq_len,'test_split_read':False,
            'tokenizer_original_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            'tokenizer_files_SHA256':{n:digest(a.model/n) for n in ('tokenizer.json','tokenizer_config.json')},
            'rows':selected,'quality_scope':'small English language-modeling pilot, not Chinese/code/agent capability evaluation'}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'TRAIN_rows':len(selected['train']),'VAL_rows':len(selected['validation']),
                      'tokens_per_row':a.seq_len,'output_SHA256':digest(a.output),'test_split_read':False}))

if __name__=='__main__':main()
