"""Prepare bounded authored chat contexts plus the existing pinned WikiText rows."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from control import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','wikitext','contexts','diagnostic-prompts','output'):
        p.add_argument('--'+n,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve the previous prepared data')
    wiki=json.loads(a.wikitext.read_text(encoding='utf-8'))
    corpus=json.loads(a.contexts.read_text(encoding='utf-8'))
    diagnostic=json.loads(a.diagnostic_prompts.read_text(encoding='utf-8'))
    if wiki['format']!='qwen38-wikitext-pilot-v1' or wiki['test_split_read'] or wiki['seq_len']!=128:
        raise ValueError('Expected the exact previously verified WikiText pilot')
    if corpus['format']!='qwen38-authored-bilingual-development-contexts-v1':raise ValueError('Wrong contexts format')
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    tokenizer_hashes={n:sha(a.model/n) for n in ('tokenizer.json','tokenizer_config.json')}
    if wiki['tokenizer_files_SHA256']!=tokenizer_hashes:raise ValueError('Tokenizer changed')
    def normalize(s):return ''.join(s.casefold().split())
    forbidden={normalize(r['prompt']) for r in diagnostic}
    prompt_seen=set();tokens_seen=set();rows={}
    for split,count in (('train',32),('validation',8)):
        if len(wiki['rows'][split])!=count or len(corpus[split])!=count:raise ValueError('Unexpected split sizes')
        selected=[]
        for index,pair in enumerate(corpus[split]):
            prompt,answer=pair
            key=normalize(prompt)
            if key in forbidden or key in prompt_seen:raise ValueError('Prompt overlap across splits/diagnostic')
            prompt_seen.add(key)
            text=tokenizer.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':answer}],tokenize=False,enable_thinking=False)
            ids=tokenizer.encode(text,add_special_tokens=False)
            if not 16<=len(ids)<=128:raise ValueError(f'Context length outside16..128: {split}/{index}/{len(ids)}')
            h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
            if h in tokens_seen:raise ValueError('Duplicate token row')
            tokens_seen.add(h)
            selected.append({'source':'project_authored_chat','source_row':index,'input_ids':ids,'input_IDs_SHA256':h})
            w=wiki['rows'][split][index]
            h=hashlib.sha256(json.dumps(w['input_ids'],separators=(',',':')).encode()).hexdigest()
            if h!=w['input_IDs_SHA256'] or h in tokens_seen:raise ValueError('WikiText token identity/overlap')
            tokens_seen.add(h)
            selected.append(dict(w,source='pinned_wikitext'))
        rows[split]=selected
    result={'format':'qwen38-mixed-context-pilot-v1','source_revision':wiki['tokenizer_original_revision'],
      'tokenizer_files_SHA256':tokenizer_hashes,'wikitext_batches_SHA256':sha(a.wikitext),
      'authored_contexts_SHA256':sha(a.contexts),'diagnostic_prompts_SHA256':sha(a.diagnostic_prompts),
      'source_provenance':corpus['provenance'],'test_split_read':False,'max_seq_len':128,
      'loss_scope':'Teacher KL and normalized hidden-state MSE over all actual input positions, including chat template and assistant context; no padding or token labels.',
      'diagnostic_prompts_used_as_training_rows':False,'development_informed_design':True,
      'formal_quality_benchmark':False,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'TRAIN_rows':len(rows['train']),'VAL_rows':len(rows['validation']),
      'TRAIN_tokens':sum(len(r['input_ids']) for r in rows['train']),'VAL_tokens':sum(len(r['input_ids']) for r in rows['validation']),
      'minimum_tokens':min(len(r['input_ids']) for v in rows.values() for r in v),'maximum_tokens':max(len(r['input_ids']) for v in rows.values() for r in v),
      'output_SHA256':sha(a.output),'no_exact_prompt_overlap':True,'test_split_read':False}))
if __name__=='__main__':main()
