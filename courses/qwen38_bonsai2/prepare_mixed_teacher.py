"""Cache original-BF16 hidden states for a frozen mixed-context pilot; no training."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from control import sha
from offload import load_model
from verify_source import check_file

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','data','output'):p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--data-sha256',required=True)
    a=p.parse_args()
    if sha(a.data)!=a.data_sha256:raise ValueError('Prepared data identity changed')
    data=json.loads(a.data.read_text(encoding='utf-8'))
    if data['format']!='qwen38-mixed-context-pilot-v1' or data['test_split_read']:raise ValueError('Wrong pilot format')
    if len(data['rows']['train'])!=64 or len(data['rows']['validation'])!=16:raise ValueError('Unexpected split sizes')
    rows=data['rows']['train']+data['rows']['validation']
    seen=set()
    for row in rows:
        ids=row['input_ids'];h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if not 16<=len(ids)<=128 or h!=row['input_IDs_SHA256'] or h in seen:raise ValueError('Invalid/duplicate token row')
        if any(type(v) is not int or not 0<=v<248320 for v in ids):raise ValueError('Invalid token ID')
        seen.add(h)
    if torch.cuda.device_count()!=1:raise ValueError('Expose one admitted GPU')
    mem=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    if int(mem['anon'])+int(mem.get('shmem',0))+80*2**30>=limit:raise ValueError('CPU reserve cannot admit teacher')
    a.output.mkdir(parents=True,exist_ok=False)
    report={'format':'qwen38-mixed-context-teacher-cache-v1','passed':False,'actual_optimizer_updates':0,
      'quality_accepted':False,'data_SHA256':a.data_sha256,'teacher_rows_completed':0,'teacher_row_count':len(rows),
      'TRAIN_rows':64,'VAL_rows':16,'model_weights_modified':False}
    started=time.monotonic()
    def save(stage):
        report.update(state=stage,seconds=time.monotonic()-started)
        temp=a.output/'progress.tmp';temp.write_text(json.dumps(report,indent=2),encoding='utf-8');temp.replace(a.output/'progress.json')
    save('verifying_original_files')
    spec=json.loads(Path(__file__).with_name('source-files.json').read_text(encoding='utf-8'))
    if spec['revision']!=data['source_revision']:raise ValueError('Original revision differs')
    for i,row in enumerate(spec['files']):
        check_file(a.model/row['name'],row);report['verified_original_files']=i+1;save('verifying_original_files')
    for n,h in data['tokenizer_files_SHA256'].items():
        if sha(a.model/n)!=h:raise ValueError('Tokenizer changed')
    torch.set_num_threads(8);save('loading_original_BF16_teacher')
    teacher,_=load_model(a.model,torch.device('cuda',0),quantized=False)
    targets=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            ids=torch.tensor(row['input_ids'],dtype=torch.long,device='cuda')[None]
            value=teacher.model(ids,use_cache=False).last_hidden_state.cpu()
            if list(value.shape)!=[1,len(row['input_ids']),5120] or value.dtype!=torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise ValueError('Teacher target shape/dtype/finite check failed')
            targets.append(value);del ids,value
            report['teacher_rows_completed']=i+1;save('caching_original_BF16_hidden_states')
    del teacher;gc.collect();torch.cuda.empty_cache()
    path=a.output/'teacher_states.pt';partial=path.with_suffix('.partial')
    torch.save({'format':'qwen38-mixed-context-teacher-states-v1','source_revision':spec['revision'],
      'data_SHA256':a.data_sha256,'row_input_IDs_SHA256':[r['input_IDs_SHA256'] for r in rows],'targets':targets},partial)
    partial.rename(path)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    if not all(torch.equal(x,y) for x,y in zip(targets,loaded['targets'])) or len(loaded['targets'])!=len(rows):raise ValueError('Teacher cache reread differs')
    report.update(passed=True,teacher_cache_SHA256=sha(path),teacher_cache_bytes=path.stat().st_size,cache_reload_all_targets_bitwise_equal=True,
      source_revision=spec['revision'],original_BF16_CPU_weight_offload=True)
    save('mixed_teacher_cache_saved_and_reread')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
