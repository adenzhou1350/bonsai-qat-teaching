"""Bounded real-data Qwen3.8 QAT pilot; no official recipe or capability-recovery claim."""
import argparse,gc,hashlib,json,math,shutil,time
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from transformers.optimization import Adafactor
from safetensors import safe_open
from control import memory_admission,packed_reload,sha,tensor_sha
from offload import load_model,source_name
from buffered_staging import BufferedStagedTernary
from recovery_checkpoint import load_checkpoint,save_checkpoint
from rotation import rotate
from verify_source import check_file

def export_model(student,cfg,kept,directory,updates,revision,data_sha):
    directory.mkdir()
    (directory/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
    matrices=[]
    for name,module in student.named_modules():
        if not isinstance(module,BufferedStagedTernary):continue
        value=module.packed(device=torch.device('cuda',0));path=directory/f'{len(matrices):03d}.pt'
        torch.save(value,path);matrices.append({'module':name,'file':path.name,'SHA256':sha(path)});del value
    torch.save({n:p.detach().cpu() for n,p in student.named_parameters() if n in kept},directory/'retained.pt')
    manifest={'format':'teaching-qwen38-rotated-ternary-recovery-pilot-v1','matrices':matrices,
              'config_SHA256':sha(directory/'config.json'),'retained_SHA256':sha(directory/'retained.pt'),
              'actual_optimizer_updates':updates,'source_revision':revision,'data_SHA256':data_sha,
              'quality_accepted':False,'official_Bonsai_GGUF_compatible':False,'fused_kernel':False,
              'retained_state_hashes':kept,'control_only':False,'small_recovery_pilot_only':True}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def gradients(named):
    norms=[]
    for _,param in named:
        value=param.grad
        if value is None or value.device.type!='cpu' or value.dtype!=torch.float32:
            raise ValueError('Missing CPU FP32 master gradient')
        flat=value.detach().view(-1)
        for low in range(0,flat.numel(),2*1024*1024):
            if not bool(torch.isfinite(flat[low:low+2*1024*1024]).all()):raise ValueError('Non-finite gradient')
        norms.append(torch.linalg.vector_norm(value))
    norm=torch.stack(norms).norm()
    if not bool(torch.isfinite(norm)):raise ValueError('Non-finite total gradient norm')
    return float(norm)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--data',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--steps',type=int,default=32,choices=range(1,33))
    p.add_argument('--lr',type=float,default=1e-5);a=p.parse_args()
    if not 0<a.lr<=1e-5:raise ValueError('Bounded pilot learning rate')
    a.output.mkdir(parents=True,exist_ok=False)
    report={'state':'preflight','actual_optimizer_updates':0,'full_recovery_training_complete':False,
            'quality_accepted':False,'official_training_recipe_reproduced':False,'events':[],'validation':[]}
    started=time.monotonic()
    def save(stage,**fields):
        report.update(state=stage,seconds=time.monotonic()-started,**fields)
        target=a.output/'progress.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(target)
    data=json.loads(a.data.read_text());data_sha=sha(a.data)
    if data['format']!='qwen38-wikitext-pilot-v1' or data['test_split_read'] or data['seq_len']!=128:
        raise ValueError('Expected prepared128-token WikiText pilot with unopened test split')
    if len(data['rows']['train'])!=32 or len(data['rows']['validation'])!=8:raise ValueError('Pilot split size changed')
    seen=set()
    for rows in data['rows'].values():
        for row in rows:
            h=hashlib.sha256(json.dumps(row['input_ids'],separators=(',',':')).encode()).hexdigest()
            if len(row['input_ids'])!=128 or h!=row['input_IDs_SHA256'] or h in seen:raise ValueError('Data identity/overlap failure')
            if not all(isinstance(i,int) and 0<=i<248320 for i in row['input_ids']):raise ValueError('Invalid token ID')
            seen.add(h)
    for name,h in data['tokenizer_files_SHA256'].items():
        if sha(a.model/name)!=h:raise ValueError('Tokenizer changed')
    admission=memory_admission(26869760000)
    available=int(dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())['MemAvailable'].split()[0])*1024
    if available<214958080000+64*2**30:raise ValueError('Host RAM reserve does not admit the full CPU state')
    disk_required=80_000_000_000+6_000_000_000+12*2**30
    if shutil.disk_usage(a.output).free<disk_required:raise ValueError('Disk must admit checkpoint cap, new packed export and reserve')
    save('verifying_all_pinned_original_files',memory_admission=admission,data_SHA256=data_sha,
         TRAIN_rows=32,VAL_rows=8,tokens_per_row=128,pinned_buffer_bytes=64*2**20,
         checkpoint_maximum_bytes=80_000_000_000,checkpoint_written=False,
         scope='One small English WikiText distillation pilot; not Chinese/code/agent capability evaluation')
    spec=json.loads(Path(__file__).with_name('source-files.json').read_text())
    if spec['revision']!='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0':raise ValueError('Wrong source revision')
    verified=[]
    for row in spec['files']:
        verified.append(check_file(a.model/row['name'],row));save('verifying_all_pinned_original_files',verified_original_files=len(verified))
    report['source_revision']=spec['revision'];report['original_source_spec_SHA256']=sha(Path(__file__).with_name('source-files.json'))
    (a.output/'original-proof.json').write_text(json.dumps({'passed':True,'revision':spec['revision'],'completed_files':verified},indent=2))
    torch.set_num_threads(8);torch.manual_seed(381027)
    if torch.cuda.device_count()!=1:raise ValueError('Expose only one freshly admitted GPU')
    device=torch.device('cuda',0)
    train=[torch.tensor(r['input_ids'],device=device,dtype=torch.long)[None] for r in data['rows']['train']]
    val=[torch.tensor(r['input_ids'],device=device,dtype=torch.long)[None] for r in data['rows']['validation']]
    rows=train+val
    save('loading_BF16_teacher')
    teacher,_=load_model(a.model,device,quantized=False)
    targets=[]
    with torch.no_grad():
        for index,ids in enumerate(rows):
            targets.append(teacher.model(ids,use_cache=False).last_hidden_state.cpu())
            save('preparing_teacher_states',teacher_rows_completed=index+1)
    teacher_head=teacher.lm_head
    teacher_head.weight=torch.nn.Parameter(teacher_head.weight.detach().to(device),requires_grad=False)
    del teacher;gc.collect();torch.cuda.empty_cache()
    cache=a.output/'teacher_states.pt';torch.save({'source_revision':spec['revision'],'data_SHA256':data_sha,'targets':targets},cache)
    report['teacher_cache_SHA256']=sha(cache)
    def loaded(name,count):
        if count%16==0:save('loading_rotated_CPU_masters',matrices_loaded=count)
    save('loading_rotated_CPU_masters',matrices_loaded=0)
    student,cfg=load_model(a.model,device,quantized=True,on_loaded=loaded,staged_type=BufferedStagedTernary)
    named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
    if len(named)!=402 or sum(p.numel() for _,p in named)!=26869760000:raise ValueError('Full matrix scope changed')
    kept={n:tensor_sha(p) for n,p in student.named_parameters() if not p.requires_grad}
    optimizer=Adafactor([p for _,p in named],lr=a.lr,relative_step=False,scale_parameter=False,
                       warmup_init=False,beta1=None,clip_threshold=1.,weight_decay=0.)
    student.train();student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def objective(index):
        prediction=student.model(rows[index],use_cache=False).last_hidden_state;target=targets[index].to(device)
        def head(x,y):
            logits=student.lm_head(x).float()
            with torch.no_grad():probabilities=teacher_head(y).float().softmax(-1)
            return torch.nn.functional.kl_div(logits.log_softmax(-1),probabilities,reduction='sum')/128
        kl=checkpoint(head,prediction,target,use_reentrant=False) if torch.is_grad_enabled() else head(prediction,target)
        mse=(prediction.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-12)
        return kl+.1*mse,kl,mse
    def validate(step):
        measured=[]
        with torch.no_grad():
            for index in range(32,40):
                value=[float(v) for v in objective(index)]
                if not all(math.isfinite(v) for v in value):raise ValueError('Non-finite validation')
                measured.append(value)
        means=[sum(row[k] for row in measured)/8 for k in range(3)]
        report['validation'].append({'after_updates':step,'mean_loss_KL_MSE':means,'rows_loss_KL_MSE':measured})
        save('validation_completed_'+str(step));return means
    before=validate(0)
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True);lr=a.lr*min(1.,(step+1)/4)
        for group in optimizer.param_groups:group['lr']=lr
        tick=time.monotonic();save('training_update_'+str(step+1))
        loss,kl,mse=objective(step%32)
        if not bool(torch.isfinite(loss)):raise ValueError('Non-finite training loss')
        loss.backward();norm=gradients(named);optimizer.step()
        report['actual_optimizer_updates']=step+1
        report['events'].append({'step':step+1,'TRAIN_row':step%32,'lr':lr,'loss':float(loss.detach()),
            'KL':float(kl.detach()),'state_MSE':float(mse.detach()),'finite_CPU_gradient_matrices':402,
            'global_gradient_norm_not_used_for_clipping':norm,'seconds':time.monotonic()-tick})
        del loss,kl,mse;optimizer.zero_grad(set_to_none=True);save('completed_update_'+str(step+1))
        if (step+1)%8==0 or step+1==a.steps:
            after=validate(step+1)
            if after[1]>1.5*before[1]:
                report['early_stop_reason']='Predeclared VAL KL guard: greater than1.5 times initial mean';break
    updates=report['actual_optimizer_updates'];report['validation_mean_KL_improved']=report['validation'][-1]['mean_loss_KL_MSE'][1]<before[1]
    if kept!={n:tensor_sha(p) for n,p in student.named_parameters() if n in kept}:raise ValueError('Retained parameter changed')
    report['retained_tensor_hashes_unchanged']=True
    with torch.no_grad():reference=student.lm_head(student.model(val[0],use_cache=False).last_hidden_state[:,-1:]).cpu()
    save('exporting_same_GPU_packed_grids')
    manifest=export_model(student,cfg,kept,a.output/'packed',updates,spec['revision'],data_sha)
    report.update(packed_manifest_SHA256=sha(a.output/'packed/manifest.json'),packed_file_bytes=sum(p.stat().st_size for p in (a.output/'packed').iterdir()))
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    def base_factory(name,param):
        module_name,_,leaf=name.rpartition('.')
        if leaf!='master':raise ValueError('Expected a rotated master name')
        module=student.get_submodule(module_name);key=source_name(module_name+'.weight')
        with safe_open(str(a.model/index[key]),framework='pt',device='cpu') as stream:value=stream.get_tensor(key)
        return rotate(value.float(),module.signs,module.block)
    identity={'step':updates,'source_revision':spec['revision'],'data_SHA256':data_sha,
              'teacher_cache_SHA256':report['teacher_cache_SHA256'],'optimizer':'Adafactor_beta1None_relativeFalse_scaleFalse',
              'base_learning_rate':a.lr,'warmup_steps':4,'rotation_seed':381027,'rotation_block':1024,'group':128}
    del teacher_head;gc.collect();torch.cuda.empty_cache()
    save('saving_lossless_XOR_master_optimizer_RNG_checkpoint')
    state=save_checkpoint(student,optimizer,a.output/'checkpoint-final',identity,maximum_bytes=80_000_000_000,base_factory=base_factory)
    report.update(checkpoint_written=True,checkpoint_file_bytes=state['written_tensor_and_state_bytes'],
                  checkpoint_manifest_SHA256=sha(a.output/'checkpoint-final/manifest.json'))
    save('reconstructing_original_basis_before_checkpoint_restore')
    with torch.no_grad():
        for name,param in named:
            base=base_factory(name,param);param.copy_(base);del base
    save('restoring_all402_FP32_masters_optimizer_and_RNG_from_disk')
    load_checkpoint(student,optimizer,a.output/'checkpoint-final',identity)
    with torch.no_grad():restored=student.lm_head(student.model(val[0],use_cache=False).last_hidden_state[:,-1:]).cpu()
    if not torch.equal(reference,restored):raise ValueError('Restored master forward differs')
    report['checkpoint_restored_master_forward_bitwise_equal']=True
    del named,optimizer,student,param;gc.collect();torch.cuda.empty_cache()
    save('reconstructing_packed_model')
    packed,_=packed_reload(a.output/'packed',device)
    with torch.no_grad():actual=packed.lm_head(packed.model(val[0],use_cache=False).last_hidden_state[:,-1:]).cpu()
    if not torch.equal(reference,actual):raise ValueError('Packed model differs from trained pilot')
    report.update(passed=True,selected_matrices=len(manifest['matrices']),packed_last_position_logits_bitwise_equal=True,
                  peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
                  full27_next_optimizer_update_after_restore_verified=False,
                  limitation='At most32 short English TRAIN rows; no complete capability recovery claim; only final checkpoint is written.')
    save('bounded_real_data_pilot_export_checkpoint_restore_completed')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
