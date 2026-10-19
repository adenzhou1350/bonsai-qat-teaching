"""Continue a verified32-step checkpoint on a bounded mixed-context teacher cache."""
import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from transformers.optimization import Adafactor
from buffered_staging import BufferedStagedTernary
from control import packed_reload, sha, tensor_sha
from mapped_gradients import MappedGradients, memory_admission
from offload import load_model, source_name
from recovery_checkpoint import load_checkpoint, save_checkpoint
from recovery_pilot_mapped import export_model, gradients
from rotation import rotate
from verify_source import check_file

def distillation_objective(prediction, target, student_head, teacher_head):
    """Normalize KL by actual sequence length; no padding, same forward loss as pilot."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError('Expected one matched hidden-state sequence')
    def head(x,y):
        logits=student_head(x).float()
        with torch.no_grad():probabilities=teacher_head(y).float().softmax(-1)
        return torch.nn.functional.kl_div(logits.log_softmax(-1),probabilities,reduction='sum')/x.shape[1]
    kl=checkpoint(head,prediction,target,use_reentrant=False) if torch.is_grad_enabled() else head(prediction,target)
    mse=(prediction.float()-target.float()).square().mean()/target.float().square().mean().clamp_min(1e-12)
    return kl+.1*mse,kl,mse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('model','data','teacher-cache','resume-checkpoint','resume-data','resume-packed','resume-proof','output'):
        p.add_argument('--'+n,required=True,type=Path)
    for n in ('data','teacher-cache','resume-manifest','resume-proof'):
        p.add_argument('--'+n+'-sha256',required=True)
    p.add_argument('--steps',type=int,default=64,choices=range(1,65))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    report={'format':'qwen38-mixed-context-checkpoint-continuation-v1','actual_optimizer_updates':0,
      'cumulative_optimizer_updates':32,'source_checkpoint_updates':32,'quality_accepted':False,
      'full_recovery_training_complete':False,'official_training_recipe_reproduced':False,
      'events':[],'validation':[],'checkpoint_written':False}
    started=time.monotonic()
    def save(stage,**fields):
        report.update(state=stage,seconds=time.monotonic()-started,**fields)
        tmp=a.output/'progress.tmp';tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');tmp.replace(a.output/'progress.json')
    for path,digest in ((a.data,a.data_sha256),(a.teacher_cache,a.teacher_cache_sha256),
                        (a.resume_checkpoint/'manifest.json',a.resume_manifest_sha256),(a.resume_proof,a.resume_proof_sha256)):
        if sha(path)!=digest:raise ValueError('Frozen data/cache/checkpoint/proof identity changed')
    data=json.loads(a.data.read_text(encoding='utf-8'));proof=json.loads(a.resume_proof.read_text())
    old=json.loads((a.resume_checkpoint/'manifest.json').read_text())
    packed_manifest=json.loads((a.resume_packed/'manifest.json').read_text())
    old_data=json.loads(a.resume_data.read_text())
    expected={'step':32,'source_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
      'data_SHA256':sha(a.resume_data),'teacher_cache_SHA256':'563ad9a03ba4ff5f05eb4e3ac97ff5a05b6b168adb3400df7983688104cc9d15',
      'optimizer':'Adafactor_beta1None_relativeFalse_scaleFalse','base_learning_rate':1e-5,'warmup_steps':4,
      'gradient_storage':'local_file_mmap_FP32','rotation_seed':381027,'rotation_block':1024,'group':128}
    if old['metadata']!=expected or not old['complete'] or len(old['masters'])!=402:raise ValueError('Wrong source checkpoint')
    if not proof['passed'] or proof['checkpoint_manifest_SHA256']!=a.resume_manifest_sha256 or proof['packed_manifest_SHA256']!=sha(a.resume_packed/'manifest.json'):
        raise ValueError('Restore proof does not identify source artifacts')
    if data['format']!='qwen38-mixed-context-pilot-v1' or data['test_split_read'] or data['wikitext_batches_SHA256']!=sha(a.resume_data):
        raise ValueError('Expected new mixed development context data')
    if len(data['rows']['train'])!=64 or len(data['rows']['validation'])!=16:raise ValueError('Split size changed')
    rows=data['rows']['train']+data['rows']['validation'];seen=set()
    for row in rows:
        ids=row['input_ids'];h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        if not 16<=len(ids)<=128 or h!=row['input_IDs_SHA256'] or h in seen:raise ValueError('Invalid/duplicate row')
        if any(type(i) is not int or not 0<=i<248320 for i in ids):raise ValueError('Invalid token')
        seen.add(h)
    admission=memory_admission(26869760000,248320*5120)
    disk_required=120_000_000_000+6_000_000_000+107479040000+20*2**30
    if shutil.disk_usage(a.output).free<disk_required:raise ValueError('Disk reserve does not admit new120GB checkpoint budget and gradient scratch')
    if torch.cuda.device_count()!=1:raise ValueError('Expose one admitted GPU')
    spec=json.loads(Path(__file__).with_name('source-files.json').read_text())
    if spec['revision']!=expected['source_revision'] or data['source_revision']!=spec['revision']:raise ValueError('Source revision changed')
    save('verifying_all_original_files',memory_admission=admission,data_SHA256=a.data_sha256,
      teacher_cache_SHA256=a.teacher_cache_sha256,source_checkpoint_manifest_SHA256=a.resume_manifest_sha256,
      checkpoint_maximum_bytes=120_000_000_000,TRAIN_rows=64,VAL_rows=16,
      scope='One additional mixed-development-data epoch; not an official recipe or capability acceptance.')
    for i,row in enumerate(spec['files']):
        check_file(a.model/row['name'],row);save('verifying_all_original_files',verified_original_files=i+1)
    for n,h in data['tokenizer_files_SHA256'].items():
        if sha(a.model/n)!=h:raise ValueError('Tokenizer changed')
    cache=torch.load(a.teacher_cache,map_location='cpu',weights_only=True)
    if cache['source_revision']!=spec['revision'] or cache['data_SHA256']!=a.data_sha256 or cache['row_input_IDs_SHA256']!=[r['input_IDs_SHA256'] for r in rows]:
        raise ValueError('Teacher cache source/data/order mismatch')
    targets=cache['targets'];del cache
    if len(targets)!=len(rows):raise ValueError('Teacher row count changed')
    for row,target in zip(rows,targets):
        if list(target.shape)!=[1,len(row['input_ids']),5120] or target.dtype!=torch.bfloat16 or not bool(torch.isfinite(target).all()):
            raise ValueError('Teacher target invalid')
    torch.set_num_threads(8);torch.manual_seed(381027);device=torch.device('cuda',0)
    def loaded(name,count):
        if count%16==0:save('reconstructing_original_rotated_basis',matrices_loaded=count)
    student,cfg=load_model(a.model,device,quantized=True,on_loaded=loaded,staged_type=BufferedStagedTernary)
    named=[(n,v) for n,v in student.named_parameters() if v.requires_grad]
    if len(named)!=402 or sum(v.numel() for _,v in named)!=26869760000:raise ValueError('Master scope differs')
    optimizer=Adafactor([v for _,v in named],lr=1e-5,relative_step=False,scale_parameter=False,warmup_init=False,beta1=None,clip_threshold=1.,weight_decay=0.)
    save('restoring_source_checkpoint_masters_optimizer_and_RNG')
    load_checkpoint(student,optimizer,a.resume_checkpoint,expected)
    if any(optimizer.state[v]['step']!=32 for _,v in named):raise ValueError('Source optimizer step counters differ')
    kept={n:tensor_sha(v) for n,v in student.named_parameters() if not v.requires_grad}
    if kept!=packed_manifest['retained_state_hashes']:raise ValueError('Retained state changed')
    student.train()
    original_val=torch.tensor(old_data['rows']['validation'][0]['input_ids'],device=device)[None]
    with torch.no_grad():resumed=student.lm_head(student.model(original_val,use_cache=False).last_hidden_state[:,-1:]).cpu()
    if tensor_sha(resumed)!=proof['restored_last_position_logits_SHA256']:raise ValueError('Resumed originalVAL forward differs from verified source checkpoint')
    del resumed,original_val
    save('source_checkpoint_restore_forward_and_step32_verified',source_checkpoint_forward_SHA256_equal_independent_proof=True,
      restored_optimizer_all402_step_counters=32)
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(str(a.model/index['lm_head.weight']),framework='pt',device='cpu') as stream:
        teacher_weight=stream.get_tensor('lm_head.weight').to(device)
    def teacher_head(x):return torch.nn.functional.linear(x,teacher_weight)
    gradient_store=MappedGradients(named,a.output/'gradient-scratch')
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def objective(i):
        ids=torch.tensor(rows[i]['input_ids'],dtype=torch.long,device=device)[None]
        prediction=student.model(ids,use_cache=False).last_hidden_state
        return distillation_objective(prediction,targets[i].to(device),student.lm_head,teacher_head)
    def validate(step):
        measured=[]
        with torch.no_grad():
            for i in range(64,80):
                v=[float(x) for x in objective(i)]
                if not all(math.isfinite(x) for x in v):raise ValueError('Non-finite validation')
                measured.append(v)
        groups={}
        for source in ('project_authored_chat','pinned_wikitext'):
            group=[r for i,r in enumerate(measured) if rows[64+i]['source']==source]
            groups[source]=[sum(r[k] for r in group)/len(group) for k in range(3)]
        means=[sum(r[k] for r in measured)/len(measured) for k in range(3)]
        report['validation'].append({'after_new_updates':step,'cumulative_updates':32+step,
          'mean_loss_KL_MSE':means,'by_source_mean_loss_KL_MSE':groups,'rows_loss_KL_MSE':measured})
        save('validation_after_new_updates_'+str(step));return groups
    before=validate(0)
    for step in range(a.steps):
        gradient_store.zero_grad();tick=time.monotonic();save('training_new_update_'+str(step+1))
        for group in optimizer.param_groups:group['lr']=1e-5
        loss,kl,mse=objective(step)
        if not bool(torch.isfinite(loss)):raise ValueError('Non-finite training loss')
        loss.backward();gradient_store.assert_complete();norm=gradients(named);optimizer.step()
        cumulative=33+step
        if any(optimizer.state[v]['step']!=cumulative for _,v in named):raise ValueError('Optimizer counters did not advance from restored state')
        report.update(actual_optimizer_updates=step+1,cumulative_optimizer_updates=cumulative)
        report['events'].append({'new_step':step+1,'cumulative_step':cumulative,'TRAIN_row':step,'source':rows[step]['source'],
          'tokens':len(rows[step]['input_ids']),'lr':1e-5,'loss':float(loss.detach()),'KL':float(kl.detach()),'state_MSE':float(mse.detach()),
          'finite_CPU_gradient_matrices':402,'global_gradient_norm_not_used_for_clipping':norm,'seconds':time.monotonic()-tick})
        del loss,kl,mse;gradient_store.zero_grad();save('completed_new_update_'+str(step+1))
        if (step+1)%8==0 or step+1==a.steps:
            after=validate(step+1)
            if any(after[k][1]>1.5*before[k][1] for k in before):
                report['early_stop_reason']='Either fixed VAL source mean KL exceeds1.5x its own initial value';break
    gradient_store.close();del gradient_store
    if kept!={n:tensor_sha(v) for n,v in student.named_parameters() if n in kept}:raise ValueError('Retained state changed')
    report['retained_tensor_hashes_unchanged']=True
    val_ids=torch.tensor(rows[64]['input_ids'],device=device)[None]
    with torch.no_grad():reference=student.lm_head(student.model(val_ids,use_cache=False).last_hidden_state[:,-1:]).cpu()
    updates=report['cumulative_optimizer_updates'];save('exporting_new_packed_artifact')
    manifest=export_model(student,cfg,kept,a.output/'packed',updates,spec['revision'],a.data_sha256)
    manifest.update(source_checkpoint_manifest_SHA256=a.resume_manifest_sha256,new_optimizer_updates=report['actual_optimizer_updates'],
      additional_training_format='qwen38-mixed-context-pilot-v1',small_recovery_pilot_only=True)
    (a.output/'packed/manifest.json').write_text(json.dumps(manifest,indent=2))
    report.update(packed_manifest_SHA256=sha(a.output/'packed/manifest.json'),packed_file_bytes=sum(v.stat().st_size for v in (a.output/'packed').iterdir()))
    def base_factory(name,param):
        module_name,_,leaf=name.rpartition('.')
        if leaf!='master':raise ValueError('Expected master parameter')
        module=student.get_submodule(module_name);key=source_name(module_name+'.weight')
        with safe_open(str(a.model/index[key]),framework='pt',device='cpu') as stream:value=stream.get_tensor(key)
        return rotate(value.float(),module.signs,module.block)
    identity={'step':updates,'source_revision':spec['revision'],'data_SHA256':a.data_sha256,'teacher_cache_SHA256':a.teacher_cache_sha256,
      'optimizer':'Adafactor_beta1None_relativeFalse_scaleFalse','base_learning_rate':1e-5,'warmup_steps':0,
      'schedule':'continue_existing_lr_no_new_warmup','gradient_storage':'local_file_mmap_FP32','rotation_seed':381027,'rotation_block':1024,'group':128,
      'source_checkpoint_manifest_SHA256':a.resume_manifest_sha256,'source_checkpoint_step':32,'new_optimizer_updates':report['actual_optimizer_updates']}
    del teacher_weight;gc.collect();torch.cuda.empty_cache();save('saving_new_lossless_checkpoint')
    saved=save_checkpoint(student,optimizer,a.output/'checkpoint-final',identity,maximum_bytes=120_000_000_000,base_factory=base_factory)
    report.update(checkpoint_written=True,checkpoint_file_bytes=saved['written_tensor_and_state_bytes'],checkpoint_manifest_SHA256=sha(a.output/'checkpoint-final/manifest.json'))
    save('reconstructing_original_basis_for_new_checkpoint_readback')
    with torch.no_grad():
        for name,param in named:
            base=base_factory(name,param);param.copy_(base);del base
    save('restoring_new_checkpoint')
    load_checkpoint(student,optimizer,a.output/'checkpoint-final',identity)
    with torch.no_grad():actual=student.lm_head(student.model(val_ids,use_cache=False).last_hidden_state[:,-1:]).cpu()
    if not torch.equal(reference,actual):raise ValueError('New restored forward differs')
    report['new_checkpoint_restored_forward_bitwise_equal']=True
    del named,optimizer,student,param;gc.collect();torch.cuda.empty_cache();save('checking_new_packed_forward')
    packed,_=packed_reload(a.output/'packed',device)
    with torch.no_grad():actual=packed.lm_head(packed.model(val_ids,use_cache=False).last_hidden_state[:,-1:]).cpu()
    if not torch.equal(reference,actual):raise ValueError('New packed forward differs')
    report.update(passed=True,packed_last_position_logits_bitwise_equal=True,
      source_checkpoint_next_update_executed_with_all402_finite_gradients=True,
      bitwise_next_update_comparison_to_original_pre_save_live_process=False,
      limitation='Bounded mixed development contexts; semantic generation and independent unseen evaluation still required.')
    save('mixed_continuation_export_checkpoint_and_readback_completed')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
