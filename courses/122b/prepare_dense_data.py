"""Rebuild approved dense calibration data from pinned public references.

This freezes the historical approved row selection. It never opens evaluation
or held-out files, reruns their filtering, or creates recovery-training data.
"""
import argparse,hashlib,json,random
from pathlib import Path

RECIPE_SHA='29f698f738ae4c2d7208c029c2db43768033a9117d8d7b686bf85388de745aa9'
SOURCE_SHA='f5e84c780740ad1a69a73047e4d41b06b7ad7a73dd767af8e57e8191782be7fa'
TOKEN_FILE_SHA='ce298b506b086b7fecfaaf2b3a25a08895f651ec0a73cc640a0d6a9b44f8c1d9'

def sha(data):return hashlib.sha256(data).hexdigest()

def rebuild_corpus(recipe,source):
 assert recipe['format']=='frozen-approved-reference-chat-corpus-v1'
 assert recipe['source_sha256']==SOURCE_SHA
 result={};payloads={};seen=set()
 for split in ('validation','train'):
  spec=recipe['splits'][split];rows=[]
  for row in spec['code_rows']:
   original=source[row['source_index']]
   prompt=original['instruction'].strip()+('\n\n'+original['input'].strip() if original.get('input','').strip() else '')
   text=original['output']
   rows.append({'id':row['id'],'kind':'code','instruction':prompt,'response':text,
    'reasoning_mode':False,'response_origin':'public_codealpaca_reference',
    'source_index':row['source_index'],'reference_response_sha256':sha(text.encode())})
  for row in spec['authored_math_rows']:
   assert row['kind']=='math' and row['response_origin']=='authored_exact_arithmetic_reference'
   assert row['response']==str(row['expected_integer']) and row['reasoning_mode'] is False
   rows.append(row)
  assert len(rows)==spec['count']
  for row in rows:
   key=' '.join(row['instruction'].split()).casefold();assert key not in seen;seen.add(key)
  # Original corpus JSON was written on Windows. Match its byte identity
  # explicitly instead of depending on this machine's newline convention.
  text=json.dumps(rows,ensure_ascii=False,indent=2)
  candidates=[text.encode(),text.replace('\n','\r\n').encode()]
  matches=[data for data in candidates if sha(data)==spec['expected_corpus_sha256']]
  assert len(matches)==1,('Approved corpus bytes differ',split)
  result[split]=rows;payloads[split]=matches[0]
 return result,payloads

def make_blocks(tokenizer,datasets,recipe):
 tokens={};metadata={}
 for split,count,seed,selected,domains in (('train',512,113,2663,{'code':2113,'math':550}),('validation',16,114,100,{'code':51,'math':49})):
  rows=list(datasets[split]);random.Random(seed).shuffle(rows);pool=[];selection=[]
  for row in rows:
   ids=tokenizer.apply_chat_template([{'role':'user','content':row['instruction']},
    {'role':'assistant','content':row['response']}],tokenize=True,return_dict=False,
    add_generation_prompt=False,enable_thinking=False)
   assert isinstance(ids,list) and len(ids)>=2 and all(type(v) is int and 0<=v<248320 for v in ids)
   selection.append({'row_id':row['id'],'kind':row['kind'],'start_token':len(pool),'tokens':len(ids)})
   pool.extend(ids)
   if len(pool)>=count*513:break
  assert len(pool)>=count*513,'Insufficient distinct approved conversations; never repeat data to fill the budget.'
  spans=[pool[i*513:(i+1)*513] for i in range(count)]
  digest=sha(json.dumps(spans).encode());assert digest==recipe['selected_token_sha256'][split],('Tokenizer or serialization differs',split)
  actual_domains={k:sum(v['kind']==k for v in selection) for k in ('code','math')}
  assert len(selection)==selected and actual_domains==domains
  tokens[split]=spans;metadata[split]={'blocks':count,'serialized_tokens':count*513,
   'model_input_positions':count*512,'selected_rows':len(selection),'selected_domains':actual_domains,
   'serialized_tokens_before_final_cut':len(pool),'selected_token_sha256':digest,'selection':selection}
 assert len({tuple(v) for split in tokens.values() for v in split})==528
 return tokens,metadata

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--codealpaca',required=True,type=Path,help='Pinned data/code_alpaca_20k.json; source SHA is checked.')
 p.add_argument('--recipe',required=True,type=Path,help='approved-recipe.json shipped with this lesson.')
 p.add_argument('--tokenizer',required=True,type=Path,help='Local original Qwen3.5-122B-A10B tokenizer directory.')
 p.add_argument('--output',required=True,type=Path)
 a=p.parse_args();assert not a.output.exists(),'Choose a new output directory; old data is preserved.'
 recipe_bytes=a.recipe.read_bytes();source_bytes=a.codealpaca.read_bytes()
 assert sha(recipe_bytes)==RECIPE_SHA and sha(source_bytes)==SOURCE_SHA
 recipe=json.loads(recipe_bytes);datasets,payloads=rebuild_corpus(recipe,json.loads(source_bytes))
 from transformers import AutoTokenizer,__version__
 assert __version__=='5.12.1','Use the verified tokenizer implementation.'
 tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
 tokens,metadata=make_blocks(tokenizer,datasets,recipe)
 token_bytes=json.dumps(tokens,indent=2).encode();assert sha(token_bytes)==TOKEN_FILE_SHA
 a.output.mkdir(parents=True,exist_ok=False)
 for split,data in payloads.items():(a.output/(split+'.json')).write_bytes(data)
 (a.output/'tokens.json').write_bytes(token_bytes)
 report={'passed':True,'format':'portable122-approved-dense-data-reconstruction-v1',
  'source_sha256':SOURCE_SHA,'source_commit':recipe['source_commit'],'recipe_sha256':RECIPE_SHA,
  'corpus_sha256':{k:sha(v) for k,v in payloads.items()},'tokens_file_sha256':sha(token_bytes),
  'splits':metadata,'tokenizer_files_sha256':{n:sha((a.tokenizer/n).read_bytes())
   for n in ('tokenizer.json','tokenizer_config.json')},'reserved_inputs_opened':False,
  'recovery_training_data_generated':False,'complete_portable_training_recipe':False,
  'limits':'Reconstructs fixed historical approved calibration TRAIN and calibration VAL, including reference responses. No benchmark inputs are read. Only TRAIN updates Hessians; VAL is a reconstruction control. Blocks concatenate complete conversations and may cut the last one; no padding or repetition. CodeAlpaca license is CC BY-NC4.0. Frozen selection does not rerun historical exclusion filtering or establish semantic contamination/answer correctness guarantees. Recovery OpenThoughts4096/128 remains separate.'}
 (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k!='splits'}))

if __name__=='__main__':main()
