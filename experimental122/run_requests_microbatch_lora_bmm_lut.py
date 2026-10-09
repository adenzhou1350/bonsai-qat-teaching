"""Experimental fixed B4 low-rank BMM CLI with exact ternary LUT decoding."""
import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))

def main():
 import run_requests_microbatch_lora_bmm as entry
 assert hashlib.sha256(Path(entry.__file__).read_bytes()).hexdigest()=='76f1d9b64a6f224bda5df78bbd3d4b96a064658c4669298699396d5f32484727'
 if any(v in ('--help','-h') for v in sys.argv[1:]):
  entry.main();return
 import direct122_original_recovery_runtime_v38_expert_stream as runtime
 from native122_ternary_lut_decode_v1533 import install
 original=runtime.assemble_controlled;installed=[]
 def assemble(*args,**kwargs):
  result=original(*args,**kwargs)
  assert not installed
  installed.append(install())
  return result
 runtime.assemble_controlled=assemble
 try:entry.main()
 finally:
  runtime.assemble_controlled=original
  for module,old,lookup in installed:module.TritBank.weights=old

if __name__=='__main__':main()
