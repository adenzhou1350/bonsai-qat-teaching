"""Experimental serial CLI with exact ternary LUT weight decoding."""
import hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))

def main():
 import run_requests as entry
 assert hashlib.sha256(Path(entry.__file__).read_bytes()).hexdigest()=='e106bad7b274cd5980d9471ec39c3f5523271a376593caf5507ecac985e045ab'
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
