"""Experimental entry preserving the original request/EOS/cache behavior."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from run_requests import main
if __name__=='__main__':
 if '--help' not in sys.argv:
  from tune122_decode_geometry_v1 import install
  install()
 main()
