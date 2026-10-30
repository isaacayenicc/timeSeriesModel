"""Create and hash an archive excluding user raw data."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rotmach.release import package, verify
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    verify(root)
    result=package(root,root.parent/'MECG7780_Updated_Fault_Analysis_Project_Release.zip')
    (root.parent/'MECG7780_updated_release_archive.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
