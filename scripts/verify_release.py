"""Verify available deliverables without implying blocked work is complete."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rotmach.release import verify
if __name__=='__main__':
    print(verify(Path(__file__).resolve().parents[1]))
