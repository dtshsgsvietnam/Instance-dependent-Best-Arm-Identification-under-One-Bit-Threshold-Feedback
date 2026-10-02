"""Print the most recently modified ONEBIT trial/log files under ~/onebit."""
from pathlib import Path
import argparse, time
p=argparse.ArgumentParser(); p.add_argument('--root',default='~/onebit'); p.add_argument('--limit',type=int,default=15); a=p.parse_args()
root=Path(a.root).expanduser()
files=[]
for pat in ('**/trials.csv','**/*.log'):
    for f in root.glob(pat):
        try: files.append((f.stat().st_mtime,f))
        except OSError: pass
for mt,f in sorted(files,reverse=True)[:a.limit]:
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(mt)),f)
