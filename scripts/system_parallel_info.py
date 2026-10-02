from pathlib import Path
import os

def physical_cores():
    try:
        pairs=set(); phys=core=None
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('physical id'): phys=int(line.split(':',1)[1])
            elif line.startswith('core id'): core=int(line.split(':',1)[1])
            elif not line.strip() and phys is not None and core is not None:
                pairs.add((phys,core)); phys=core=None
        if phys is not None and core is not None: pairs.add((phys,core))
        if pairs:return len(pairs)
    except Exception: pass
    return max(1,(os.cpu_count() or 2)//2)

logical=os.cpu_count() or 1
physical=physical_cores()
print('logical CPUs :',logical)
print('physical cores:',physical)
print('recommended first try:',physical,'workers')
print('optional HT try       :',min(logical, int(round(physical*1.35))),'workers')
print('BLAS env should all be 1:')
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    print(' ',v,'=',os.environ.get(v,'<unset>'))
