"""Validate archive hashes and safely restore fixed inputs and dated forecasts."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 manifest=json.loads((ROOT/'data/artifact_parts.json').read_text());parts=[manifest['input_snapshot'],*manifest['forecast_parts']];count=0
 for spec in parts:
  archive=ROOT/spec['path']
  if sha(archive)!=spec['sha256']:raise ValueError('Archive changed: '+spec['path'])
  with zipfile.ZipFile(archive) as z:
   for info in z.infolist():
    target=(ROOT/info.filename).resolve()
    if not target.is_relative_to(ROOT) or info.filename.startswith('/') or ':' in info.filename:raise ValueError('Unsafe archive path')
    if info.is_dir():continue
    data=z.read(info)
    if target.exists():
     if sha(target)!=hashlib.sha256(data).hexdigest():raise ValueError('Existing file differs; preserving it: '+str(target.relative_to(ROOT)))
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    count+=1
  print('Restored',spec['path'],flush=True)
 print('Validated/restored',count,'files. No training performed.')
if __name__=='__main__':main()
