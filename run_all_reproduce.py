"""Run the corrected audit against a frozen input snapshot. Long-running and resumable."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--data-run',type=Path,required=True)
 p.add_argument('--out-dir',type=Path,required=True)
 p.add_argument('--model-revision',required=True)
 p.add_argument('--device',default='auto')
 args=p.parse_args(); root=Path(__file__).resolve().parent; out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 data=args.data_run.resolve();scripts=root/'code'/'scripts';stages=[]
 common=['--data-run',str(data),'--device',args.device]
 def add(name,command):stages.append((name,[sys.executable,'-u',*map(str,command)]))
 def verify(name):add('verify_'+name,[root/'verify_results.py','--run-dir',out/name,'--data-run',data])
 chronos=['--model-revision',args.model_revision]
 add('chronos_sampled',[scripts/'run_chronos_zero_shot_audit.py',*common,'--out-dir',out/'chronos_sampled',*chronos]);verify('chronos_sampled')
 subsets=[]
 for dataset in ['dow30','nasdaq_large_tech','sp100_largecap','hang_seng_largecap','etf_basket']:
  name='chronos_full_'+dataset;subsets.append(out/name)
  add(name,[scripts/'run_chronos_zero_shot_audit.py',*common,'--out-dir',out/name,*chronos,'--datasets',dataset,'--limit-assets','2','--max-origins','10000','--model-type','chronos-t5-small-full-origin-subset']);verify(name)
 add('aggregate_full_origin',[scripts/'build_cross_market_evidence_tables.py','--run-dirs',*subsets,'--out-dir',out/'full_origin_evidence'])
 add('supervised',[scripts/'run_cross_market_model_audit.py',*common,'--out-dir',out/'supervised','--models','lstm','gru','dlinear','patchtst','itransformer','timemixer','--window-sizes','60','--seeds','42','123','777','--epochs','80','--patience','8','--hidden-size','64','--batch-size','128']);verify('supervised')
 add('aggregate_main',[scripts/'build_cross_market_evidence_tables.py','--run-dirs',out/'supervised',out/'chronos_sampled','--out-dir',out/'corrected_evidence','--expected-supervised-rows','35640','--expected-chronos-rows','990'])
 add('split_appendix',[scripts/'build_split_appendix.py','--run-dirs',out/'supervised','--out-dir',out/'corrected_evidence'])
 records=[]; started=time.time()
 for name,command in stages:
  record=dict(stage=name,command=command,started=time.time(),status='running');records.append(record)
  status=dict(started=started,stages=records,complete=False)
  (out/'pipeline_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  print('START',name,flush=True)
  with (out/(name+'.log')).open('a',encoding='utf-8') as log:
   result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  record.update(returncode=result.returncode,finished=time.time(),status='complete' if result.returncode==0 else 'failed')
  (out/'pipeline_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  if result.returncode:raise SystemExit(f'{name} failed; inspect {out/(name+".log")}')
  print('DONE',name,flush=True)
 status['complete']=True;status['finished']=time.time()
 (out/'pipeline_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
if __name__=='__main__':main()
