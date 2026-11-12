"""Verify shipped evidence, or audit dated predictions with an explicit scope."""
from pathlib import Path
import argparse,hashlib,itertools,json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/scripts'))
from audit_verification import verify_run,SUP_KEYS,ZERO_KEYS
RUNS=['supervised','chronos_sampled']+['chronos_full_'+d for d in ['dow30','nasdaq_large_tech','sp100_largecap','hang_seng_largecap','etf_basket']]
def check(ok,message):
 if not ok:raise ValueError(message)
def verify_package():
 listed={}
 for line in (ROOT/'MANIFEST_SHA256.txt').read_text().splitlines():
  digest,name=line.split('  ',1);p=(ROOT/name).resolve();check(p.is_relative_to(ROOT),'Unsafe manifest path');check(p.is_file(),name+' missing');check(hashlib.sha256(p.read_bytes()).hexdigest()==digest,name+' hash mismatch');listed[name]=digest
 main=pd.read_csv(ROOT/'evidence_package/combined_model_results.csv',low_memory=False);full=pd.read_csv(ROOT/'evidence_package/chronos_full_origin_subset_60_rows.csv');frames={}
 for name in RUNS:
  p=ROOT/'runs'/name; m=json.loads((p/'run_manifest.json').read_text());a=m['arguments'];sup=name=='supervised';keys=SUP_KEYS if sup else ZERO_KEYS
  source=pd.read_csv(p/('model_results.csv' if sup else 'chronos_zero_shot_results.csv'));frames[name]=source;check(not source.duplicated(keys).any(),name+' duplicate key')
  axes=[a[x] for x in ['price_modes','horizons','feature_sets','models','window_sizes','seeds']] if sup else [a['price_modes'],a['horizons']];expected=set()
  for asset in m['input_sha256']:
   dataset,ticker=asset.split('/',1)
   for combo in itertools.product(*axes):expected.add((dataset,ticker,*combo) if sup else (dataset,ticker,*combo,a['model_type'],a['model_id'],a['max_origins']))
  check(set(source[keys].itertuples(index=False,name=None))==expected,name+' incomplete/unexpected experiment keys')
  target=main if name in RUNS[:2] else full;selected=target[target.source_file.eq('runs/'+name)];j=source.merge(selected,on=keys,suffixes=('_run','_combined'),validate='one_to_one');check(len(j)==len(source),name+' missing combined rows')
  for col in ['test_RMSE','test_MAE','test_MAPE','latest_RMSE','latest_MAE','latest_MAPE','RMSE_ratio_vs_latest','MAE_ratio_vs_latest','MAPE_ratio_vs_latest','passes_latest_rmse_gate']:
   check(np.allclose(j[col+'_run'],j[col+'_combined'],rtol=1e-12,atol=1e-12),name+' combined mismatch '+col)
  for metric in ['RMSE','MAE','MAPE']:check(np.allclose(source['test_'+metric]/source['latest_'+metric],source[metric+'_ratio_vs_latest'],rtol=1e-12,atol=1e-12),name+' ratio mismatch')
  check(np.array_equal((source.RMSE_ratio_vs_latest<1).astype(int),source.passes_latest_rmse_gate),name+' strict gate mismatch')
 for df,filename in [(main,'model_gate_summary.csv'),(full,'full_origin_model_gate_summary.csv')]:
  summary=pd.read_csv(ROOT/'evidence_package'/filename);check(df.groupby(['protocol','model']).ngroups==len(summary),filename+' groups')
  for r in summary.itertuples():
   g=df[df.protocol.eq(r.protocol)&df.model.eq(r.model)];check(len(g)==r.rows and int((g.RMSE_ratio_vs_latest<1).sum())==r.passes,filename+' counts');check(int((g[[k+'_ratio_vs_latest' for k in ['RMSE','MAE','MAPE']]]<1).all(axis=1).sum())==r.all_three_passes,filename+' all-three');check(np.isclose(g.RMSE_ratio_vs_latest.median(),r.median_rmse_ratio),filename+' median')
 result={'scope':'Shipped file hashes, complete experiment keys, ratio/gate arithmetic and grouped corrected evidence. This does not recompute dated forecasts; use --all-runs --forecast-only for that.','files_checked':len(listed),'supervised_rows':len(frames['supervised']),'sampled_rows':len(frames['chronos_sampled']),'full_subset_rows':len(full),'observed_rmse_wins':{'supervised':int(frames['supervised'].passes_latest_rmse_gate.sum()),'sampled':int(frames['chronos_sampled'].passes_latest_rmse_gate.sum()),'full_subset':int(full.passes_latest_rmse_gate.sum())}}
 print(json.dumps(result,indent=2));return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path);p.add_argument('--all-runs',action='store_true');p.add_argument('--data-run',type=Path,default=ROOT/'data/input_snapshot');p.add_argument('--forecast-only',action='store_true');p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
 if not a.run_dir and not a.all_runs:verify_package();return
 if a.run_dir and a.all_runs:p.error('Choose --run-dir or --all-runs')
 for run in ([a.run_dir] if a.run_dir else [ROOT/'runs'/n for n in RUNS]):
  df,report=verify_run(run,a.allow_partial,a.data_run,a.forecast_only);out=ROOT/'validation_reports';out.mkdir(exist_ok=True);(out/(run.name+'_verification.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
  print(json.dumps({k:v for k,v in report.items() if k!='traces'},indent=2));print('Observed RMSE wins:',int(df.passes_latest_rmse_gate.sum()),flush=True)
if __name__=='__main__':main()
