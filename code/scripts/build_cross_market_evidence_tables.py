"""Aggregate explicit, verified run directories; never read historical runs implicitly."""
from pathlib import Path
import argparse,json
import pandas as pd
from audit_verification import verify_run,require,sha
MODEL_LABELS={'lstm':'LSTM','gru':'GRU','dlinear':'Linear temporal proxy (legacy dlinear key)','patchtst':'PatchTST-style','itransformer':'iTransformer-style','timemixer':'TimeMixer-style','chronos-t5-small':'Chronos-T5-small zero-shot','chronos-t5-small-full-origin-subset':'Chronos-T5-small full-origin subset'}
DATASET_LABELS={'dow30':'Dow 30','nasdaq_large_tech':'Nasdaq large-tech','sp100_largecap':'S&P 100 subset','hang_seng_largecap':'Hang Seng large-cap','etf_basket':'ETF basket'}
def summarize(df,columns):
    rows=[]
    for key,g in df.groupby(columns,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        ratio=g.RMSE_ratio_vs_latest
        rows.append(dict(zip(columns,key))|dict(rows=len(g),passes=int(g.passes_latest_rmse_gate.sum()),pass_rate=float(g.passes_latest_rmse_gate.mean()),all_three_passes=int(g.passes_all_three.sum()),median_rmse_ratio=float(ratio.median()),mean_rmse_ratio=float(ratio.mean()),q1_rmse_ratio=float(ratio.quantile(.25)),q3_rmse_ratio=float(ratio.quantile(.75)),min_rmse_ratio=float(ratio.min())))
    return pd.DataFrame(rows)
def esc(s):
    for a,b in [('&',r'\&'),('_',r'\_'),('%',r'\%'),('#',r'\#')]:s=str(s).replace(a,b)
    return s

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dirs',nargs='+',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--expected-supervised-rows',type=int)
    p.add_argument('--expected-chronos-rows',type=int)
    args=p.parse_args(); out=args.out_dir.resolve()
    historical=Path(__file__).resolve().parents[2]/'evidence_package'
    require(out!=historical.resolve(),'Choose a new output directory; preserve historical evidence')
    frames=[];reports=[];coverage=[]
    for directory in args.run_dirs:
        frame,report=verify_run(directory)
        manifest=json.loads((directory/'run_manifest.json').read_text())
        supervised='feature_set' in frame.columns
        frame['protocol']='full rolling-origin supervised audit' if supervised else ('full-origin zero-shot subset' if int(manifest['arguments']['max_origins'])>=10000 else 'sampled-origin zero-shot audit')
        frame['model']=frame.model_type.map(MODEL_LABELS).fillna(frame.model_type)
        frame['dataset_label']=frame.dataset.map(DATASET_LABELS).fillna(frame.dataset)
        frame['test_units']=frame['test_samples' if supervised else 'test_origins']
        frame['passes_all_three']=(frame[[x+'_ratio_vs_latest' for x in ['RMSE','MAE','MAPE']]]<1).all(axis=1).astype(int)
        frame['source_file']=str(directory.resolve());frames.append(frame);reports.append(report)
        source=pd.read_csv(Path(manifest['arguments']['data_run'])/'dataset_download_summary.csv')
        assets=set(manifest['input_sha256'])
        coverage.append(source[[f'{r.dataset}/{r.ticker}' in assets for r in source.itertuples()]])
    combined=pd.concat(frames,ignore_index=True)
    keys=['protocol','dataset','ticker','price_mode','horizon','model_type','feature_set','window_size','seed']
    for k in keys:
        if k not in combined:combined[k]=''
    require(not combined.duplicated(keys).any(),'Duplicate experiment keys across run directories')
    n_super=int(combined.protocol.str.contains('supervised').sum());n_zero=int(combined.protocol.eq('sampled-origin zero-shot audit').sum())
    if args.expected_supervised_rows is not None:require(n_super==args.expected_supervised_rows,'Unexpected supervised scope')
    if args.expected_chronos_rows is not None:require(n_zero==args.expected_chronos_rows,'Unexpected Chronos scope')
    out.mkdir(parents=True,exist_ok=True)
    combined.to_csv(out/'combined_model_results.csv',index=False)
    cov=pd.concat(coverage).drop_duplicates(['dataset','ticker']).groupby('dataset').agg(assets=('ticker','nunique'),start=('start','min'),end=('end','max')).reset_index()
    cov['dataset_label']=cov.dataset.map(DATASET_LABELS);cov.to_csv(out/'dataset_coverage.csv',index=False)
    mappings={'model_gate_summary':['protocol','model'],'model_horizon_summary':['protocol','model','horizon'],'dataset_model_summary':['dataset_label','protocol','model'],'protocol_model_summary':['protocol','model']}
    for name,cols in mappings.items():summarize(combined,cols).to_csv(out/(name+'.csv'),index=False)
    combined.sort_values('RMSE_ratio_vs_latest').head(100).to_csv(out/'best_100_model_rows.csv',index=False)
    combined.sort_values('RMSE_ratio_vs_latest',ascending=False).head(100).to_csv(out/'worst_100_model_rows.csv',index=False)
    summary=summarize(combined,['protocol','model'])
    tex=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Protocol & Model & Rows & RMSE passes & All-three passes & Median RMSE ratio \\',r'\midrule']
    for r in summary.itertuples():tex.append(f'{esc(r.protocol)} & {esc(r.model)} & {r.rows} & {r.passes} & {r.all_three_passes} & {r.median_rmse_ratio:.3f} '+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}','% Descriptive counts; correlated rows are not independent trials.']
    (out/'core_tables.tex').write_text('\n'.join(tex),encoding='utf-8')
    result=dict(scope='Selected run manifests only; a complete pilot is not the complete paper experiment',supervised_rows=n_super,sampled_chronos_rows=n_zero,run_reports=reports,inference='No row-independent confidence interval or p-value is generated',outputs={p.name:sha(p) for p in out.glob('*.csv')})
    (out/'aggregation_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(out)
if __name__=='__main__':main()
