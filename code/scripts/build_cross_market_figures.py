"""Plot verified evidence separately by evaluation protocol, retaining every ratio."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from audit_verification import sha,require

def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args()
 manifest=json.loads((args.evidence_dir/'aggregation_manifest.json').read_text());file=args.evidence_dir/'combined_model_results.csv'
 require(sha(file)==manifest['outputs'][file.name],'Evidence changed after verification')
 df=pd.read_csv(file);require(np.isfinite(df.RMSE_ratio_vs_latest).all() and (df.RMSE_ratio_vs_latest>0).all(),'Log chart requires positive finite ratios')
 protocols=list(df.protocol.unique());fig,axes=plt.subplots(1,len(protocols),figsize=(max(6,6*len(protocols)),5),squeeze=False,sharey=True)
 for ax,protocol in zip(axes[0],protocols):
  sub=df[df.protocol.eq(protocol)];models=list(sub.model.unique());values=[sub.loc[sub.model.eq(m),'RMSE_ratio_vs_latest'].to_numpy() for m in models]
  ax.boxplot(values,tick_labels=[m.replace('Linear temporal proxy (legacy dlinear key)','Linear temporal\nproxy').replace('-style','\nstyle') for m in models],showfliers=True,flierprops={'marker':'.','markersize':2},widths=.55)
  ax.set_yscale('log');ax.axhline(1,color='black',linestyle='--',linewidth=1)
  ax.set_title(protocol.replace(' audit','')+f'\n{len(sub):,} experiment rows',fontsize=10);ax.tick_params(axis='x',labelsize=8,rotation=30)
  ax.set_ylim(min(.9,float(df.RMSE_ratio_vs_latest.min())*.9),max(1.1,float(df.RMSE_ratio_vs_latest.max())*1.1));ax.grid(axis='y',alpha=.2)
 axes[0][0].set_ylabel('RMSE / matched persistence RMSE (log scale)')
 fig.text(.5,.01,'Descriptive distributions. Rows are dependent; evaluation protocols differ. No values are clipped.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.04,1,1));args.out_dir.mkdir(parents=True,exist_ok=True)
 for ext in ['pdf','png','svg']:fig.savefig(args.out_dir/f'Figure1_RMSE_Ratio_Distribution.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig);print(args.out_dir)
if __name__=='__main__':main()
