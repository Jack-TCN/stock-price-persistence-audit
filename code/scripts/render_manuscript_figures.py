"""Render the manuscript figures from the corrected public evidence."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args()
ROOT=args.repo_root.resolve();T=args.out_dir.resolve().parent
args.out_dir.mkdir(parents=True,exist_ok=True)
main=pd.read_csv(ROOT/'evidence_package/combined_model_results.csv',low_memory=False)
full=pd.read_csv(ROOT/'evidence_package/chronos_full_origin_subset_60_rows.csv')
zero=main[main.protocol.str.contains('sampled')]
order=['LSTM','GRU','Linear temporal proxy (legacy dlinear key)','PatchTST-style','iTransformer-style','TimeMixer-style']
labels={'Linear temporal proxy (legacy dlinear key)':'Linear temporal proxy'}
# Updated distribution figure; separate axes avoid hiding Chronos variation.
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,1,figsize=(7,6.4),gridspec_kw={'height_ratios':[2.1,1.1]})
vals=[main.loc[main.model.eq(m),'RMSE_ratio_vs_latest'].to_numpy() for m in order]
axes[0].boxplot(vals,vert=False,tick_labels=[labels.get(m,m) for m in order],flierprops={'marker':'.','markersize':2},widths=.5)
axes[0].set_xscale('log');axes[0].invert_yaxis();axes[0].set_title('(a) Supervised: fixed fitted models, all test origins',loc='left');axes[0].set_xlabel('Model RMSE divided by matched persistence RMSE (log scale)')
axes[1].boxplot([zero.RMSE_ratio_vs_latest.to_numpy(),full.RMSE_ratio_vs_latest.to_numpy()],vert=False,tick_labels=['Chronos sampled (990)','Chronos full subset (60)'],flierprops={'marker':'.','markersize':3},widths=.45)
axes[1].invert_yaxis();axes[1].set_title('(b) Chronos: different origin protocols',loc='left');axes[1].set_xlabel('Model RMSE divided by matched persistence RMSE (linear scale)')
for ax in axes:
    ax.axvline(1,color='black',ls='--',lw=1);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.tight_layout(h_pad=2.2);fdir=args.out_dir;fdir.mkdir(exist_ok=True)
for ext in ['pdf','png']:fig.savefig(fdir/f'corrected_rmse_distribution_20261112.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
# A compact workflow makes the shared baseline and protocol split explicit.
fig,ax=plt.subplots(figsize=(7.2,4));ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
def box(x,y,w,h,text,color='#eef3f8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=color,edgecolor='#4b5d70',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#4b5d70',lw=1.2))
box(2.8,4.9,4.4,.7,'Fixed daily data snapshot\n165 entries; 133 unique tickers')
box(.3,2.4,4.2,1.55,'Supervised: 60-day multivariate windows\nOwn usable-data splits; boundary purge\nFit on training; select on validation\nEvaluate all eligible test origins')
box(5.4,2.4,4.2,1.55,'Chronos: univariate context up to 512\nNo asset-specific fitting\nUp to 96 test origins per condition\nSmall full-origin sensitivity subset')
arrow((4.1,4.8),(2.4,4.05));arrow((5.9,4.8),(7.5,4.05))
box(1.4,.2,7.2,1.05,'For each model: persistence on the same origins and target dates\nRecompute errors and strict RMSE ratio < 1\nReport descriptive results separately by protocol',color='#eaf4eb');arrow((2.4,2.3),(3.6,1.35));arrow((7.5,2.3),(6.5,1.35))
fig.tight_layout(pad=.2)
for ext in ['pdf','png']:fig.savefig(fdir/f'corrected_workflow_20261112.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
