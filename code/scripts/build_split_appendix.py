"""Export actual per-condition boundaries from verified corrected runs."""
import argparse,json
from pathlib import Path
import pandas as pd
from audit_verification import verify_run

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dirs',nargs='+',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args()
 fields=['dataset','ticker','price_mode','horizon','feature_set','window_size','train_samples','validation_samples','test_samples','train_last_origin','train_last_target','validation_first_origin','validation_last_origin','validation_last_target','test_first_origin','test_last_origin','purged_train_rows','purged_validation_rows']
 frames=[verify_run(run)[0][fields] for run in args.run_dirs]
 exact=pd.concat(frames).drop_duplicates().sort_values(fields[:6]); args.out_dir.mkdir(parents=True,exist_ok=True)
 exact.to_csv(args.out_dir/'split_boundary_exact.csv',index=False)
 summary=exact.groupby(['dataset','feature_set','horizon']).agg(assets=('ticker','nunique'),earliest_validation_origin=('validation_first_origin','min'),latest_validation_origin=('validation_first_origin','max'),earliest_test_origin=('test_first_origin','min'),latest_test_origin=('test_first_origin','max')).reset_index()
 summary.to_csv(args.out_dir/'split_boundary_summary.csv',index=False)
 (args.out_dir/'split_boundary_note.json').write_text(json.dumps({'source':'actual verified model rows','rows':len(exact),'note':'Dates differ by asset, horizon and feature availability. No representative first ticker is substituted for a dataset.'},indent=2),encoding='utf-8');print('Exported',len(exact),'actual split records')
if __name__=='__main__':main()
