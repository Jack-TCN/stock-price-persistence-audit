"""Verify saved forecasts and input provenance, without prescribing winners."""
from pathlib import Path
import hashlib,itertools,json,io
import numpy as np
import pandas as pd
SUP_KEYS=['dataset','ticker','price_mode','horizon','feature_set','model_type','window_size','seed']
ZERO_KEYS=['dataset','ticker','price_mode','horizon','model_type','model_id','max_origins']
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition: raise ValueError(message)
def metric(y,pred):
    y=np.asarray(y,dtype=float); error=y-np.asarray(pred,dtype=float)
    return dict(RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(np.abs(error))),MAPE=float(np.mean(np.abs(error)/np.maximum(np.abs(y),1e-8))*100))
def verify_run(run_dir,allow_partial=False):
    run_dir=Path(run_dir).resolve(); manifest_path=run_dir/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')); args=manifest['arguments']
    supervised=(run_dir/'model_results.csv').exists(); keys=SUP_KEYS if supervised else ZERO_KEYS
    results=run_dir/('model_results.csv' if supervised else 'chronos_zero_shot_results.csv')
    result_bytes=results.read_bytes()
    df=pd.read_csv(io.BytesIO(result_bytes))
    require(not df.empty and not df.duplicated(keys).any(),'Empty or duplicate results')
    expected=set(); inputs=manifest['input_sha256']
    axes=[args[k] for k in ['price_modes','horizons','feature_sets','models','window_sizes','seeds']] if supervised else [args['price_modes'],args['horizons']]
    for asset in inputs:
        dataset,ticker=asset.split('/',1)
        for combo in itertools.product(*axes):
            values=(dataset,ticker,*combo) if supervised else (dataset,ticker,*combo,args['model_type'],args['model_id'],args['max_origins'])
            expected.add(tuple(map(str,values)))
    actual={tuple(str(r[k]) for k in keys) for r in df.to_dict('records')}
    require(actual<=expected,'Unexpected experiment configuration')
    require(allow_partial or actual==expected,f'Incomplete run: {len(actual)}/{len(expected)}')
    raw_cache={}; source_root=Path(args['data_run'])
    require(source_root.is_absolute(),'Verification requires absolute data-run path')
    for asset,digest in inputs.items():
        dataset,ticker=asset.split('/',1); path=source_root/'raw'/dataset/(ticker.replace('/','_')+'.csv')
        require(sha(path)==digest,f'Input changed: {asset}')
        raw=pd.read_csv(path,parse_dates=['Date'])
        require(raw.Date.is_monotonic_increasing and not raw.Date.duplicated().any(),f'Invalid dates: {asset}')
        raw_cache[asset]=raw
    traces=[]
    origin_cache={}
    for row in df.to_dict('records'):
        key='_'.join(str(row[k]) for k in keys)
        name=key if supervised else hashlib.sha256(key.encode()).hexdigest()
        path=run_dir/'predictions'/(name+'.csv')
        pred=pd.read_csv(path,parse_dates=['origin_date','target_date'])
        require(not pred.empty and pred.origin_date.is_monotonic_increasing and not pred.origin_date.duplicated().any(),f'Invalid origins: {key}')
        require(np.isfinite(pred[['target','prediction','persistence']].to_numpy()).all(),f'Nonfinite trace: {key}')
        require(len(pred)==int(row['test_samples' if supervised else 'test_origins']),f'Count mismatch: {key}')
        raw=raw_cache[f"{row['dataset']}/{row['ticker']}"]; close=raw.Close.to_numpy(float)
        if row['price_mode']=='adjusted_close': close=close*(raw['Adj Close'].to_numpy(float)/close)
        offsets=pd.Index(raw.Date).get_indexer(pred.origin_date); target_offsets=offsets+int(row['horizon'])
        require((offsets>=0).all() and (target_offsets<len(raw)).all(),f'Invalid target offset: {key}')
        condition=(row['dataset'],row['ticker'],row['price_mode'],int(row['horizon']),row.get('feature_set','zero_shot'))
        if condition not in origin_cache:
            usable=pd.DataFrame({'Close':close,'Target':pd.Series(close).shift(-int(row['horizon']))})
            if supervised:
                ratio=raw['Adj Close'].to_numpy(float)/raw.Close.to_numpy(float) if row['price_mode']=='adjusted_close' else np.ones(len(raw))
                for column in ['Open','High','Low']:usable[column]=raw[column].to_numpy(float)*ratio
                usable['Volume']=raw.Volume.to_numpy(float)
                if row['feature_set']=='technical':
                    from ta.momentum import RSIIndicator
                    from ta.trend import MACD
                    from ta.volatility import BollingerBands
                    series=pd.Series(close);macd=MACD(series);bands=BollingerBands(series)
                    usable['rsi']=RSIIndicator(series).rsi()
                    usable['macd']=macd.macd();usable['signal']=macd.macd_signal();usable['diff']=macd.macd_diff()
                    usable['bb_mean']=bands.bollinger_mavg();usable['bb_high']=bands.bollinger_hband();usable['bb_low']=bands.bollinger_lband()
            frame=usable.replace([np.inf,-np.inf],np.nan).dropna()
            split=manifest['protocol']['split_ratios']
            cut=int(len(frame)*(float(split['train'])+float(split['validation'])))
            candidates=frame.index.to_numpy()[cut:]
            if not supervised and len(candidates)>int(args['max_origins']):
                candidates=candidates[np.unique(np.linspace(0,len(candidates)-1,int(args['max_origins'])).round().astype(int))]
            origin_cache[condition]=candidates
        require(np.array_equal(offsets,origin_cache[condition]),f'Origins do not follow declared split/sampling protocol: {key}')
        require(np.array_equal(pred.target_date.to_numpy(),raw.Date.iloc[target_offsets].to_numpy()),f'Target date mismatch: {key}')
        require(np.allclose(pred.target,close[target_offsets],rtol=1e-12,atol=1e-10),f'Target data mismatch: {key}')
        require(np.allclose(pred.persistence,close[offsets],rtol=1e-12,atol=1e-10),f'Persistence data mismatch: {key}')
        model=metric(pred.target,pred.prediction); baseline=metric(pred.target,pred.persistence)
        for m in model:
            require(np.isclose(model[m],row['test_'+m],rtol=1e-9,atol=1e-9),f'Model {m} mismatch: {key}')
            require(np.isclose(baseline[m],row['latest_'+m],rtol=1e-9,atol=1e-9),f'Baseline {m} mismatch: {key}')
            require(baseline[m]>0 and np.isclose(model[m]/baseline[m],row[m+'_ratio_vs_latest'],rtol=1e-9,atol=1e-9),f'Ratio {m} mismatch: {key}')
        require(int(model['RMSE']<baseline['RMSE'])==int(row['passes_latest_rmse_gate']),f'Gate mismatch: {key}')
        hashes={'prediction_sha256':sha(path)}
        if supervised:
            require(pd.Timestamp(row['train_last_target'])<pd.Timestamp(row['validation_first_origin']),f'Train label crosses boundary: {key}')
            require(pd.Timestamp(row['validation_last_target'])<pd.Timestamp(row['test_first_origin']),f'Validation label crosses boundary: {key}')
            require(pred.origin_date.iloc[0]==pd.Timestamp(row['test_first_origin']) and pred.origin_date.iloc[-1]==pd.Timestamp(row['test_last_origin']),f'Test range mismatch: {key}')
            hashes.update(checkpoint_sha256=sha(path.with_suffix('.pt')),scaler_sha256=sha(path.with_suffix('.scaler.joblib')))
        traces.append(dict(key=key,**hashes))
    report=dict(run_dir=str(run_dir),rows=len(df),expected_rows=len(expected),complete=actual==expected,result_sha256=hashlib.sha256(result_bytes).hexdigest(),manifest_sha256=sha(manifest_path),verifier_sha256=sha(Path(__file__)),declared_origin_selection_verified=True,traces=traces,boundary='Saved prediction arithmetic, declared origin selection and input provenance; not independent training reproduction or statistical inference')
    return df,report
