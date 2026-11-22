"""Independent reconstruction of epoch metrics for checkpoint audits."""
import numpy as np

def independent_metrics(trace):
    def first(key,threshold):
        positions=np.flatnonzero(np.array([r[key] for r in trace])>=threshold)
        return int(positions[0])+1 if len(positions) else None
    result={}
    for key in ['train_online_acc','train_eval_acc','val_acc']:
        for threshold in [95,98,99]:result[f'{key}_{threshold}_epoch']=first(key,threshold/100)
    for threshold in [95,99]:
        arr=np.array([r['val_acc']>=threshold/100 for r in trace])
        windows=np.lib.stride_tricks.sliding_window_view(arr,3) if len(arr)>=3 else np.empty((0,3),bool)
        hit=np.flatnonzero(windows.all(axis=1));result[f'val_{threshold}_sustained_epoch']=int(hit[0])+1 if len(hit) else None
    for label,key in [('paper','train_online_acc'),('eval','train_eval_acc')]:
        a,b=result[f'{key}_99_epoch'],result['val_acc_98_epoch']
        result[f'{label}_delay_epochs']=None if a is None or b is None else max(0,b-a)
    return result

