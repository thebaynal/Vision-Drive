"""Optional independent depth and acceptable-set movement reference scoring."""
import numpy as np


def depth_metrics(predicted, truth, valid, *, convention, min_m=.1, max_m=10.):
    if convention != 'aligned optical-axis z-depth in meters':
        raise ValueError('Confirm aligned optical-axis z-depth in meters; no implicit alignment or scale fitting')
    predicted,truth,valid=np.asarray(predicted),np.asarray(truth),np.asarray(valid,dtype=bool)
    if predicted.shape != truth.shape or truth.shape != valid.shape:
        raise ValueError('Depth arrays and reference validity mask must be aligned and have equal shapes')
    mask=valid & np.isfinite(truth) & (truth>=min_m) & (truth<=max_m)
    if not mask.any():raise ValueError('No valid reference depths')
    if not np.all(np.isfinite(predicted[mask]) & (predicted[mask]>0)):
        raise ValueError('Invalid predictions at valid reference pixels; do not silently discard failures')
    p,t=predicted[mask],truth[mask]
    ratio=np.maximum(p/t,t/p)
    error=p-t
    result={'pixels':int(mask.sum()),'mae_m':float(np.abs(error).mean()),'rmse_m':float(np.sqrt((error**2).mean())),
            'abs_rel':float((np.abs(error)/t).mean()),'delta1':float((ratio<1.25).mean()),
            'delta2':float((ratio<1.25**2).mean()),'delta3':float((ratio<1.25**3).mean())}
    result['distance_bins']={}
    for low,high in ((.1,1.),(1.,2.),(2.,5.),(5.,10.)):
        selected=(t>=low)&(t<high)
        result['distance_bins'][f'{low}-{high}m']={'pixels':int(selected.sum()),
            'mae_m':float(np.abs(error[selected]).mean()) if selected.any() else None}
    return result


def movement_metrics(rows):
    if not rows:raise ValueError('No independent movement references')
    for r in rows:
        if not r['acceptable_actions'] or not set(r['acceptable_actions']) <= {'FORWARD','LEFT','RIGHT','STOP'}:
            raise ValueError('Explicit nonempty acceptable action sets required')
    unsafe=[r for r in rows if r['unsafe_forward']]
    clear=[r for r in rows if r['stop_unnecessary']]
    return {'scenarios':len(rows),'acceptable_set_correctness':sum(r['predicted_action'] in r['acceptable_actions'] for r in rows)/len(rows),
            'unsafe_forward_denominator':len(unsafe),'unsafe_forward_rate':sum(r['predicted_action']=='FORWARD' for r in unsafe)/len(unsafe) if unsafe else None,
            'false_stop_denominator':len(clear),'false_stop_rate':sum(r['predicted_action']=='STOP' for r in clear)/len(clear) if clear else None}
