"""Class-aware one-to-one operational detection metrics, with group bootstrap."""
from collections import defaultdict
import numpy as np
from .data import iou, NAMES


def ratio(tp, fp, fn):
    # Zero is the declared convention when no prediction/ground truth exists.
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
            'precision': float(tp/(tp+fp)) if tp+fp else 0.,
            'recall': float(tp/(tp+fn)) if tp+fn else 0.,
            'f1': float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.}


def match(row, class_id, confidence=.35, matching_iou=.5):
    truth = [o for o in row['objects'] if o['class_id'] == class_id]
    predictions = sorted([p for p in row['predictions'] if p['class_id'] == class_id and p['confidence'] >= confidence],
                         key=lambda p: -p['confidence'])
    used, matched, false = set(), [], []
    for pred in predictions:
        candidates = [(iou(pred['xyxy'], o['xyxy']), i) for i,o in enumerate(truth) if i not in used]
        overlap, index = max(candidates, default=(0., -1))
        if overlap >= matching_iou:
            used.add(index)
            matched.append({'prediction': pred, 'truth': truth[index], 'iou': overlap})
        else:
            false.append(pred)
    missed = [o for i,o in enumerate(truth) if i not in used]
    return {'matched': matched, 'false_positives': false, 'missed': missed,
            **ratio(len(matched), len(false), len(missed))}


def aggregate(rows, confidence=.35, matching_iou=.5, class_ids=None):
    per_class = {}
    for cid in (NAMES if class_ids is None else class_ids):
        name=NAMES[cid]
        values = [match(row,cid,confidence,matching_iou) for row in rows]
        counts = [sum(v[k] for v in values) for k in ('tp','fp','fn')]
        per_class[name] = ratio(*counts)
    sums = [sum(v[k] for v in per_class.values()) for k in ('tp','fp','fn')]
    micro = ratio(*sums)
    macro = {k: float(np.mean([v[k] for v in per_class.values()])) for k in ('precision','recall','f1')}
    return {'per_class': per_class, 'micro': micro, 'macro': macro}


def bootstrap(rows, repeats=1000, seed=20261017, confidence=.35, matching_iou=.5, class_ids=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    # Precompute group counts so no inference or repeated box matching occurs.
    counts = np.array([[aggregate(g,confidence,matching_iou,class_ids)['micro'][k] for k in ('tp','fp','fn')] for g in groups.values()])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        tp,fp,fn = counts[rng.integers(0,len(counts),len(counts))].sum(axis=0)
        value = ratio(tp,fp,fn)
        values.append([value[k] for k in ('precision','recall','f1')])
    interval = np.percentile(values,[2.5,97.5],axis=0)
    unit = 'recording sessions' if all(r.get('group_type') == 'recording session' for r in rows) else 'photographer groups (scene proxy)'
    return {'method': 'percentile bootstrap over '+unit, 'groups': len(groups),
            'resamples': repeats, 'seed': seed,
            'micro_95ci': {k: interval[:,i].tolist() for i,k in enumerate(('precision','recall','f1'))},
            'limitation': 'Does not account for unidentified cross-photographer scene overlap or COCO pretraining overlap; no AP CI.'}


def conditions(rows, confidence=.35, matching_iou=.5, class_ids=None):
    bins = defaultdict(lambda: [0,0])
    for row in rows:
        for cid in (NAMES if class_ids is None else class_ids):
            result = match(row,cid,confidence,matching_iou)
            matched = {tuple(v['truth']['xyxy']) for v in result['matched']}
            truth = [o for o in row['objects'] if o['class_id'] == cid]
            for o in truth:
                x1,y1,x2,y2 = o['xyxy']
                pixels = (x2-x1)*(y2-y1)*row['width']*row['height']
                labels = ['small_<32px' if pixels < 32**2 else 'medium_32-96px' if pixels < 96**2 else 'large_>=96px',
                          'occluded' if o['occluded'] else 'not_occluded',
                          'truncated' if o['truncated'] else 'not_truncated',
                          'dark_mean_<70' if row['mean_luminance'] < 70 else 'other_luminance',
                          'many_targets_>=5' if len(row['objects']) >= 5 else 'few_targets_<5']
                for label in labels:
                    bins[label][0] += int(tuple(o['xyxy']) in matched)
                    bins[label][1] += 1
    return [{'condition': k, 'tp': v[0], 'ground_truth': v[1], 'recall': v[0]/v[1]} for k,v in sorted(bins.items())]
