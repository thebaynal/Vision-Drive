"""Readable graphs, tables, prediction evidence and a measured research report."""
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .data import NAMES, read_jsonl
from .metrics import match


def table(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'figure.dpi':130, 'savefig.dpi':160,
                         'axes.spines.top':False, 'axes.spines.right':False})
    return plt


def annotated(image, objects, predictions=False, threshold=.35):
    image = image.copy()
    h,w = image.shape[:2]
    for obj in objects:
        if predictions and obj['confidence'] < threshold:
            continue
        x1,y1,x2,y2 = [int(v*s) for v,s in zip(obj['xyxy'],(w,h,w,h))]
        name = NAMES[obj['class_id']]
        if predictions:
            name += f' {obj["confidence"]:.2f}'
        color = (0,170,20) if not predictions else (20,110,240)
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        cv2.putText(image,name,(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def exploration(dataset, output):
    plt = plot_setup()
    output.mkdir(parents=True,exist_ok=True)
    rows = read_jsonl(dataset/'manifest.jsonl')
    counts = Counter((r['split'],o['label']) for r in rows for o in r['objects'])
    names=[name for name in NAMES.values() if any(o['label']==name for r in rows for o in r['objects'])]
    fig,ax = plt.subplots(figsize=(7,4))
    positions = np.arange(len(names))
    for shift,split,color in [(-.18,'dev','#3b82f6'),(.18,'test','#0f766e')]:
        bars = ax.bar(positions+shift,[counts[split,n] for n in names],.36,label=split,color=color)
        ax.bar_label(bars,padding=3)
    ax.set(xticks=positions,xticklabels=names,ylabel='Annotated instances',title='Indoor evaluation cohort: genuine source annotations')
    ax.legend(); ax.margins(y=.18)
    fig.tight_layout();fig.savefig(output/'class_distribution.png');plt.close(fig)
    # All images are tiled in small, source-ID-labelled sheets for pre-inference QA.
    qa = output/'ground_truth_review';qa.mkdir(exist_ok=True)
    for start in range(0,len(rows),12):
        batch = rows[start:start+12]
        fig,axes = plt.subplots(3,4,figsize=(16,10))
        for ax,row in zip(axes.flat,batch):
            image = cv2.imread(str(dataset/row['image']))
            ax.imshow(annotated(image,row['objects']));ax.set_title(row['image_id'],fontsize=9);ax.axis('off')
        for ax in list(axes.flat)[len(batch):]:ax.axis('off')
        fig.suptitle(f'Source ground truth — original COCO IDs — images {start+1}–{start+len(batch)}')
        fig.tight_layout();fig.savefig(qa/f'sheet_{start//12+1:02d}.jpg',dpi=110);plt.close(fig)
    return dict(counts)


def export_report(result,rows,curves,dataset,output):
    plt = plot_setup()
    exploration(dataset,output)
    per_class = [{'class':name,**values} for name,values in result['per_class'].items()]
    table(output/'per_class_metrics.csv',per_class)
    fixed = result['fixed_operating_point']
    summary = {**fixed['micro'],**{k:result['native_ap'][k] for k in ('map50','map75','map50_95')},
               'macro_precision':fixed['macro']['precision'],'macro_recall':fixed['macro']['recall'],'macro_f1':fixed['macro']['f1'],
               'test_images':len(rows),'mean_inference_ms':result['benchmark']['inference']['mean_ms']}
    table(output/'evaluation_summary.csv',[summary])
    all_rows = read_jsonl(dataset/'manifest.jsonl')
    provenance = [{'image_id':r['image_id'],'split':r['split'],'original_split':r['source']['original_split'],
                   'sha256':r['sha256'],'author':r['source']['author'],'source_url':r['source']['url'],
                   'license':r['source']['license'],'download_url':r['source']['download_url'],
                   'transformation':'none; normalized source boxes to YOLO; person hierarchy propagation'} for r in all_rows]
    table(output/'dataset_provenance.csv',provenance)
    table(output/'condition_recall.csv',result['condition_recall'])
    names = list(result['per_class'])
    count=len(names)
    # Class-aware TP/FP/FN matrix. Cross-class predictions remain separate FP+FN,
    # exactly as in the declared fixed-point scorer; do not imply class accuracy.
    matrix = np.zeros((count+1,count+1))
    for i,name in enumerate(names):
        v = result['per_class'][name]
        matrix[i,i],matrix[i,count],matrix[count,i] = v['tp'],v['fn'],v['fp']
    for normalized in (False,True):
        values = matrix / np.maximum(matrix.sum(axis=1,keepdims=True),1) if normalized else matrix
        fig,ax = plt.subplots(figsize=(7.5,5.6))
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if normalized else None)
        for i in range(count+1):
            for j in range(count+1):
                ax.text(j,i,f'{values[i,j]:.2f}' if normalized else f'{int(values[i,j])}',ha='center',va='center',
                        color='white' if values[i,j] > (1 if normalized else values.max())*.55 else 'black')
        ax.set(xticks=range(count+1),yticks=range(count+1),xticklabels=names+['missed'],yticklabels=names+['unmatched prediction'],
               xlabel='Predicted class / missed ground truth',ylabel='Ground truth / unmatched prediction',
               title='Class-aware detection confusion matrix'+ (' (row normalized)' if normalized else ' (counts)'))
        fig.colorbar(im,ax=ax,fraction=.045)
        fig.text(.5,.015,'Confidence ≥ 0.35; same-class IoU ≥ 0.50. Wrong classes contribute FP + FN.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(output/('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'));plt.close(fig)
    for filename,title,xlabel,ylabel,key in [('precision_recall_curve.png','Pretrained YOLO11n: PR at IoU 0.50','Recall','Precision','pr'),
                                           ('f1_confidence_curve.png','Pretrained YOLO11n: confidence sweep','Confidence','F1 at IoU 0.50','f1')]:
        fig,ax=plt.subplots(figsize=(7,4.7))
        for name,curve in curves.items():
            y = curve[key]
            ax.plot(np.linspace(0,1,len(y)),y,label=name,linewidth=2)
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel=xlabel,ylabel=ylabel,title=title)
        ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/filename);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    x=np.arange(len(names))
    for shift,key,label,color in [(-.18,'ap50','AP@50','#3b82f6'),(.18,'ap50_95','AP@50–95','#0f766e')]:
        bars=ax.bar(x+shift,[result['per_class'][n][key] for n in names],.36,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set(xticks=x,xticklabels=names,ylim=(0,1.08),ylabel='Average precision',title='Unchanged YOLO11n: per-class test performance')
    ax.legend();fig.tight_layout();fig.savefig(output/'per_class_performance.png');plt.close(fig)
    examples=output/'prediction_examples';examples.mkdir(exist_ok=True)
    conf=result['configuration']['confidence'];miou=result['configuration']['matching_iou']
    scored=[]
    for row in rows:
        values=[match(row,cid,conf,miou) for cid in result['configuration']['class_ids']]
        scored.append((sum(v['fn'] for v in values),sum(v['fp'] for v in values),row))
    selected=[]
    for label,sort_key in [('missed',lambda x:-x[0]),('false_positive',lambda x:-x[1])]:
        for index,(fn,fp,row) in enumerate(sorted(scored,key=sort_key)[:3]):
            selected.append((f'{label}_{index+1}',row,fn,fp))
    for index,row in enumerate(rows[:3]):selected.append((f'example_{index+1}',row,None,None))
    example_index=[]
    for label,row,fn,fp in selected:
        image=cv2.imread(str(dataset/row['image']))
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,objects,pred,title in [(axes[0],row['objects'],False,'Source ground truth'),(axes[1],row['predictions'],True,'Unchanged YOLO11n: confidence ≥ .35')]:
            objects=[o for o in objects if o['class_id'] in result['configuration']['class_ids']]
            ax.imshow(annotated(image,objects,pred,conf));ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{row["image_id"]}'+(f' — missed={fn}, false positives={fp}' if fn is not None else ''))
        fig.text(.5,.01,f'Photo: {row["source"]["author"]} · CC BY 2.0 · Source URL in dataset_provenance.csv',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(examples/f'{label}.jpg',dpi=130);plt.close(fig)
        example_index.append({'file':f'{label}.jpg','image_id':row['image_id'],'false_negatives':fn,'false_positives':fp,'source':row['source']})
    (examples/'index.json').write_text(json.dumps(example_index,indent=2),encoding='utf-8')
    p=fixed['micro'];ap=result['native_ap'];bench=result['benchmark'];env=result['environment']
    lines=['# Pretrained YOLO11n indoor evaluation','',
        '## 1. Objectives','Measure unchanged COCO-pretrained object recognition in real indoor photographs. This is not a collision-avoidance trial.','',
        '## 2. Dataset and sources',
        f'{len(all_rows)} validated images: {result["development_images"]} development, {len(rows)} test. Original Open Images validation/test assignments are retained. Fresh Building-family interiors have verified person presence/absence; prior inspected and baseline images are excluded. There is one image per photographer and 75% person-positive / 25% verified person-negative images in each split. Only PERSON is primary-scored; chair is excluded because source Stool and Chair do not exhaustively cover the COCO chair concept.',
        'Photos: individual authors, CC BY 2.0; annotations: Google/Open Images, CC BY 4.0. Attribution is in `dataset_provenance.csv`.',
        '[Open Images annotation protocol](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) · [Downloads and attributes](https://storage.googleapis.com/openimages/web/download_v7.html).','',
        '## 3. Model',f'Original YOLO11n COCO checkpoint SHA-256: `{result["weights_sha256"]}`. No training or fine-tuning. Primary ID person=0; all 80 model names are retained. Parameters before fusion: {result["model_parameters_before_fusion"]:,}; after inference fusion: {result["model_parameters"]:,}.','',
        '## 4. Experimental setup',f'Input 640×640 letterboxed, FP32, batch 1. Device: {env["gpu"] if env["device"] != "cpu" else env["cpu"]}. Environment and source digests are recorded in JSON. Git: `{result["git_commit"]}`; dirty={result["working_tree_dirty"]}.','',
        '## 5. Ground truth',
        'Human-verified person image labels and exhaustive source boxes are required. Whole images with target group-of, depiction, inside-target, unknown flags or label conflicts are excluded. Person subcategories are propagated, coincident cross-category boxes merged at IoU ≥ .95. Remaining cross-source person overlaps at IoU > .30 cause whole-image exclusion. This rule was frozen from pilot QA before fresh-cohort inference. Source protocol and visual spot checks are not independent exhaustive reannotation. No model-generated ground truth.','',
        '## 6. Definitions',
        'Operational P/R/F1 use confidence ≥ .35 and greedy confidence-ordered same-class one-to-one IoU ≥ .50 matching. Undefined ratios are zero. Micro sums counts; macro F1 is the arithmetic mean of class F1, not the harmonic mean of macro P/R. AP is native Ultralytics integration at confidence floor .001, NMS IoU .70, max 300 detections. Only person is primary-scored, so mAP equals person AP. It is not classification accuracy.','',
        '## 7. Overall results',f'Micro precision {p["precision"]:.4f}; recall {p["recall"]:.4f}; F1 {p["f1"]:.4f}; TP={p["tp"]}, FP={p["fp"]}, FN={p["fn"]}.',
        f'mAP@50={ap["map50"]:.4f}; mAP@75={ap["map75"]:.4f}; mAP@50–95={ap["map50_95"]:.4f}. Macro F1={fixed["macro"]["f1"]:.4f}.',
        f'95% group-bootstrap intervals for micro P/R/F1: {result["bootstrap"]["micro_95ci"]}. Resampling unit: photographer (scene proxy), {result["bootstrap"]["groups"]} test groups, 1,000 resamples. No AP interval claimed.','',
        '## 8. Per-class results','| Class | GT | TP | FP | FN | P | R | F1 | AP50 | AP75 | AP50–95 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in result['per_class'].items():
        lines.append(f'| {name} | {v["ground_truth"]} | {v["tp"]} | {v["fp"]} | {v["fn"]} | {v["precision"]:.3f} | {v["recall"]:.3f} | {v["f1"]:.3f} | {v["ap50"]:.3f} | {v["ap75"]:.3f} | {v["ap50_95"]:.3f} |')
    lines += ['','## 9. Error analysis','Measured condition breakdown (recall; observational and confounded, not causal):',
        '| Condition | TP / ground truth | Recall |','|---|---:|---:|']
    for v in result['condition_recall']:lines.append(f'| {v["condition"]} | {v["tp"]}/{v["ground_truth"]} | {v["recall"]:.3f} |')
    lines += ['','Miss and false-positive examples are exported in `prediction_examples/`. Object overlap/background complexity and perspective have no independent condition labels; the example sheets support qualitative review, not invented quantitative claims.',
        '','## 10. Computational performance',
        f'After {bench["warmup_iterations"]} warmups, {bench["frames"]} in-memory frames: model inference mean {bench["inference"]["mean_ms"]:.2f} ms, median {bench["inference"]["median_ms"]:.2f} ms, population SD {bench["inference"]["std_ms"]:.2f} ms, model-only throughput {bench["inference"]["fps_from_mean"]:.1f} FPS.',
        f'Preprocessing {bench["preprocess"]["mean_ms"]:.2f} ms; postprocessing {bench["postprocess"]["mean_ms"]:.2f} ms; Python predict-call wall mean {bench["predict_wall"]["mean_ms"]:.2f} ms ({bench["predict_wall"]["fps_from_mean"]:.1f} FPS). CUDA synchronization: {bench["cuda_synchronized"]}. Disk decode, capture, depth, display, UDP and motor execution excluded.','',
        '## 11. Threats to validity',
        'The 400-image target is a practical sample, not proof of generalization. Selection is stratified to 75% person-positive images, so precision depends on this chosen prevalence. Only person is primary-scored. Pilot figure QA found a Chair/Stool taxonomy mismatch; the pilot is exploratory, and the corrected person-only protocol was frozen before inference on the fresh cohort. Neither thresholds nor image selection were optimized from final test scores. Public interior photos differ from a low-mounted car camera. Exact/near duplicates and shared photographers are controlled within the cohort, but physical scene identities and COCO pretraining overlap are unproven. Published COCO benchmarks are external references only.',
        f'Depth evaluation: {result["depth_evaluation"]}. Camera evaluation: {result["camera_evaluation"]}.','',
        '## 12. Conclusions',
        'These measurements describe pretrained person localization on this explicitly selected indoor cohort. They do not validate furniture recognition or all-obstacle detection. Missed people and false detections remain. Depth cues and safeguarded movement recommendations need separate reference measurements and physical trials before navigation performance can be claimed.','']
    (output/'evaluation_report.md').write_text('\n'.join(lines),encoding='utf-8')
