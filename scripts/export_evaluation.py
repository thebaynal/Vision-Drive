"""Publish compact measured research evidence from local evaluation outputs.

Raw images, full prediction dumps and native run directories remain ignored.
"""
import csv
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visiondrive.evaluation.data import ROOT, digest, read_jsonl, write_json
from visiondrive.evaluation.metrics import aggregate, match


def retain_supplementary_report(report, previous):
    """Re-exporting person results must retain the separate object benchmark."""
    marker = '## Bottle and cup detection: additional measured benchmark'
    if marker in previous and marker not in report:
        report = report.rstrip() + '\n\n' + marker + previous.split(marker, 1)[1]
    return report


def export():
    source=ROOT/'outputs/evaluation/yolo11n'
    metrics=json.loads((source/'metrics.json').read_text())
    assert metrics['weights_unchanged'] and metrics['configuration']['class_ids']==[0]
    assert metrics['weights_sha256']==digest(ROOT/'weights/laptop/yolo11n.pt')
    assert metrics['dataset_manifest_sha256']==digest(ROOT/'data/public/indoor-evaluation-manifest.jsonl')
    rows=read_jsonl(source/'predictions.jsonl')
    cfg=metrics['configuration']
    actual=aggregate(rows,cfg['confidence'],cfg['matching_iou'],cfg['class_ids'])
    assert actual==metrics['fixed_operating_point']
    target=ROOT/'docs/results/yolo11n';target.mkdir(parents=True,exist_ok=True)
    files=['metrics.json','evaluation_summary.csv','per_class_metrics.csv','dataset_validation.json','dataset_provenance.csv',
           'environment.json','evaluation_config.yaml','condition_recall.csv','system_benchmark.json','controller_fixture_checks.json']
    for name in files:
        if (source/name).exists():shutil.copyfile(source/name,target/name)
    with (target/'per_image_counts.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=['image_id','group_id','tp','fp','fn']);writer.writeheader()
        for row in rows:
            value=match(row,0,cfg['confidence'],cfg['matching_iou'])
            writer.writerow({'image_id':row['image_id'],'group_id':row['group_id'],**{k:value[k] for k in ('tp','fp','fn')}})
    assets=ROOT/'assets/evaluation';assets.mkdir(parents=True,exist_ok=True)
    graphs=['confusion_matrix.png','confusion_matrix_normalized.png','precision_recall_curve.png','f1_confidence_curve.png',
            'class_distribution.png','per_class_performance.png']
    for name in graphs:shutil.copyfile(source/name,assets/name)
    for name in ['missed_1.jpg','false_positive_1.jpg','example_1.jpg']:
        shutil.copyfile(source/'prediction_examples'/name,assets/name)
    index=json.loads((source/'prediction_examples/index.json').read_text())
    write_json(assets/'image-attribution.json',[r for r in index if (assets/r['file']).exists()])
    report=(source/'evaluation_report.md').read_text(encoding='utf-8')
    report+='\n## Submission evidence\n\n[Measured JSON](results/yolo11n/metrics.json) · [Tables](results/yolo11n/per_class_metrics.csv) · [Per-image counts](results/yolo11n/per_image_counts.csv) · [Image/source attribution](results/yolo11n/dataset_provenance.csv) · [Taxonomy exclusion audit](dataset_card.md#chair-stool-taxonomy).\n\n'
    for name in graphs:report+=f'![{name.replace("_"," ").removesuffix(".png")}](../assets/evaluation/{name})\n\n'
    report+='![Actual missed person examples](../assets/evaluation/missed_1.jpg)\n\n![Actual unmatched person predictions](../assets/evaluation/false_positive_1.jpg)\n\nPhoto credits and original pages are recorded in [image-attribution.json](../assets/evaluation/image-attribution.json).\n'
    system=source/'system_benchmark.json'
    if system.exists():
        value=json.loads(system.read_text())
        report+=f'\n## Separate hybrid processing benchmark\n\nOver {value["frames"]} images, serial decode + YOLO11n + depth + controller + render averaged {value["mean_ms"]:.2f} ms ({value["fps_from_mean"]:.2f} FPS). This excludes camera, async scheduling, GUI and motors. It is separate from person-only YOLO timing.\n'
    published = ROOT/'docs/evaluation_results.md'
    previous = published.read_text(encoding='utf-8') if published.exists() else ''
    published.write_text(retain_supplementary_report(report, previous),encoding='utf-8')
    exploratory=ROOT/'outputs/evaluation/exploratory-person-chair'
    if (exploratory/'metrics.json').exists():
        dest=ROOT/'docs/results/exploratory-person-chair';dest.mkdir(parents=True,exist_ok=True)
        original=json.loads((exploratory/'metrics.json').read_text())
        original['status']='EXPLORATORY ONLY: Chair/Stool source taxonomy is incomplete for COCO chair. Not primary evidence.'
        write_json(dest/'metrics.json',original)
        shutil.copyfile(exploratory/'per_class_metrics.csv',dest/'per_class_metrics.csv')
    print('Verified measured counts and exported compact submission evidence.')


if __name__=='__main__':export()
