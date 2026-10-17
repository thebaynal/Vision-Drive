"""Actual unchanged-pretrained inference, native AP and synchronized benchmarks."""
import importlib.metadata
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import cv2
import numpy as np

from .data import ROOT, COCO_NAMES, NAMES, digest, read_jsonl, validate, write_json
from .metrics import aggregate, bootstrap, conditions


def load_config(path=ROOT/'config/evaluation.yaml', device=None):
    config = json.loads(Path(path).read_text())  # JSON is a YAML subset.
    if device is not None:
        config['device'] = device
    return config


def environment(config):
    import torch
    packages = {}
    for package in ('torch','torchvision','ultralytics','numpy','pandas','matplotlib','opencv-python','transformers','nbformat','nbclient'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'python': platform.python_version(), 'os': platform.platform(), 'cpu': platform.processor(),
            'logical_cpus': os.cpu_count(), 'packages': packages, 'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'device': config['device'], 'image_size': config['image_size'], 'batch': config['batch'], 'half': config['half']}


def load_model(config):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.cache/ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO
    import torch
    path = ROOT / config['weights']
    if not path.exists() or digest(path) != config['weights_sha256']:
        raise ValueError('Original YOLO11n checksum required. Run scripts/setup_laptop_models.py; never substitute fine-tuned weights.')
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install matching GPU torch/torchvision or explicitly choose --device cpu.')
    torch.set_num_threads(2)
    model = YOLO(str(path), task='detect')
    model.visiondrive_original_parameters = sum(p.numel() for p in model.model.parameters())
    if [model.names[i] for i in range(80)] != COCO_NAMES:
        raise ValueError('Original 80 COCO class IDs/names required')
    return model


def native_dataset(dataset, output):
    """Materialize absolute local image lists from the frozen relative manifest."""
    yaml_path = output / 'native-dataset.yaml'
    for split in ('dev','test'):
        (output/f'native-{split}.txt').write_text(''.join(str((dataset/r['image']).resolve())+'\n' for r in read_jsonl(dataset/'manifest.jsonl') if r['split']==split),encoding='utf-8')
    write_json(yaml_path, {'path': str(output.resolve()), 'train': 'native-dev.txt', 'val': 'native-dev.txt', 'test': 'native-test.txt', 'names': COCO_NAMES})
    return yaml_path


def native_evaluation(model, config, dataset, output):
    """The 80-class YAML and original class filter are essential to valid IDs."""
    yaml_path = native_dataset(dataset,output)
    metrics = model.val(data=str(yaml_path), split='test', imgsz=config['image_size'], batch=config['batch'],
        device=config['device'], half=config['half'], classes=config['class_ids'],
        conf=config['ap_confidence_floor'], iou=config['nms_iou'], max_det=config['max_detections'],
        plots=True, save_json=False, workers=0, rect=False, verbose=False,
        project=str(output), name='native', exist_ok=True)
    return extract_native(metrics,config)


def extract_native(metrics, config):
    indexes = list(map(int, metrics.box.ap_class_index))
    if set(indexes) != set(config['class_ids']):
        raise ValueError(f'Expected AP for original IDs {config["class_ids"]}; got {indexes}')
    classes = {}
    curves = {}
    for pos, cid in enumerate(indexes):
        ap = np.asarray(metrics.box.all_ap[pos])
        classes[NAMES[cid]] = {'ap50': float(ap[0]), 'ap75': float(ap[5]), 'ap50_95': float(ap.mean())}
        curves[NAMES[cid]] = {key: np.asarray(getattr(metrics.box, attr)[pos]).tolist() for key,attr in
                             [('precision','p_curve'),('recall','r_curve'),('f1','f1_curve'),('pr','prec_values')]}
    return {'map50': float(metrics.box.map50), 'map75': float(metrics.box.all_ap[:,5].mean()),
            'map50_95': float(metrics.box.map), 'per_class': classes,
            'native_validation_speed_ms': {k: float(v) for k,v in metrics.speed.items()}}, curves


def predictions_and_benchmark(model, config, dataset, records):
    import torch
    use_gpu = config['device'] != 'cpu'
    def sync():
        if use_gpu:
            torch.cuda.synchronize()
    kwargs = dict(imgsz=config['image_size'], rect=False, conf=config['ap_confidence_floor'],
                  iou=config['nms_iou'], max_det=config['max_detections'], device=config['device'],
                  half=config['half'], classes=config['class_ids'], verbose=False, save=False)
    first = cv2.imread(str(dataset / records[0]['image']))
    for _ in range(config['benchmark_warmup']):
        model.predict(first, **kwargs)
        sync()
    rows, timing = [], []
    count = min(config['benchmark_frames'],len(records))
    for index, row in enumerate(records):
        image = cv2.imread(str(dataset / row['image']))  # Disk decode excluded.
        sync()
        start = time.perf_counter()
        result = model.predict(image, **kwargs)[0]
        sync()
        elapsed = (time.perf_counter()-start)*1000
        if index < count:
            timing.append({**{k: float(v) for k,v in result.speed.items()}, 'predict_wall': elapsed})
        predictions = [{'class_id': int(cid), 'confidence': float(conf), 'xyxy': list(map(float, box))} for box,conf,cid in
                       zip(result.boxes.xyxyn.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
        rows.append({**row,'predictions': predictions})
    summary = {}
    for phase in timing[0]:
        values = [r[phase] for r in timing]
        summary[phase] = {'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
                          'std_ms': statistics.pstdev(values), 'p95_ms': float(np.percentile(values,95)),
                          'fps_from_mean': 1000/statistics.mean(values)}
    summary.update({'warmup_iterations': config['benchmark_warmup'], 'frames': count, 'batch': 1,
        'cuda_synchronized': use_gpu, 'image_size': config['image_size'], 'half': config['half'],
        'includes': 'in-memory BGR to YOLO preprocessing, network, NMS; predict_wall adds Python overhead',
        'excludes': 'disk decode, camera, depth, display, controller and network; not complete-application FPS'})
    return rows, summary


def evaluate(config=None, model=None):
    config = config or load_config()
    dataset, output = ROOT/config['dataset'], ROOT/config['output']
    output.mkdir(parents=True, exist_ok=True)
    validation = validate(dataset)
    write_json(output/'dataset_validation.json', validation)
    if not validation['valid']:
        raise ValueError(validation['errors'])
    if validation['manifest_sha256'] != config.get('dataset_manifest_sha256'):
        raise ValueError('Frozen manifest checksum mismatch; use an explicit new configuration for another dataset')
    records = [r for r in read_jsonl(dataset/'manifest.jsonl') if r['split']=='test']
    model = model or load_model(config)
    initial_sha = digest(ROOT/config['weights'])
    env = environment(config)
    write_json(output/'environment.json', env)
    write_json(output/'evaluation_config.yaml', config)
    native, curves = native_evaluation(model, config, dataset, output)
    rows, benchmark = predictions_and_benchmark(model, config, dataset, records)
    result = assemble_results(model,config,validation,native,rows,benchmark,initial_sha)
    save_results(result,rows,curves,dataset,output)
    return result


def assemble_results(model,config,validation,native,rows,benchmark,initial_sha):
    env = environment(config)
    fixed = aggregate(rows,config['confidence'],config['matching_iou'],config['class_ids'])
    per_class = {}
    for name, v in fixed['per_class'].items():
        per_class[name] = {**v, **native['per_class'][name], 'ground_truth': v['tp']+v['fn']}
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except (subprocess.SubprocessError,OSError):
        commit, dirty = 'unavailable', True
    source_files = {p.relative_to(ROOT).as_posix(): digest(p) for folder in ('visiondrive/evaluation','scripts')
                    for p in (ROOT/folder).glob('*.py')}
    result = {'protocol':config['protocol'], 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'weights_sha256':initial_sha, 'weights_unchanged': digest(ROOT/config['weights'])==initial_sha,
        'model_parameters': sum(p.numel() for p in model.model.parameters()),
        'model_parameters_before_fusion':model.visiondrive_original_parameters,
        'git_commit': commit, 'working_tree_dirty': dirty,
        'source_sha256':source_files, 'dataset_manifest_sha256': validation['manifest_sha256'],
        'test_images':len(rows), 'development_images':validation['splits']['dev'],
        'configuration':config, 'environment':env, 'fixed_operating_point':fixed, 'native_ap':native,
        'per_class':per_class, 'bootstrap':bootstrap(rows,config['bootstrap_samples'],config['seed'],config['confidence'],config['matching_iou'],config['class_ids']),
        'benchmark':benchmark, 'condition_recall':conditions(rows,config['confidence'],config['matching_iou'],config['class_ids']),
        'limitations':validation['warnings'], 'depth_evaluation':config['depth_evaluation'], 'camera_evaluation':config['camera_evaluation']}
    if not result['weights_unchanged']:
        raise ValueError('Weights changed during evaluation')
    return result


def save_results(result,rows,curves,dataset,output):
    write_json(output/'dataset_validation.json', validate(dataset))
    write_json(output/'environment.json',result['environment'])
    write_json(output/'evaluation_config.yaml',result['configuration'])
    write_json(output/'metrics.json', result)
    write_json(output/'curves.json', curves)
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    from .report import export_report
    export_report(result,rows,curves,dataset,output)
