"""Serial decode + YOLO/depth + controller + explanation-render benchmark.

This measures actual software processing, not camera/GUI/ESP32 or physical driving.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from visiondrive.config import Config
from visiondrive.decision import Controller
from visiondrive.explain import render_explanation
from visiondrive.evaluation.data import ROOT, read_jsonl, write_json
from visiondrive.evaluation.runner import load_config


def benchmark_system(device='0',frames=20):
    from visiondrive.laptop_detector import LaptopDetector
    import torch
    config=Config.load(ROOT/'config/default.json')
    detector=LaptopDetector(ROOT/'weights/laptop/yolo11n.pt',config,ROOT/'weights/laptop/depth-indoor',device)
    evaluation=load_config();dataset=ROOT/evaluation['dataset']
    records=[r for r in read_jsonl(dataset/'manifest.jsonl') if r['split']=='test'][:frames]
    for _ in range(3):detector.warmup()
    controller=Controller(config)
    def sync():
        if device!='cpu':torch.cuda.synchronize()
    timings=[]
    for i,row in enumerate(records):
        sync();start=time.perf_counter()
        image=cv2.imread(str(dataset/row['image']))
        detections=detector.predict(image)
        now=time.monotonic()
        decision=controller.update(detections,i,now,now)
        canvas=render_explanation(image,decision,config,detector.name,{'status':'running','metric_depth':True},detector.label)
        assert canvas.size
        sync();timings.append((time.perf_counter()-start)*1000)
    result={'measurement':'serial source-image decode + actual YOLO11n + metric depth + controller + explanation render',
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'gpu':torch.cuda.get_device_name(0) if device!='cpu' else None,'torch_version':torch.__version__,
        'frames':len(timings),'warmup_iterations':3,'device':device,'cuda_synchronized':device!='cpu',
        'mean_ms':statistics.mean(timings),'median_ms':statistics.median(timings),'std_ms':statistics.pstdev(timings),
        'fps_from_mean':1000/statistics.mean(timings),'yolo_input':640,'depth_input':392,
        'excludes':'camera capture, asynchronous scheduling, GUI display, UDP, motor execution; not physical navigation success'}
    write_json(ROOT/'outputs/evaluation/yolo11n/system_benchmark.json',result)
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device',default='0');cli.add_argument('--frames',type=int,default=20)
    args=cli.parse_args()
    if args.frames<1:cli.error('--frames must be positive')
    print(benchmark_system(args.device,args.frames))
