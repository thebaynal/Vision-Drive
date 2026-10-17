"""Controller fixture checks; these are NOT camera/navigation research metrics."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visiondrive.config import Config
from visiondrive.decision import Controller, Detection
from visiondrive.evaluation.data import ROOT, write_json
from visiondrive.evaluation.system import movement_metrics


def run():
    config=Config()
    center=Detection(.40,.45,.60,.85,.99,0,'fixture',1.0)
    left=Detection(.02,.45,.30,.85,.99,0,'fixture',1.0)
    right=Detection(.70,.45,.98,.85,.99,0,'fixture',1.0)
    emergency=Detection(.40,.45,.60,.85,.99,0,'fixture',.2)
    fixtures=[('clear',[],{'FORWARD'},False,True),('center',[center],{'LEFT','RIGHT'},True,True),
              ('left',[left],{'FORWARD'},False,True),('right',[right],{'FORWARD'},False,True),
              ('center_left',[center,left],{'RIGHT'},True,True),('center_right',[center,right],{'LEFT'},True,True),
              ('all_blocked',[left,center,right],{'STOP'},True,False),('emergency',[emergency],{'STOP'},True,False)]
    rows=[]
    for name,detections,acceptable,unsafe,stop_unnecessary in fixtures:
        controller=Controller(config);start=time.perf_counter()
        for i in range(config.stable_frames):decision=controller.update(detections,i,i*.05,i*.05)
        rows.append({'scenario':name,'acceptable_actions':sorted(acceptable),'predicted_action':decision.action.value,
                     'unsafe_forward':unsafe,'stop_unnecessary':stop_unnecessary,'latency_ms':(time.perf_counter()-start)*1000})
    for name in ('invalid_input','stale_input'):
        controller=Controller(config)
        decision=controller.tick(10) if name=='invalid_input' else controller.update([],1,0,10)
        rows.append({'scenario':name,'acceptable_actions':['STOP'],'predicted_action':decision.action.value,
                     'unsafe_forward':True,'stop_unnecessary':False,'latency_ms':None})
    result={'scope':'hand-authored software fixtures, not model or physical driving evidence',
            'pending_reference_scenarios':['moving obstacles','low light with genuine camera labels'],
            'rows':rows,'fixture_metrics':movement_metrics(rows)}
    write_json(ROOT/'outputs/evaluation/yolo11n/controller_fixture_checks.json',result)
    return result


if __name__=='__main__':print(run())
