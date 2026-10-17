import numpy as np
import pytest
from visiondrive.evaluation.system import depth_metrics,movement_metrics
from scripts.import_camera_dataset import import_recordings


def test_depth_reference_convention_invalid_predictions_and_known_errors():
    with pytest.raises(ValueError,match='Confirm aligned'):
        depth_metrics([1],[1],[True],convention='unknown')
    metrics=depth_metrics(np.array([1.,3.]),np.array([1.,2.]),[True,True],convention='aligned optical-axis z-depth in meters')
    assert metrics['mae_m']==.5 and metrics['rmse_m']==pytest.approx(np.sqrt(.5)) and metrics['abs_rel']==.25
    with pytest.raises(ValueError,match='Invalid predictions'):
        depth_metrics([float('nan')],[1],[True],convention='aligned optical-axis z-depth in meters')


def test_movement_accepts_both_safe_turns_and_defines_denominators():
    rows=[{'acceptable_actions':['LEFT','RIGHT'],'predicted_action':'RIGHT','unsafe_forward':True,'stop_unnecessary':True},
          {'acceptable_actions':['STOP'],'predicted_action':'FORWARD','unsafe_forward':True,'stop_unnecessary':False}]
    result=movement_metrics(rows)
    assert result['acceptable_set_correctness']==.5 and result['unsafe_forward_rate']==.5
    assert result['false_stop_rate']==0 and result['false_stop_denominator']==1


def test_camera_import_requires_complete_annotation_declaration(tmp_path):
    import cv2,json
    cv2.imwrite(str(tmp_path/'frame.png'),np.zeros((32,32,3),dtype=np.uint8))
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'image':'frame.png','split':'dev','session_id':'one',
                                   'objects':[],'exhaustive_classes':['person']}))
    with pytest.raises(ValueError,match='exhaustive'):
        import_recordings(manifest,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_camera_import_converts_reviewed_boxes_and_keeps_recording_sessions(tmp_path):
    import cv2,json
    rng=np.random.default_rng(99)
    rows=[]
    for i,split in enumerate(('dev','test')):
        cv2.imwrite(str(tmp_path/f'{i}.png'),rng.integers(0,256,(64,64,3),dtype=np.uint8))
        rows.append({'image':f'{i}.png','split':split,'session_id':f'session{i}',
                     'exhaustive_classes':['person','chair'],'objects':[{'class_id':0,'xyxy':[.1,.1,.6,.8]}],
                     'source':{'author':'test fixture','license':'fixture only','url':f'recording:session{i}'}})
    manifest=tmp_path/'camera.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=import_recordings(manifest,tmp_path/'output')
    assert report['valid'] and report['images']==2
    assert (tmp_path/'output/labels/test/camera_000001.txt').read_text().split()[0]=='0'
    assert report['independent_group_proxy']=='recording session supplied by annotator'
