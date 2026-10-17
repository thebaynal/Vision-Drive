import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from visiondrive.evaluation.data import ambiguous_hierarchy, COCO_NAMES, dhash, digest, map_objects, validate
from visiondrive.evaluation.metrics import aggregate, bootstrap, match


def source(mid='/m/01g317', box=(.1,.1,.6,.8)):
    return dict(zip(('XMin','YMin','XMax','YMax'),map(str,box)),LabelName=mid,IsOccluded='0',IsTruncated='0')


def test_mapping_preserves_person_and_chair_original_ids_and_merges_only_coincident_hierarchy():
    values,n = map_objects([source(),source('/m/03bt1vf'),source('/m/01mzpv',(.7,.1,.9,.5))])
    assert [o['class_id'] for o in values] == [0,56] and n == 1
    assert COCO_NAMES[0] == 'person' and COCO_NAMES[56] == 'chair' and len(COCO_NAMES)==80
    values,n = map_objects([source(),source('/m/03bt1vf',(.11,.11,.62,.82))])
    assert ambiguous_hierarchy(values) and n == 0


def row():
    obj={'class_id':0,'xyxy':[.1,.1,.6,.8]}
    return {'group_id':'session1','objects':[obj],
            'predictions':[{**obj,'confidence':.9},{**obj,'confidence':.8}]}


def test_one_to_one_matching_duplicate_prediction_is_false_positive():
    result=match(row(),0)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,0)
    r=row();r['predictions']=[{**r['predictions'][0],'class_id':56}]
    result=aggregate([r])
    assert result['micro']['tp']==0 and result['micro']['fp']==1 and result['micro']['fn']==1


def test_macro_f1_is_mean_class_f1_and_bootstrap_groups_are_reproducible():
    r=row();result=aggregate([r])
    assert result['macro']['f1']==pytest.approx((2/3)/2)
    assert bootstrap([r],10,42)==bootstrap([r],10,42)
    assert bootstrap([r],10,42)['groups']==1


def test_primary_scope_excludes_auxiliary_chair_ground_truth():
    r=row();r['objects'].append({'class_id':56,'xyxy':[.2,.2,.8,.9]})
    result=aggregate([r],class_ids=[0])
    assert set(result['per_class'])=={'person'}
    assert result['micro']['fn']==0 and result['micro']['tp']==1


def fixture(tmp_path):
    rows=[]
    rng=np.random.default_rng(42)
    for i,split in enumerate(('dev','test')):
        img=rng.integers(0,256,(64,64,3),dtype=np.uint8)
        p=tmp_path/f'images/{split}/{i}.png';p.parent.mkdir(parents=True);cv2.imwrite(str(p),img)
        label=tmp_path/f'labels/{split}/{i}.txt';label.parent.mkdir(parents=True);label.write_text('0 0.35 0.45 0.5 0.7\n')
        rows.append({'image_id':str(i),'split':split,'image':p.relative_to(tmp_path).as_posix(),
                     'sha256':digest(p),'group_id':f'scene{i}','verified':{'0':1,'56':0},
                     'objects':[{'class_id':0,'xyxy':[.1,.1,.6,.8],'source_class':'/m/01g317'}],
                     'source':{'url':'https://example.org/fixture','license':'test only','author':'fixture'}})
    def save(): (tmp_path/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    save()
    return rows,save


def test_validator_detects_mutated_labels_and_missing_completeness(tmp_path):
    rows,save=fixture(tmp_path)
    assert validate(tmp_path)['valid']
    (tmp_path/'labels/test/1.txt').write_text('1 0.5 0.5 1 1')
    assert any('Conflicting' in x for x in validate(tmp_path)['errors'])
    rows[0]['verified']={'0':1};save()
    assert any('verification' in x for x in validate(tmp_path)['errors'])


def test_validator_rejects_split_leakage_and_near_duplicate_pixels(tmp_path):
    rows,save=fixture(tmp_path)
    rows[1]['group_id']=rows[0]['group_id'];save()
    assert any('crosses' in x for x in validate(tmp_path)['errors'])
    image=cv2.imread(str(tmp_path/rows[0]['image']))
    cv2.imwrite(str(tmp_path/rows[1]['image']),image)
    rows[1]['sha256']=digest(tmp_path/rows[1]['image']);save()
    assert any('Duplicate' in x for x in validate(tmp_path)['errors'])
    assert dhash(image)==dhash(image.copy())
