"""Freeze a fresh person-only indoor cohort BEFORE inference.

Uses official source annotations cached/downloaded by the acquisition workflow.
Prior inspected cohort and ESP baseline IDs are excluded. No model is imported.
The distributed manifest, not this selection script, is the normal restore input.
"""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visiondrive.evaluation.data import (ROOT,CACHE,PEOPLE,LICENSE,SOURCE,acquire,ambiguous_hierarchy,
                                       descendants,digest,fetch,map_objects,read_jsonl,write_json)


def build():
    old_manifest=ROOT/'data/public/indoor-evaluation-manifest.jsonl'
    archive=ROOT/'data/public/exploratory-indoor-manifest.jsonl'
    if not archive.exists():archive.write_bytes(old_manifest.read_bytes())
    excluded={r['image_id'] for p in [archive,ROOT/'data/public/bottle-cup-box-manifest.jsonl'] for r in read_jsonl(p)}
    hierarchy=CACHE/'hierarchy.json'
    fetch('https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json',hierarchy)
    building_ids=descendants(json.loads(hierarchy.read_text()),'/m/0cgh4')
    pool={};audit={};source_files=[]
    for split,partition,folder in [('test','test','openimages-test-supplement'),('validation','dev','openimages-objects')]:
        boxes_path=ROOT/f'data/raw/{folder}/boxes.csv'
        metadata_path=ROOT/f'data/raw/{folder}/metadata.csv'
        labels_path=CACHE/f'{split}-labels.csv'
        for p,url in [(boxes_path,f'https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv'),
                      (metadata_path,f'https://storage.googleapis.com/openimages/2018_04/{split}/{split}-images-with-rotation.csv'),
                      (labels_path,f'https://storage.googleapis.com/openimages/v5/{split}-annotations-human-imagelabels-boxable.csv')]:
            sha=fetch(url,p)
            source_files.append({'url':url,'sha256':sha,'bytes':p.stat().st_size})
        inside=set();groups=defaultdict(list);bad=set()
        with boxes_path.open(newline='',encoding='utf-8-sig') as src:
            for b in csv.DictReader(src):
                if b['LabelName'] in building_ids and b['IsInside']=='1':inside.add(b['ImageID'])
                if b['LabelName'] in PEOPLE:
                    groups[b['ImageID']].append(b)
                    if any(b[k]!='0' for k in ('IsGroupOf','IsDepiction','IsInside')):bad.add(b['ImageID'])
        verified={}
        with labels_path.open(newline='',encoding='utf-8-sig') as src:
            for x in csv.DictReader(src):
                if x['LabelName']=='/m/01g317' and x['ImageID'] in inside and x['Confidence'] in ('0','1'):
                    verified[x['ImageID']]=int(x['Confidence'])
        metadata={}
        with metadata_path.open(newline='',encoding='utf-8-sig') as src:
            for m in csv.DictReader(src):
                if m['ImageID'] in verified:metadata[m['ImageID']]=m
        rows=[];counts=Counter()
        for image_id in sorted(inside):
            if image_id in excluded:counts['prior_inspected_or_baseline_image']+=1;continue
            if image_id not in verified:counts['unverified_person_presence_absence']+=1;continue
            if image_id in bad:counts['target_attribute_exclusion']+=1;continue
            objects,collapsed=map_objects(groups[image_id])
            if ambiguous_hierarchy(objects):counts['ambiguous_person_hierarchy']+=1;continue
            if bool(objects)!=bool(verified[image_id]):counts['label_box_conflict']+=1;continue
            m=metadata.get(image_id,{})
            if m.get('License')!=LICENSE or not m.get('OriginalLandingURL') or not m.get('AuthorProfileURL'):
                counts['attribution_license_exclusion']+=1;continue
            rows.append({'image_id':image_id,'split':partition,'image':f'images/{partition}/{image_id}.jpg',
                'group_id':m['AuthorProfileURL'],'objects':objects,'verified':{'0':verified[image_id]},'required_class_ids':[0],
                'collapsed_hierarchy_boxes':collapsed,'source':{'dataset':'Open Images V5 boxes / V7 protocol',
                'original_split':split,'url':m['OriginalLandingURL'],'original_image_url':m['OriginalURL'],
                'download_url':f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg',
                'author':m['Author'],'author_url':m['AuthorProfileURL'],'title':m['Title'],'license':m['License']},
                'indoor_evidence':'official Building-family box with IsInside=1','annotation_protocol':SOURCE})
        pool[partition]=rows;audit[partition]={'source_exclusions':dict(counts),'eligible':len(rows)}
    rng=random.Random(20261017)
    # Whole photographer groups are kept in only one split. One photo per
    # photographer reduces within-cohort dependence; no test score is consulted.
    selected=[];authors=set()
    for partition,total,positive_count in [('test',320,240),('dev',80,60)]:
        candidates=sorted(pool[partition],key=lambda r:r['image_id']);rng.shuffle(candidates)
        for state,target in [(1,positive_count),(0,total-positive_count)]:
            accepted=0
            for row in candidates:
                if row['verified']['0']!=state or row['group_id'] in authors:continue
                selected.append(row);authors.add(row['group_id']);accepted+=1
                if accepted==target:break
            if accepted!=target:raise ValueError(f'Insufficient independent {partition} state={state}: {accepted}/{target}')
    assert len(selected)==400
    protocol={'protocol':'indoor-pretrained-v3-fresh-person','selection_seed':20261017,'target_images':400,
        'partitions':{'dev':80,'test':320},'positive_image_fraction':.75,'one_image_per_photographer':True,
        'prior_excluded_manifest_sha256':digest(archive),'source_files':source_files,'selection_audit':audit,
        'test_policy':'New images not in prior inspected cohort or baseline; source-only selection, fixed person-only protocol before model predictions.',
        'prior_protocol_issue':'Source Stool/Chair is not exhaustive for COCO chair; chair excluded from primary metrics.'}
    write_json(ROOT/'data/public/indoor-selection-audit.json',protocol)
    # Stage the source-only manifest before any image/model inspection. acquire
    # restores it, records pixel hashes and rejects exact/near duplicate pairs.
    temp=CACHE/'fresh-source-selection.jsonl'
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    output=ROOT/'datasets/indoor-person-evaluation'
    records=acquire(output,temp)
    if len(records)!=400:raise ValueError('Pixel deduplication changed selection; review source-only replacements before inference')
    old_manifest.write_bytes((output/'manifest.jsonl').read_bytes())
    config_path=ROOT/'config/evaluation.yaml';config=json.loads(config_path.read_text())
    config.update(protocol=protocol['protocol'],dataset=output.relative_to(ROOT).as_posix(),dataset_manifest_sha256=digest(old_manifest),
                  selection='Fresh person-only verified indoor cohort; 75% person-positive, one image per photographer; official validation=dev and test=test; prior inspected images excluded')
    write_json(config_path,config)
    print('Frozen fresh 400-image cohort; ready for first model inference.',flush=True)


if __name__=='__main__':build()
