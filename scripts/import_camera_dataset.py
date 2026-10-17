"""Import independently annotated camera images; never generates ground truth.

Input JSONL: image (relative), split (dev/test), session_id, exhaustive_classes
['person','chair'], objects [{class_id:0 or 56, xyxy:[normalized x1,y1,x2,y2]}],
source {author, license, url}. Empty annotations require exhaustive-class review.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visiondrive.evaluation.data import COCO_NAMES, NAMES, digest, read_jsonl, validate, valid_box, write_json


def import_recordings(manifest,output):
    manifest,output=Path(manifest).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; existing data is never overwritten')
    rows=read_jsonl(manifest)
    prepared=[]
    sessions={}
    # Validate declarations and paths before writing output.
    for index,row in enumerate(rows):
        image=(manifest.parent/row['image']).resolve()
        if not image.is_relative_to(manifest.parent) or not image.is_file():
            raise ValueError('Source image missing or escapes the import directory')
        if row['split'] not in ('dev','test') or set(row.get('exhaustive_classes',[])) != set(NAMES.values()):
            raise ValueError('Explicit exhaustive person/chair annotation review is required')
        session=row['session_id']
        if not session or sessions.setdefault(session,row['split']) != row['split']:
            raise ValueError('Entire recording sessions must remain in one split')
        if any(o['class_id'] not in NAMES or not valid_box(o['xyxy']) for o in row['objects']):
            raise ValueError('Invalid original COCO ID or normalized box')
        if not all(row.get('source',{}).get(k) for k in ('author','license','url')):
            raise ValueError('Recording attribution and usage terms required')
        objects=[{**o,'source_class':'independent-camera-annotation','occluded':o.get('occluded',0),
                  'truncated':o.get('truncated',0),'label':NAMES[o['class_id']]} for o in row['objects']]
        pixels=cv2.imread(str(image))
        if pixels is None:raise ValueError('Unreadable source image')
        prepared.append({**row,'image_id':f'camera_{index:06d}','group_id':session,'group_type':'recording session','objects':objects,
            'source':{**row['source'],'dataset':'Independent group camera recordings','original_split':row['split'],
                      'download_url':row['source']['url']},
            'verified':{str(cid):int(any(o['class_id']==cid for o in objects)) for cid in NAMES},
            'sha256':digest(image),'width':pixels.shape[1],'height':pixels.shape[0],
            'mean_luminance':float(cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY).mean()),
            'image':f'images/{row["split"]}/camera_{index:06d}{image.suffix.lower()}',
            '_source':image})
    if set(sessions.values()) != {'dev','test'}:
        raise ValueError('Separate development and test sessions are required')
    for row in prepared:
        image=output/row['image'];image.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(row.pop('_source'),image)
        label=output/'labels'/row['split']/(row['image_id']+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        lines=[]
        for obj in row['objects']:
            x1,y1,x2,y2=obj['xyxy']
            lines.append(f'{obj["class_id"]} {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {x2-x1:.10f} {y2-y1:.10f}')
        label.write_text('\n'.join(lines),encoding='utf-8')
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prepared),encoding='utf-8')
    report=validate(output)
    report['independent_group_proxy']='recording session supplied by annotator'
    write_json(output/'dataset_validation.json',report)
    if not report['valid']:
        raise ValueError(report['errors'])
    return report


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('manifest',type=Path)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    print(import_recordings(args.manifest,args.output))
