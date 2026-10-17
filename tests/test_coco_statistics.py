"""Check image/instance counting without downloading COCO."""
import pytest

from scripts.summarize_coco import summarize


def test_distinct_images_instances_and_crowds():
    result = summarize(dict(
        categories=[dict(id=1, name='person'), dict(id=3, name='car')],
        images=[dict(id=10), dict(id=11), dict(id=12)],
        annotations=[dict(category_id=1, image_id=10),
                     dict(category_id=1, image_id=10),
                     dict(category_id=1, image_id=11, iscrowd=1),
                     dict(category_id=3, image_id=10)]))
    assert result['total_images'] == 3  # Includes images with no annotations.
    person, car = result['classes']
    assert person['images'] == 2
    assert person['annotation_records'] == 3
    assert person['noncrowd_images'] == 1
    assert person['noncrowd_instances'] == 2
    assert person['crowd_records'] == 1
    assert car['images'] == 1  # Same image may count toward several classes.


def test_unknown_image_reference_is_rejected():
    with pytest.raises(ValueError, match='unknown'):
        summarize(dict(categories=[dict(id=1, name='person')], images=[],
                       annotations=[dict(category_id=1, image_id=99)]))
