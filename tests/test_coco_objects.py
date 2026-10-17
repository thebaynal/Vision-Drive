"""Official COCO fixed-threshold match accounting, without neural inference."""
from scripts.evaluate_coco_objects import fixed_counts


def test_crowd_matches_do_not_inflate_true_positives():
    row = dict(category_id=44, maxDet=100, aRng=[0, 1e10],
               dtMatches=[[1, 2, 0]], dtIgnore=[[False, True, False]],
               gtMatches=[[1, 2, 0]], gtIgnore=[False, True, False])
    result = fixed_counts([row, None], 44)
    assert (result['tp'], result['fp'], result['fn']) == (1, 1, 1)
    assert result['precision'] == result['recall'] == result['f1'] == .5


def test_other_categories_and_repeated_area_bins_are_excluded():
    rows = [dict(category_id=47, maxDet=100, aRng=[0, 1e10]),
            dict(category_id=44, maxDet=100, aRng=[0, 1024])]
    assert fixed_counts(rows, 44)['tp'] == 0
