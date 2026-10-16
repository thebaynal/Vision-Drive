from scripts.evaluate_objects import score


def row(predictions, objects):
    return {"predictions": predictions, "objects": objects}


def prediction(class_id=0, confidence=.9):
    return dict(x1=.1, y1=.1, x2=.6, y2=.8, confidence=confidence, class_id=class_id)


def test_duplicate_predictions_do_not_match_one_target_twice():
    target = {"label": "bottle", "xyxy": [.1, .1, .6, .8]}
    report = score([row([prediction(), prediction()], [target])], .35)
    assert report["per_class"]["bottle"]["tp"] == 1
    assert report["per_class"]["bottle"]["fp"] == 1


def test_class_mismatch_is_false_positive_and_false_negative():
    target = {"label": "cup", "xyxy": [.1, .1, .6, .8]}
    report = score([row([prediction(class_id=0)], [target])], .35)
    assert report["per_class"]["bottle"]["fp"] == 1
    assert report["per_class"]["cup"]["fn"] == 1


def test_below_threshold_predictions_do_not_count_as_detections():
    target = {"label": "bottle", "xyxy": [.1, .1, .6, .8]}
    report = score([row([prediction(confidence=.2)], [target])], .35)
    assert report["per_class"]["bottle"]["tp"] == 0
    assert report["per_class"]["bottle"]["fn"] == 1
