"""Scoring for VSI-Bench / VSTI-Bench, copied verbatim from the benchmarks' own
lmms-eval task utils so G2VLM is scored identically to the baselines.

Sources:
  VSI : SpatialStack/src/lmms_eval/tasks/vsibench/utils.py
  VSTI: VLM-3R/thinking-in-space/lmms_eval/tasks/vstibench/utils.py

Differences kept intact:
  - question-type sets differ per benchmark
  - VSI MCA scores prediction vs `ground_truth` (a letter); VSTI MCA vs `mc_answer`
  - VSI aggregation merges easy/medium/hard rel-direction; VSTI aggregation is a
    flat mean over all per-type metrics.
No lmms-eval / datasets dependency, so this runs in g2vlm_env.
"""
from functools import partial

import numpy as np
import pandas as pd

MCA_QUESTION_TYPES = {
    "vsi": [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ],
    "vsti": [
        "obj_obj_relative_pos_nf",
        "obj_obj_relative_pos_ud",
        "obj_obj_relative_pos_lr",
        "camera_obj_rel_dist_v1",
        "camera_obj_rel_dist_v2",
        "camera_obj_rel_dist_v3",
        "camera_movement_direction",
    ],
}
NA_QUESTION_TYPES = {
    "vsi": [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ],
    "vsti": [
        "camera_obj_abs_dist",
        "camera_displacement",
        "camera_obj_dist_change",
    ],
}

METRICS_FOR_MCA = {"accuracy": "exact_match"}
METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)"
}
WORST_CASE_FOR_METRICS = {"accuracy": 0.0, "MRA:.5:.95:.05": 0.0}


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        return float(pred)
    except BaseException:
        return None


def process_results(doc, prediction, benchmark):
    """Mutates `doc` with metric columns, mirroring vsibench_process_results.

    For MCA, the comparison target is `mc_answer` if present (VSTI) else
    `ground_truth` (VSI) -- both are the option letter.
    """
    mca_types = MCA_QUESTION_TYPES[benchmark]
    na_types = NA_QUESTION_TYPES[benchmark]
    doc["prediction"] = prediction
    qt = doc["question_type"]
    if qt in mca_types:
        target = doc.get("mc_answer") or doc["ground_truth"]
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(prediction), target)
    elif qt in na_types:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(
                    to_float(fuzzy_matching(prediction)), to_float(doc["ground_truth"])
                )
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {qt}")
    return doc


def aggregate_results(docs, benchmark):
    """Returns (overall_score_x100, per_type_dict). Mirrors each benchmark's
    vsibench_aggregate_results."""
    mca_types = MCA_QUESTION_TYPES[benchmark]
    na_types = NA_QUESTION_TYPES[benchmark]
    results = pd.DataFrame(docs)
    output = {}
    for question_type, idx in results.groupby("question_type").groups.items():
        per = results.iloc[idx]
        if question_type in mca_types:
            for metric in METRICS_FOR_MCA:
                output[f"{question_type}_{metric}"] = per[metric].mean()
        elif question_type in na_types:
            for metric in METRICS_FOR_NA:
                output[f"{question_type}_{metric}"] = per[metric].mean()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    if benchmark == "vsi":
        direction_keys = [
            "object_rel_direction_easy_accuracy",
            "object_rel_direction_medium_accuracy",
            "object_rel_direction_hard_accuracy",
        ]
        direction_values = [output[k] for k in direction_keys if k in output]
        if direction_values:
            output["object_rel_direction_accuracy"] = sum(direction_values) / len(
                direction_values
            )
            for k in direction_keys:
                output.pop(k, None)

    if not output:
        return 0.0, {}
    overall = sum(output.values()) / len(output)
    return overall * 100.0, output
