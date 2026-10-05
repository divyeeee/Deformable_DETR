from src.evaluation.verify_coco_map import get_test_cases, run_pycocotools_eval
from src.evaluation.coco_map import COCOEvaluator


def test_coco_map_vs_pycocotools():
    """
    Verify that custom COCOEvaluator metrics match pycocotools COCOeval stats
    within an absolute tolerance of 1e-3 across all 3 synthetic test cases.
    """
    cases = get_test_cases()
    tol = 1e-3

    for case_name, preds, targets in cases:
        custom_eval = COCOEvaluator()
        custom_eval.update(preds, targets)
        custom_metrics = custom_eval.evaluate()

        pycoco_metrics = run_pycocotools_eval(preds, targets)

        for metric in ["AP", "AP50", "AP75"]:
            val_c = custom_metrics[metric]
            val_p = pycoco_metrics[metric]
            diff = abs(val_c - val_p)

            assert diff <= tol, (
                f"Verification failed on {case_name} for metric '{metric}': "
                f"Custom={val_c:.4f}, pycocotools={val_p:.4f}, diff={diff:.6f} > {tol}"
            )


if __name__ == "__main__":
    test_coco_map_vs_pycocotools()
    print("\nPyCOCOTools verification test PASSED successfully!")
