"""Model tests: validation-only selection, untouched test set, different feature sets, SHAP faithful to the model."""
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import explainability as xai
from src.churn_model import evaluate_on_test, run_modeling
from src.feature_engineering import FEATURES, select_features


@pytest.fixture(scope="module")
def modeled(tiny):
    feats, _ = select_features(tiny["train"], tiny["av"])
    return feats, run_modeling(tiny["train"], tiny["val"], feats, metric="pr_auc")


def test_selection_uses_validation_metric_only(tiny, modeled):
    feats, res = modeled
    assert "test" not in " ".join(inspect.signature(run_modeling).parameters)            # the test set cannot even be passed in
    vt = res["val_table"].set_index("model")
    assert res["best_name"] == vt.val_pr_auc.idxmax()
    for n, (m, cols) in res["models"].items():                                           # reported val metric = recomputed on validation
        assert vt.loc[n, "val_pr_auc"] == pytest.approx(average_precision_score(tiny["val"].churned, m.predict_proba(tiny["val"][cols])[:, 1]))
    r2 = run_modeling(tiny["train"], tiny["val"], feats, metric="roc_auc")
    assert r2["best_name"] == r2["val_table"].set_index("model").val_roc_auc.idxmax()


def test_test_set_cannot_change_selection_or_models(tiny, modeled):
    feats, res = modeled
    snapshot = (res["best_name"], dict(res["thresholds"]), res["models"]["Gradient boosting"][0].predict_proba(tiny["val"][feats])[:, 1].copy())
    shuffled = tiny["test"].copy(); shuffled["churned"] = np.random.default_rng(0).permutation(shuffled.churned.values)
    comp_a, _ = evaluate_on_test(res, tiny["test"])
    comp_b, _ = evaluate_on_test(res, shuffled)                                           # wildly different test labels...
    assert res["best_name"] == snapshot[0] and res["thresholds"] == snapshot[1]           # ...same selection and thresholds
    np.testing.assert_array_equal(res["models"]["Gradient boosting"][0].predict_proba(tiny["val"][feats])[:, 1], snapshot[2])
    assert comp_a.pr_auc.iloc[:3].tolist() != comp_b.pr_auc.iloc[:3].tolist()           # while test metrics DO react to test labels


def test_preprocessing_is_fitted_on_training_data_only(tiny, modeled):
    feats, res = modeled
    pipe, cols = res["models"]["Logistic (all features)"]
    Ztr, Zte = pipe[:2].transform(tiny["train"][cols]), pipe[:2].transform(tiny["test"][cols])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, Ztr.mean(0), atol=1e-8)   # scaler = train statistics...
    assert not np.allclose(pipe.named_steps["scale"].mean_, Zte.mean(0), atol=1e-8)       # ...not test statistics


def test_models_work_with_reduced_feature_sets(tiny):
    for drop in (["n_unique_categories", "avg_discount", "discount_order_share"], ["n_unique_products", "return_rate", "days_since_registration"]):
        feats = [f for f in FEATURES if f not in drop]
        res = run_modeling(tiny["train"], tiny["val"], feats)
        comp, probs = evaluate_on_test(res, tiny["test"])
        assert set(res["models"]["Gradient boosting"][1]) == set(feats) and np.isfinite(comp.roc_auc.iloc[:3]).all()


def test_feature_selection_excludes_unavailable_inputs(tiny):
    av = dict(tiny["av"], category=False, discount=False, product_id=False, registration_date=False)
    train = tiny["train"].copy(); train["n_unique_categories"] = np.nan
    used, excl = select_features(train, av)
    for f in ("n_unique_categories", "avg_discount", "discount_order_share", "n_unique_products", "days_since_registration"):
        assert f in excl and f not in used
    assert "recency_days" in used


def test_shap_reproduces_the_selected_models_output(tiny, modeled):
    feats, res = modeled
    X = tiny["test"][feats].head(60)
    for name in ("Gradient boosting", "Logistic (all features)"):                         # tree path and model-agnostic path
        model = res["models"][name][0]
        sv, base = xai.shap_values(model, X)
        p = model.predict_proba(X)[:, 1]
        np.testing.assert_allclose(base + sv.sum(1), np.log(p / (1 - p)), atol=1e-4)      # additive & faithful to THIS model
        assert sv.shape == X.shape
    up, down = xai.explain_row(sv[0], X.iloc[0], 3)                                        # text explanation is produced from the same values
    assert len(up) + len(down) > 0 and all(isinstance(x, str) and x for x in up + down)
