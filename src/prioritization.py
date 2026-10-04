"""Business layer: turn churn probability into value-aware retention priorities."""
import numpy as np
import pandas as pd

from .config import GROSS_MARGIN

# (risk, value tier) -> (priority, action). Actions are SUGGESTIONS for a human team, not automatic decisions.
PLAYBOOK = {
    ("High", "High"): ("P1 - Critical", "Personal outreach: customer-service call or account-owner email, loyalty gesture; discount only if needed"),
    ("High", "Medium"): ("P2 - High", "Personalised email with product recommendations based on favourite category"),
    ("High", "Low"): ("P3 - Low-cost", "Automated reminder email / 'we miss you' message (no discount)"),
    ("Medium", "High"): ("P2 - High", "Loyalty communication (early access, thank-you) plus a gentle reminder"),
    ("Medium", "Medium"): ("P3 - Low-cost", "Automated reminder or product-recommendation email"),
    ("Medium", "Low"): ("P4 - Monitor", "Keep in regular newsletter; re-score next month"),
    ("Low", "High"): ("P5 - Maintain", "Loyalty / VIP communication; do NOT discount (customer is likely to stay anyway)"),
    ("Low", "Medium"): ("P5 - Maintain", "Business-as-usual communication"),
    ("Low", "Low"): ("P5 - Maintain", "Business-as-usual communication"),
}
WINBACK = {"High": ("W1 - Win-back (high value)", "Personal win-back message; an offer is justified because the customer has already lapsed"),
           "Medium": ("W2 - Win-back", "Reactivation email with recommendations"),
           "Low": ("W3 - Low priority", "Include in low-cost reactivation batch or suppress")}


def value_tier(s: pd.Series) -> pd.Series:
    pct = s.rank(pct=True, method="average")
    return pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0], labels=["Low", "Medium", "High"], include_lowest=True).astype(str)


def add_value_and_priority(df: pd.DataFrame, H: int, margin=GROSS_MARGIN) -> pd.DataFrame:
    """
    df: one row per customer with monetary, days_since_first_order, status ('Active'/'Lapsed'),
        churn_prob (NaN for lapsed) and risk_level.

    CLV proxy ASSUMPTIONS (a heuristic, not a rigorous probabilistic CLV model such as BG/NBD):
      * annual revenue run-rate = historical spend / observed life (>= 90 days) * 365
      * margin is a flat share of revenue (GROSS_MARGIN)
      * the churn probability for one window of H days is assumed constant in each future window
      * 12-month value = annual margin * average survival over the ceil(365/H) windows
    """
    out = df.copy()
    life = out.days_since_first_order.clip(lower=90)
    out["annual_revenue_run_rate"] = out.monetary / life * 365
    out["annual_margin_run_rate"] = out.annual_revenue_run_rate * margin
    n_w = int(np.ceil(365 / H))
    p = out.churn_prob.fillna(1.0)
    surv = sum((1 - p) ** k for k in range(1, n_w + 1)) / n_w
    out["clv_12m_proxy"] = out.annual_margin_run_rate * surv
    out["margin_at_risk"] = out.annual_margin_run_rate * out.churn_prob  # NaN for lapsed
    out["value_tier"] = value_tier(out.annual_margin_run_rate)
    act = out.status == "Active"
    pr = [PLAYBOOK[(r, v)] if a else WINBACK[v] for r, v, a in zip(out.risk_level.astype(str), out.value_tier, act)]
    out["priority"] = [x[0] for x in pr]
    out["suggested_action"] = [x[1] for x in pr]
    # expected annual margin at stake (active customers only; lapsed customers are ranked by annual_margin_run_rate)
    out["priority_score"] = out.margin_at_risk
    return out
