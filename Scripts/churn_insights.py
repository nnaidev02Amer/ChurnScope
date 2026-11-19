"""Non-ML helper logic for the churn app: input validation, risk bands, descriptive
risk observations and rule-based retention ideas.

Nothing here retrains or alters the model. The observations are DESCRIPTIVE (how often
similar customers churned in the training split); they are not model attributions (no SHAP)
and do not claim causes.
"""
import json
import math
from dataclasses import dataclass, field

import pandas as pd

import churn_pipeline as cp

# ------------------------------------------------------------------ risk bands
# Documented probability bands. "High" starts at the saved decision threshold, so
# High  <=> the model classifies the customer as "likely to churn".
LOW_MAX = 0.20  # below this: Low. From here up to the threshold: Medium.


def risk_band(probability, threshold):
    if probability >= threshold:
        return "High"
    if probability >= min(LOW_MAX, threshold):
        return "Medium"
    return "Low"


def band_definitions(threshold):
    lo = min(LOW_MAX, threshold)
    return {
        "Low": f"below {lo:.0%}",
        "Medium": f"{lo:.0%} to just under {threshold:.0%}",
        "High": f"{threshold:.0%} and above (the model's churn decision threshold)",
    }


# ------------------------------------------------------------------ validation
TENURE_RANGE = (1, 72)             # observed range in the data the model was trained on
MONTHLY_OBSERVED = (18.25, 118.75)  # observed range; outside it is extrapolation (warning)
ADDON_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies"]


def allowed_categories(bundle):
    """Category values the model was fitted on, read from its fitted encoder."""
    enc = bundle["pipeline"].named_steps["prep"].named_transformers_["cat"]
    return {col: [str(v) for v in cats] for col, cats in zip(cp.CATEGORICAL_FEATURES, enc.categories_)}


@dataclass
class Validation:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    customer: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.errors


def _number(value, name, errors):
    if isinstance(value, bool):
        errors.append(f"{name} must be a number.")
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number (got {value!r}).")
        return None
    if not math.isfinite(x):
        errors.append(f"{name} must be a finite number.")
        return None
    return x


def validate_customer(customer, allowed):
    """Check a raw customer dict. Returns Validation(errors, warnings, cleaned customer).

    Errors block prediction (the model would silently ignore an unseen category, which would
    give a misleading score, so unseen categories are rejected here). Warnings do not."""
    v = Validation()
    unknown = sorted(set(customer) - set(cp.FEATURES))
    missing = [f for f in cp.FEATURES if f not in customer]
    if unknown:
        v.errors.append("Unknown fields: " + ", ".join(unknown) + ".")
    if missing:
        v.errors.append("Missing fields: " + ", ".join(missing) + ".")
        return v

    clean = dict(customer)
    for col in cp.CATEGORICAL_FEATURES:
        if str(clean[col]) not in allowed[col]:
            v.errors.append(f"{col}: {clean[col]!r} is not a value the model knows "
                            f"(allowed: {', '.join(allowed[col])}).")
        else:
            clean[col] = str(clean[col])

    tenure = _number(clean["tenure"], "Tenure", v.errors)
    monthly = _number(clean["MonthlyCharges"], "Monthly charges", v.errors)
    total = _number(clean["TotalCharges"], "Total charges", v.errors)
    if tenure is not None:
        if tenure != int(tenure) or not (TENURE_RANGE[0] <= tenure <= TENURE_RANGE[1]):
            v.errors.append(f"Tenure must be a whole number of months from {TENURE_RANGE[0]} to "
                            f"{TENURE_RANGE[1]} (the range the model was trained on).")
        else:
            clean["tenure"] = int(tenure)
    if monthly is not None:
        if not (0 < monthly <= 500):
            v.errors.append("Monthly charges must be greater than 0 (and at most 500).")
        else:
            clean["MonthlyCharges"] = monthly
            if not (MONTHLY_OBSERVED[0] <= monthly <= MONTHLY_OBSERVED[1]):
                v.warnings.append(f"Monthly charges {monthly:.2f} are outside the range seen in the "
                                  f"training data ({MONTHLY_OBSERVED[0]}-{MONTHLY_OBSERVED[1]}); "
                                  "the estimate is less reliable.")
    if total is not None:
        if total < 0:
            v.errors.append("Total charges cannot be negative.")
        else:
            clean["TotalCharges"] = total
            if tenure and monthly and not v.errors:
                expected = tenure * monthly
                if expected > 0 and abs(total - expected) / expected > 0.5:
                    v.warnings.append(f"Total charges ({total:,.2f}) differ by more than 50% from "
                                      f"tenure x monthly charges ({expected:,.2f}); check the value.")

    # Service dependencies that always hold in the data.
    if clean["PhoneService"] == "No" and clean["MultipleLines"] != "No phone service":
        v.errors.append("Without phone service, Multiple lines must be 'No phone service'.")
    if clean["PhoneService"] == "Yes" and clean["MultipleLines"] == "No phone service":
        v.errors.append("With phone service, Multiple lines must be 'Yes' or 'No'.")
    if clean["InternetService"] == "No":
        bad = [c for c in ADDON_COLUMNS if clean[c] != "No internet service"]
        if bad:
            v.errors.append("Without internet service these must be 'No internet service': "
                            + ", ".join(bad) + ".")
    else:
        bad = [c for c in ADDON_COLUMNS if clean[c] == "No internet service"]
        if bad:
            v.errors.append("With internet service these must be 'Yes' or 'No': " + ", ".join(bad) + ".")

    v.customer = clean
    return v


# ---------------------------------------------------- descriptive reference stats
FACTOR_LABELS = {
    "Contract": "Contract", "InternetService": "Internet service", "PaymentMethod": "Payment method",
    "TechSupport": "Tech support", "OnlineSecurity": "Online security",
    "PaperlessBilling": "Paperless billing", "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner", "Dependents": "Has dependents",
    "tenure_band": "Tenure", "charge_band": "Monthly charges",
}
MIN_GROUP = 100  # ignore groups smaller than this
RISK_LIFT, PROTECT_LIFT = 1.25, 0.75


def tenure_band(months):
    if months <= 12:
        return "1-12 months"
    if months <= 24:
        return "13-24 months"
    if months <= 48:
        return "25-48 months"
    return "49-72 months"


def charge_band(monthly):
    if monthly < 35:
        return "under 35"
    if monthly < 70:
        return "35 to 70"
    if monthly < 90:
        return "70 to 90"
    return "90 and over"


def _with_bands(frame):
    out = frame.copy()
    out["tenure_band"] = out["tenure"].map(tenure_band)
    out["charge_band"] = out["MonthlyCharges"].map(charge_band)
    return out


def build_reference_stats(X_train, y_train):
    """Churn rate per factor value on the TRAINING split (descriptive only)."""
    frame = _with_bands(X_train)
    table = {}
    for factor in FACTOR_LABELS:
        grouped = y_train.groupby(frame[factor]).agg(["mean", "size"])
        table[factor] = {str(k): (float(r["mean"]), int(r["size"])) for k, r in grouped.iterrows()}
    return {"overall": float(y_train.mean()), "n": int(len(y_train)), "table": table}


def explain_risk(customer, stats, max_risk=4, max_protect=3):
    """Transparent observations: how often similar customers churned historically.

    Returns {"risk": [...], "protective": [...]}, each item a dict with the factor, the
    customer's value, the historical churn rate, group size and lift vs overall."""
    row = _with_bands(pd.DataFrame([customer])).iloc[0]
    overall = stats["overall"]
    risk, protect = [], []
    for factor, label in FACTOR_LABELS.items():
        value = str(row[factor])
        if value == "No internet service":  # redundant with the Internet service factor
            continue
        rate_n = stats["table"][factor].get(value)
        if not rate_n or rate_n[1] < MIN_GROUP:
            continue
        rate, n = rate_n
        lift = rate / overall
        item = {"factor": label, "value": value, "rate": rate, "n": n, "lift": lift}
        if lift >= RISK_LIFT:
            risk.append(item)
        elif lift <= PROTECT_LIFT:
            protect.append(item)
    risk.sort(key=lambda i: -i["lift"])
    protect.sort(key=lambda i: i["lift"])
    return {"risk": risk[:max_risk], "protective": protect[:max_protect], "overall": overall}


# --------------------------------------------------------- retention suggestions
def retention_suggestions(customer, band, max_items=5):
    """Rule-based ideas keyed on the customer's attributes and risk band.
    These are heuristics for a retention team to consider, NOT model outputs."""
    c = customer
    has_internet = c["InternetService"] != "No"
    ideas = []

    def add(title, detail):
        ideas.append({"title": title, "detail": detail})

    if c["Contract"] == "Month-to-month":
        add("Offer a term-contract incentive",
            "Month-to-month customers churn far more often than customers on one- or two-year "
            "terms. A modest discount or perk for moving to a longer term can add commitment.")
    if c["PaymentMethod"] == "Electronic check":
        add("Encourage automatic payment",
            "Electronic-check customers churned more often than those paying by bank transfer or "
            "card in this dataset. Offer a small incentive to switch to automatic payment.")
    if c["InternetService"] == "Fiber optic" and c["MonthlyCharges"] >= 70:
        add("Review fibre price and service quality",
            "Fibre customers on higher bills churned more often historically. Check for outages or "
            "speed complaints and consider a loyalty price or a plan right-size.")
    if has_internet and c["TechSupport"] == "No":
        add("Offer a tech-support trial",
            "Internet customers without tech support churned more often. A free trial or a "
            "proactive support call can improve the experience.")
    if has_internet and c["OnlineSecurity"] == "No":
        add("Offer an online-security trial",
            "Customers without online security churned more often; a bundled trial adds value "
            "and another reason to stay.")
    if c["tenure"] <= 12:
        add("Schedule an early-tenure check-in",
            "Newer customers (first year) churn most. A welcome call or onboarding follow-up "
            "can resolve early problems.")
    if c["SeniorCitizen"] == "Yes":
        add("Offer assisted, simplified support",
            "Senior customers churned more often in this data. Consider a dedicated support line "
            "and a plan review in plain language.")
    if c["MonthlyCharges"] >= 90 and c["InternetService"] != "Fiber optic":
        add("Check the plan fits their usage",
            "Very high monthly bills are associated with more churn; a right-sized bundle may "
            "keep the customer.")

    if band == "Low":
        head = ("Low risk: no intervention is needed. Keep the service quality steady; the ideas "
                "below are optional, light-touch actions.")
        ideas = ideas[:2]
        if not ideas:
            add("Maintain the relationship",
                "Nothing in this profile stands out. Routine communication and good service are enough.")
    elif band == "Medium":
        head = "Medium risk: worth a low-cost, proactive touch before the risk grows."
        ideas = ideas[:3] or [{"title": "Monitor", "detail": "No specific trigger found; re-check after the next bill."}]
    else:
        head = "High risk: prioritise this customer for proactive outreach."
        ideas = ideas[:max_items] or [{"title": "Personal outreach",
                                       "detail": "No single trigger stands out; a personal call to ask about satisfaction is a good start."}]
    return {"headline": head, "ideas": ideas}


# ------------------------------------------------------- verified model metrics
def load_metrics(path=None):
    path = path or (cp.PROJECT_DIR / "reports" / "metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def band_calibration(bundle, X_test, y_test):
    """Observed churn rate per risk band on the held-out test split, using the saved model
    and threshold (this reads data, it does not retrain anything)."""
    thr = bundle["threshold"]
    proba = bundle["pipeline"].predict_proba(X_test[cp.FEATURES])[:, 1]
    bands = pd.Series([risk_band(p, thr) for p in proba], index=X_test.index)
    rows = []
    for name in ("Low", "Medium", "High"):
        mask = bands == name
        n = int(mask.sum())
        rows.append({"Band": name, "Customers": n,
                     "Observed churn rate": float(y_test[mask].mean()) if n else float("nan")})
    return pd.DataFrame(rows)
