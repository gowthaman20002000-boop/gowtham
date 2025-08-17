# train_model.py
# ------------------------------------------------------------
# Refit your model under scikit-learn 1.7.1 and save as model.joblib
# Feature ORDER matches your Streamlit app's 23 inputs.
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ========== CONFIG: EDIT THESE TO MATCH YOUR CSV SCHEMA ==========
DATA_CSV = "data/train.csv"   # <-- path to your training data

# Column names in your CSV (change if needed)
COLS = {
    "age": "age",
    "attendance": "attendance",
    "midterm": "midterm",
    "final": "final",
    "assignments": "assignments",
    "quizzes": "quizzes",
    "participation": "participation",
    "projects": "projects",
    "total_score": "total_score",
    "study_hours": "study_hours",
    "stress": "stress",
    "sleep": "sleep",
    "gender": "gender",               # values like: Male / Female
    "dept": "dept",                   # values: CS / Engineering / Mathematics / Business
    "extra": "extra",                 # values: Yes / No
    "internet": "internet",           # values: Yes / No
    "parent_edu": "parent_edu",       # values: None / High School / Bachelor's / Master's / PhD
    "income": "income",               # values: Low / Medium / High
    "target": "dropped_out",          # <-- your label column (0/1)
}

# Allowed category values (used to guard against typos & unseen categories)
CATS = {
    "gender": {"Male", "Female"},
    "dept": {"CS", "Engineering", "Mathematics", "Business"},
    "extra": {"Yes", "No"},
    "internet": {"Yes", "No"},
    "parent_edu": {"None", "High School", "Bachelor's", "Master's", "PhD"},
    "income": {"Low", "Medium", "High"},
}

# ========== FEATURE ORDER (must match app.py exactly) ==========
# 12 numeric + 11 one-hots = 23 total
FEATURE_ORDER = [
    COLS["age"], COLS["attendance"], COLS["midterm"], COLS["final"], COLS["assignments"],
    COLS["quizzes"], COLS["participation"], COLS["projects"], COLS["total_score"],
    COLS["study_hours"], COLS["stress"], COLS["sleep"],

    # One-hots / binary flags (match app.py encodings & order)
    "gender_male",
    "dept_cs", "dept_engineering", "dept_mathematics",
    "extra_yes", "internet_yes",
    "parent_hs", "parent_masters", "parent_phd",
    "income_low", "income_medium",
]
# Baselines (not included as features): dept=Business, parent_edu=Bachelor's, income=High, gender_female, extra_no, internet_no

# ========== LOAD DATA ==========
path = Path("data/Students Performance Dataset.csv")
if not path.exists():
    raise FileNotFoundError(f"Training data not found at {path.resolve()}")

df = pd.read_csv(path)

# Basic validations
missing = [c for c in COLS.values() if c not in df.columns]
if missing:
    raise ValueError(f"Your CSV is missing expected columns: {missing}")

# Optional: enforce types for numeric fields (coerce errors to NaN, then fill)
for num_col in [
    COLS["age"], COLS["attendance"], COLS["midterm"], COLS["final"], COLS["assignments"],
    COLS["quizzes"], COLS["participation"], COLS["projects"], COLS["total_score"],
    COLS["study_hours"], COLS["stress"], COLS["sleep"]
]:
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

# Handle NaNs simply (customize if you have a better imputation strategy)
df.fillna({
    COLS["age"]: df[COLS["age"]].median(),
    COLS["attendance"]: df[COLS["attendance"]].median(),
    COLS["midterm"]: df[COLS["midterm"]].median(),
    COLS["final"]: df[COLS["final"]].median(),
    COLS["assignments"]: df[COLS["assignments"]].median(),
    COLS["quizzes"]: df[COLS["quizzes"]].median(),
    COLS["participation"]: df[COLS["participation"]].median(),
    COLS["projects"]: df[COLS["projects"]].median(),
    COLS["total_score"]: df[COLS["total_score"]].median(),
    COLS["study_hours"]: df[COLS["study_hours"]].median(),
    COLS["stress"]: df[COLS["stress"]].median(),
    COLS["sleep"]: df[COLS["sleep"]].median(),
}, inplace=True)

# Optional: standardize/clean categorical text
def norm(s):
    if pd.isna(s):
        return s
    return str(s).strip()

for k in ["gender", "dept", "extra", "internet", "parent_edu", "income"]:
    df[COLS[k]] = df[COLS[k]].map(norm)

# Warn if unexpected categories appear
def check_categories(col_key):
    col = COLS[col_key]
    allowed = CATS[col_key]
    bad = set(df[col].dropna().unique()) - allowed
    if bad:
        print(f"[WARN] Unexpected values in '{col}': {bad}. They will fall back to baseline (0).")

for k in CATS:
    check_categories(k)

# ========== ENCODING (must mirror app.py exactly) ==========
def encode_row(r):
    # numerics
    row = [
        r[COLS["age"]], r[COLS["attendance"]], r[COLS["midterm"]], r[COLS["final"]],
        r[COLS["assignments"]], r[COLS["quizzes"]], r[COLS["participation"]],
        r[COLS["projects"]], r[COLS["total_score"]], r[COLS["study_hours"]],
        r[COLS["stress"]], r[COLS["sleep"]],
    ]

    # binaries / one-hots (baselines commented)
    gender_male = 1 if r[COLS["gender"]] == "Male" else 0           # Female baseline
    dept_cs = 1 if r[COLS["dept"]] == "CS" else 0                   # Business baseline
    dept_engineering = 1 if r[COLS["dept"]] == "Engineering" else 0
    dept_mathematics = 1 if r[COLS["dept"]] == "Mathematics" else 0
    extra_yes = 1 if r[COLS["extra"]] == "Yes" else 0               # No baseline
    internet_yes = 1 if r[COLS["internet"]] == "Yes" else 0         # No baseline
    parent_hs = 1 if r[COLS["parent_edu"]] == "High School" else 0  # Bachelor's baseline
    parent_masters = 1 if r[COLS["parent_edu"]] == "Master's" else 0
    parent_phd = 1 if r[COLS["parent_edu"]] == "PhD" else 0
    income_low = 1 if r[COLS["income"]] == "Low" else 0             # High baseline
    income_medium = 1 if r[COLS["income"]] == "Medium" else 0

    row.extend([
        gender_male,
        dept_cs, dept_engineering, dept_mathematics,
        extra_yes, internet_yes,
        parent_hs, parent_masters, parent_phd,
        income_low, income_medium
    ])
    return row

# Build X with the exact feature order
X_list = df.apply(encode_row, axis=1).tolist()
X = pd.DataFrame(X_list, columns=FEATURE_ORDER)

y = df[COLS["target"]].astype(int)

# ========== TRAIN ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Quick sanity check
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, pred, zero_division=0))

# ========== SAVE (joblib) ==========
out = {
    "model": model,
    "feature_order": FEATURE_ORDER,
    "encodings": {
        "dept_baseline": "Business",
        "parent_edu_baseline": "Bachelor's",
        "income_baseline": "High",
        "gender_baseline": "Female",
        "extra_baseline": "No",
        "internet_baseline": "No",
    },
}
dump(out, "model.joblib")
print("Saved model.joblib")
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

