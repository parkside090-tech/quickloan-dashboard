# app.py — QuickLoan Risk Dashboard
# Run:  streamlit run app.py   (works from any folder; files are found relative to this script)
# Files next to app.py:  loans.csv (required)  and  loan_model.joblib (optional: the champion saved from the notebook)
# If loan_model.joblib is missing or can't be used (missing file, library version mismatch, XGBoost not loading),
# the app retrains the notebook's champion (Stacking, reject if P(default) >= 0.14) from loans.csv once and caches it.
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="QuickLoan Risk Dashboard", page_icon="💰", layout="wide")

HERE = Path(__file__).resolve().parent
DATA_PATH, MODEL_PATH = HERE / "loans.csv", HERE / "loan_model.joblib"
TEST_SIZE, SEED = 0.25, 1          # the same split as the notebook
INTEREST = 0.20                    # interest earned on a repaid loan (from the brief)
CHAMPION_THRESHOLD = 0.14          # tuned by cross-validation in the notebook
HOME_OPTIONS = ["rent", "own", "mortgage"]


# ---------------------------------------------------------------- helpers
def encode(frame, columns):
    """The notebook's encoding: one-hot home_ownership, then align to the training columns."""
    return (pd.get_dummies(frame, columns=["home_ownership"])
              .reindex(columns=columns, fill_value=0).astype(float))


def money_saved(y_true, y_pred, loan, interest=INTEREST):
    """The money metric from the brief: saving compared with approving everyone."""
    y_true, y_pred, loan = np.asarray(y_true), np.asarray(y_pred), np.asarray(loan, dtype=float)
    caught = (y_pred == 1) & (y_true == 1)   # defaulter rejected      -> whole loan saved
    wrong = (y_pred == 1) & (y_true == 0)    # good borrower rejected  -> interest lost
    return float(loan[caught].sum() - interest * loan[wrong].sum())


def box():
    """A bordered card on newer Streamlit, a plain container on older versions."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def default_proba(model, X):
    return np.asarray(model.predict_proba(X))[:, 1]


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Training the champion model (one time only, about a minute)…")
def train_champion(path):
    """Rebuild the notebook's champion: Stacking (RF+GB+XGB+LR -> LR). The 0.14 threshold is applied by the app."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.model_selection import StratifiedKFold
    try:
        from xgboost import XGBClassifier
        booster = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9,
                                colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1)
        booster_name = "XGBoost"
    except Exception:  # not installed, or its library fails to load (e.g. libomp missing on macOS)
        from sklearn.ensemble import HistGradientBoostingClassifier
        booster = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED)
        booster_name = "HistGradientBoosting (XGBoost unavailable)"

    data = pd.read_csv(path)
    X = pd.get_dummies(data.drop(columns="default"), columns=["home_ownership"]).astype(float)
    y = data["default"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    stack = StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
                    ("gb", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)),
                    ("xgb", booster),
                    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        stack_method="predict_proba", n_jobs=-1)
    stack.fit(X_tr, y_tr)
    return stack, list(X_tr.columns), CHAMPION_THRESHOLD, f"Stacking (RF+GB+{booster_name}+LR)"


@st.cache_resource(show_spinner="Loading the saved model…")
def load_saved_model(path, data_path):
    """Load the notebook's model and prove it works on real rows; any failure raises so the app can fall back."""
    bundle = joblib.load(path)
    model, columns = bundle["model"], list(bundle["columns"])
    thr = getattr(model, "threshold", 0.5)
    thr = float(thr) if isinstance(thr, (int, float)) else 0.5
    inner = getattr(model, "estimator", model)
    name = type(inner).__name__
    if name == "Pipeline":
        name = type(inner.steps[-1][1]).__name__
    sample = pd.read_csv(data_path).drop(columns="default").head(5)
    default_proba(model, encode(sample, columns))   # raises if the model can't actually predict here
    return model, columns, thr, name


@st.cache_data(show_spinner="Scoring the test portfolio…")
def portfolio(_model, model_key, threshold, columns):
    """Score the 1,500 test loans once per model (not on every slider move)."""
    X = encode(df.drop(columns="default"), columns)
    y = df["default"]
    _, X_te, _, y_te, _, loan_te = train_test_split(X, y, df["loan_amount"], test_size=TEST_SIZE,
                                                    random_state=SEED, stratify=y)
    pred = (default_proba(_model, X_te) >= threshold).astype(int)
    return pred, y_te.to_numpy(), loan_te.to_numpy(dtype=float)


# ---------------------------------------------------------------- load
if not DATA_PATH.exists():
    st.error(f"`loans.csv` was not found next to app.py (looked in `{HERE}`). Add it to the same folder.")
    st.stop()
df = load_data(str(DATA_PATH))

@st.cache_resource(show_spinner=False)
def get_model(model_path, data_path):
    """Decide once per session: use the saved model if it works here, otherwise retrain."""
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError
        return (*load_saved_model(model_path, data_path), "notebook (loan_model.joblib)", None)
    except Exception as e:
        problem = None if isinstance(e, FileNotFoundError) else f"{type(e).__name__}: {e}"
        return (*train_champion(data_path), "retrained from loans.csv", problem)

model, COLUMNS, THRESHOLD, model_name, source, load_problem = get_model(str(MODEL_PATH), str(DATA_PATH))

# ---------------------------------------------------------------- header
st.title("💰 QuickLoan — Risk Dashboard")
st.caption(f"Model: **{model_name}** · reject when P(default) ≥ **{THRESHOLD:.2f}** · source: {source}")
if load_problem:
    with st.expander("ℹ️ Why the saved model wasn't used"):
        st.write("`loan_model.joblib` couldn't be used in this environment, so the champion was retrained "
                 "(same settings, same results). This is usually a library version mismatch. "
                 "Installing `requirements.txt` fixes it.")
        st.code(load_problem[:800])

# ---------------------------------------------------------------- CARD 1: decision
st.header("1 · Loan decision")
with box():
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 21, 64, 35)
        annual_income = st.number_input("Annual income (AED)", min_value=24000, max_value=400000, value=90000, step=1000)
        loan_amount = st.number_input("Loan amount (AED)", min_value=5000, max_value=250000, value=40000, step=1000)
        loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
        employment_years = st.slider("Employment years", 0.0, 35.0, 4.0, 0.5)
    with c2:
        credit_history_months = st.slider("Credit history (months)", 0, 300, 60)
        num_existing_loans = st.slider("Existing loans", 0, 8, 1)
        past_defaults = st.slider("Past defaults", 0, 6, 0)
        home_ownership = st.selectbox("Home ownership", HOME_OPTIONS)
        debt_to_income = min(round(loan_amount / (annual_income + 1), 2), 3.0)   # capped at 3.0 as in the data
        st.metric("Debt-to-income (calculated)", f"{debt_to_income:.2f}")

    applicant = pd.DataFrame([dict(
        age=age, annual_income=float(annual_income), loan_amount=float(loan_amount),
        loan_term_months=loan_term_months, employment_years=employment_years,
        credit_history_months=credit_history_months, num_existing_loans=num_existing_loans,
        past_defaults=past_defaults, debt_to_income=debt_to_income, home_ownership=home_ownership)])
    # Declare all categories so every home_ownership dummy exists for a single row
    applicant["home_ownership"] = pd.Categorical(applicant["home_ownership"], categories=HOME_OPTIONS)
    prob = float(default_proba(model, encode(applicant, COLUMNS))[0])
    reject = prob >= THRESHOLD

    st.divider()
    if reject:
        st.error(f"## ❌ REJECT\nHigh default risk: **{prob:.0%}** probability of default "
                 f"(the rejection threshold is {THRESHOLD:.0%}).")
    else:
        st.success(f"## ✅ APPROVE\nLow default risk: **{prob:.0%}** probability of default "
                   f"(the rejection threshold is {THRESHOLD:.0%}).")

    ev = INTEREST * (1 - prob) * loan_amount - prob * loan_amount
    d1, d2 = st.columns(2)
    d1.metric("Probability of default", f"{prob:.1%}", delta=f"{(prob - THRESHOLD) * 100:+.1f} pts vs threshold",
              delta_color="inverse")
    d2.metric("Expected value of approving", f"AED {ev:,.0f}",
              help="0.20 × P(repay) × loan − P(default) × loan. A negative value means approving is expected to lose money.")

# ---------------------------------------------------------------- CARD 2: portfolio
st.header("2 · Money saved (portfolio view)")
with box():
    pred_te, yt, loan_te = portfolio(model, f"{source}|{model_name}", THRESHOLD, COLUMNS)
    saved = money_saved(yt, pred_te, loan_te)
    caught = int(((pred_te == 1) & (yt == 1)).sum())
    total_def = int((yt == 1).sum())
    wrong = int(((pred_te == 1) & (yt == 0)).sum())
    total_good = int((yt == 0).sum())
    max_saving = float(loan_te[yt == 1].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("💵 Money saved vs approving everyone", f"AED {saved:,.0f}",
              help="Loans of defaulters caught − 20% interest on good borrowers wrongly rejected")
    m2.metric("🎯 Defaulters caught", f"{caught} / {total_def}", f"recall {caught / total_def:.0%}")
    m3.metric("😕 Good borrowers rejected", f"{wrong}", f"{wrong / total_good:.0%} of {total_good:,} good applicants",
              delta_color="off")

    st.markdown(f"**This model would save QuickLoan AED {saved:,.0f} compared with approving everyone.** "
                f"That is {saved / max_saving:.0%} of the AED {max_saving:,.0f} a perfect model could recover.")
    st.progress(float(min(max(saved / max_saving, 0.0), 1.0)))

    with st.expander("Decision breakdown on the test loans"):
        breakdown = pd.DataFrame(
            {"Model approves": [int(((pred_te == 0) & (yt == 0)).sum()), total_def - caught],
             "Model rejects": [wrong, caught]},
            index=["Actually repaid", "Actually defaulted"])
        st.table(breakdown)
        st.caption(f"Scored on {len(yt):,} held-out test loans (test_size={TEST_SIZE}, random_state={SEED}, "
                   "stratified), the same split as the notebook leaderboard.")
