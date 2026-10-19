# 💰 QuickLoan Risk Dashboard

A Streamlit dashboard for the QuickLoan ensemble-learning challenge. It uses the champion model from the notebook: a **Stacking ensemble** (RF + GB + XGBoost + LR, with an LR meta-model) that **rejects when P(default) ≥ 0.14**.

## What it shows
1. **Loan decision.** Enter an applicant's details to get **APPROVE / REJECT**, the probability of default, and the expected value of approving.
2. **Portfolio view.** On the held-out test split (1,500 loans), it shows **money saved compared with approving everyone**, **defaulters caught**, and **good borrowers wrongly rejected**.

Expected portfolio results: **AED 6,113,140 saved · 102 / 188 defaulters caught · 131 good borrowers rejected**.

## Files
| File | Purpose |
|---|---|
| `QuickLoan_BakeOff.ipynb` | The notebook: EDA → single models → Voting, Bagging, Boosting, Stacking → recall tuning → leaderboard → verdict |
| `app.py` | The dashboard |
| `loans.csv` | Data (required for the portfolio card) |
| `loan_model.joblib` | The champion saved from the notebook. Download it from Colab's Files panel |
| `requirements.txt` | Library versions pinned to match the Colab environment that trained the model |

`loan_model.joblib` is optional. If it's missing or won't load, the app retrains the same champion from `loans.csv` the first time it starts (about 30–60 s, then cached). The results are identical.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
Push this folder to a GitHub repo, then create a new app at share.streamlit.io with the main file `app.py`.
