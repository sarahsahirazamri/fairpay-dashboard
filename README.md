# FairPay Validator Dashboard (WQD7003, Group 12)

`hr_dashboard.py` is the production Streamlit web app for the FairPay salary project. It loads
the `.joblib` artefacts written by Section 6.1 of the project notebook and serves the tuned
Gradient Boosting salary model through three modules, with no coding required from the user.

## Modules

- **A. Salary Estimator (Objective 4).** Enter job title, experience, education, company size,
  location, industry, employment type, work arrangement, years of experience, benefits score,
  job description length, and key skills. Returns a predicted salary with a ± MAE confidence
  band, the market percentile it falls in, and the closest career band. Unseen job titles or
  locations are handled with a frequency fallback and an honest warning.
- **B. Geographic & Skill Premium Explorer (Objectives 2 and 3).** A per country salary box
  plot against the global median, a ranked skill premium chart (top 15 skills), and a work
  arrangement comparison. The skill table can be exported to CSV.
- **C. Career Progression Ladder (Objective 1).** The Entry to Executive median ladder plus an
  expected uplift calculator between any two levels.

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run hr_dashboard.py

cd C:\Users\User\Documents\fairpay_dashboard
pip install streamlit pandas numpy plotly joblib
python -m streamlit run hr_dashboard.py
```

The folder `deployment_artefacts/` must sit beside `hr_dashboard.py`. The app opens at
`http://localhost:8501`.

## Deploy free on Streamlit Community Cloud

1. Push this folder, including `deployment_artefacts/` and `requirements.txt`, to a GitHub repo.
2. Open https://share.streamlit.io, connect the repo, and set the main file to `hr_dashboard.py`.
3. Community Cloud installs the requirements and hosts the app at a public URL.

## About the artefacts in this bundle

The nine `.joblib` files included here are a faithful reproduction of your notebook pipeline,
fitted with your tuned hyperparameters. The reproduced metrics are R squared 0.9015 and MAE
about USD 16,545, which are within rounding of your notebook's tuned figures (R squared 0.9030,
MAE about USD 16,426). The small gap exists because your RandomizedSearchCV selected those
parameters as optimal on your exact feature matrix.

For the final submission, regenerate the artefacts from your own notebook so the app shows your
exact numbers: run the updated Section 6.1 cell (see `notebook_updates.md`), which writes all
nine files including `market_aggregates.joblib`. Copy the new `deployment_artefacts/` folder
beside `hr_dashboard.py` and redeploy. The app code does not change.

## Files

```
hr_dashboard.py
requirements.txt
README.md
deployment_artefacts/
├── tuned_gbr_model.joblib        # tuned Gradient Boosting model
├── standard_scaler.joblib        # fitted StandardScaler (Ridge reference only)
├── ordinal_encoders.joblib       # {column: OrdinalEncoder}
├── frequency_maps.joblib         # {column: {category: frequency}}
├── feature_names.joblib          # ordered 21 feature list
├── skill_feature_names.joblib    # top 5 skill flag columns
├── top5_skill_strings.joblib     # top 5 skill names
├── dashboard_data.joblib         # per row salary, location, skill flags (Module B box plots)
└── market_aggregates.joblib      # medians, ladder, country and skill tables, metrics (Modules B and C)
```
