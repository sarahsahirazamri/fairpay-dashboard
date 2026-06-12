# Notebook updates for Group 12 (paste these into your own notebook)

These three snippets wire your notebook to `hr_dashboard.py`. They use your existing variable
names (`tuned_model`, `df_ml`, `X_test`, `y_test`, `y_train`, `train_idx`, `FEATURES`,
`SKILL_FEATS`, `top5_skills`, `freq_maps`, `ordinal_encoders`, `ordinal_specs`, `scaler`).

--------------------------------------------------------------------------------
## 1. Replace your Section 6.1 code cell with this

It saves the original eight artefacts **plus** `market_aggregates.joblib`, which makes
Modules B and C of the dashboard self contained and carries your exact tuned metrics.
(Also change the words "Eight `.joblib` files" to "Nine `.joblib` files" in the markdown
cell just above.)

```python
import os, joblib, numpy as np
from sklearn.metrics import (r2_score, mean_absolute_error,
                             mean_squared_error, mean_absolute_percentage_error)

deploy_dir = "deployment_artefacts"
os.makedirs(deploy_dir, exist_ok=True)

# ---- metrics of the tuned model, on the original USD scale ----
pred_log = tuned_model.predict(X_test)
t, p = np.expm1(y_test), np.expm1(pred_log)
metrics = {
    "r2"  : round(float(r2_score(y_test, pred_log)), 4),
    "mae" : int(round(mean_absolute_error(t, p))),
    "rmse": int(round(np.sqrt(mean_squared_error(t, p)))),
    "mape": round(float(mean_absolute_percentage_error(t, p)) * 100, 2),
    "naive_mae": int(round(mean_absolute_error(t, np.full_like(t, np.median(np.expm1(y_train)))))),
}

# ---- market aggregates for Modules B and C ----
EXP    = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
REMOTE = {0: "On-site", 50: "Hybrid", 100: "Fully remote"}
df_ml["exp_label"]    = df_ml["experience_level"].map(EXP)
df_ml["remote_label"] = df_ml["remote_ratio"].map(REMOTE)

career_ladder = {k: int(round(df_ml.loc[df_ml.exp_label == k, "salary_usd"].median()))
                 for k in ["Entry", "Mid", "Senior", "Executive"]}

top_countries = df_ml["company_location"].value_counts().head(15).index.tolist()
country_median = {c: int(round(df_ml.loc[df_ml.company_location == c, "salary_usd"].median()))
                  for c in top_countries}
country_median = dict(sorted(country_median.items(), key=lambda kv: -kv[1]))

from collections import Counter
skill_counter = Counter()
for lst in df_ml["skills_list"]:
    skill_counter.update(lst)
top15 = [s for s, _ in skill_counter.most_common(15)]
skill_premium = {s: int(round(df_ml[df_ml["skills_list"].apply(lambda l: s in l)]["salary_usd"].median()))
                 for s in top15}
skill_premium = dict(sorted(skill_premium.items(), key=lambda kv: -kv[1]))

remote_median = {v: int(round(df_ml.loc[df_ml.remote_label == v, "salary_usd"].median()))
                 for v in ["On-site", "Hybrid", "Fully remote"]}
quant = {str(q): int(round(df_ml["salary_usd"].quantile(q))) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
num_medians = {c: float(df_ml.loc[train_idx, c].median())
               for c in ["remote_ratio", "years_experience", "job_description_length",
                         "benefits_score", "n_skills"]}
emp_cats = [c.replace("employment_type_", "") for c in FEATURES if c.startswith("employment_type_")]

market_aggregates = {
    "metrics": metrics, "num_medians": num_medians, "ordinal_specs": ordinal_specs,
    "emp_cats": emp_cats, "career_ladder": career_ladder, "country_median": country_median,
    "skill_premium": skill_premium, "remote_median": remote_median, "salary_quantiles": quant,
    "global_median": int(round(df_ml["salary_usd"].median())),
    "overall_skill_freq_mean": {c: float(np.mean(list(freq_maps[c].values()))) for c in freq_maps},
}

# ---- save all nine artefacts ----
artefacts = {
    "tuned_gbr_model.joblib"     : tuned_model,
    "standard_scaler.joblib"     : scaler,
    "ordinal_encoders.joblib"    : ordinal_encoders,
    "frequency_maps.joblib"      : freq_maps,
    "feature_names.joblib"       : FEATURES,
    "skill_feature_names.joblib" : SKILL_FEATS,
    "top5_skill_strings.joblib"  : top5_skills,
    "dashboard_data.joblib"      : df_ml[["salary_usd", "company_location",
                                          "exp_label", "remote_label"] + SKILL_FEATS].copy(),
    "market_aggregates.joblib"   : market_aggregates,
}
for fname, obj in artefacts.items():
    joblib.dump(obj, os.path.join(deploy_dir, fname))
    print(f"  Saved: {fname}")

print(f"\nAll {len(artefacts)} artefacts written to ./{deploy_dir}/")
print("Tuned metrics:", metrics)
```

--------------------------------------------------------------------------------
## 2. Replace the local launch code cell (your current one points to "xxx" and blocks the kernel)

Running a Streamlit server from inside a notebook cell freezes the kernel, so present the local
launch as instructions instead. Use a **markdown** cell:

> **Run the dashboard locally**
>
> 1. Put `hr_dashboard.py` and the `deployment_artefacts/` folder in the same directory.
> 2. In a terminal in that directory, install the requirements once:
>    `pip install -r requirements.txt`
> 3. Start the app: `streamlit run hr_dashboard.py`
> 4. The dashboard opens at `http://localhost:8501`.

If you want a one-cell sanity check that the files are present (this does **not** block), use:

```python
from pathlib import Path
need = ["hr_dashboard.py", "deployment_artefacts"]
for n in need:
    print(("  found:  " if Path(n).exists() else "  MISSING:") + " " + n)
print("Ready. Launch from a terminal with:  streamlit run hr_dashboard.py")
```

--------------------------------------------------------------------------------
## 3. Working Google Colab launch cell (replaces your commented-out cell)

Uncomment, set `DRIVE_FOLDER`, and optionally add a free ngrok token for a stable URL.

```python
import os, time, shutil, subprocess

DRIVE_FOLDER = "/content/drive/MyDrive/Data Analytics"   # <-- set to your Drive folder
WORK = "/content/fairpay"
os.makedirs(WORK, exist_ok=True)

from google.colab import drive
drive.mount("/content/drive")

# copy app + artefacts into the Colab working directory
shutil.copy2(os.path.join(DRIVE_FOLDER, "hr_dashboard.py"), os.path.join(WORK, "hr_dashboard.py"))
art_src = os.path.join(DRIVE_FOLDER, "deployment_artefacts")
art_dst = os.path.join(WORK, "deployment_artefacts")
if os.path.isdir(art_dst):
    shutil.rmtree(art_dst)
shutil.copytree(art_src, art_dst)
print("Copied app and", len(os.listdir(art_dst)), "artefacts.")

subprocess.run(["pip", "install", "-q", "streamlit", "pyngrok"], check=True)

os.chdir(WORK)
proc = subprocess.Popen(
    ["streamlit", "run", "hr_dashboard.py", "--server.port", "8501",
     "--server.headless", "true", "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(6)

from pyngrok import ngrok
# ngrok.set_auth_token("YOUR_TOKEN")   # optional: free token from dashboard.ngrok.com
public_url = ngrok.connect(8501).public_url
print("\n" + "=" * 56)
print("FairPay Validator is live at:", public_url)
print("=" * 56)
print("To stop:  proc.terminate(); ngrok.kill()")
```
