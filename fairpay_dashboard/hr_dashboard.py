"""
hr_dashboard.py  —  FairPay Validator Dashboard
WQD7003 Data Analytics · Group 12

Run:   streamlit run hr_dashboard.py
Note:  Works without scikit-learn (Market Estimate fallback mode).
"""

import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairPay Validator · AI Salary Benchmarking",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"  # Forces the sidebar open on load!
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":     "#080D1A",
    "card":   "#0F1629",
    "card2":  "#162035",
    "border": "#1E2D45",
    "blue":   "#3B82F6",
    "indigo": "#6366F1",
    "gold":   "#F59E0B",
    "green":  "#10B981",
    "red":    "#EF4444",
    "amber":  "#F97316",
    "text":   "#E2E8F0",
    "subtle": "#94A3B8",
    "muted":  "#4B5E78",
}

PLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,41,0.6)",
    font=dict(color=C["text"], family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    title_font=dict(size=13, color=C["subtle"]),
    xaxis=dict(gridcolor=C["border"], zeroline=False),
    yaxis=dict(gridcolor=C["border"], zeroline=False),
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ARTEFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deployment_artefacts")

EXP_MAP   = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
EXP_OPTS  = ["EN", "MI", "SE", "EX"]
EDU_OPTS  = ["Associate", "Bachelor", "Master", "PhD"]
SIZE_MAP  = {"S": "Small (< 50)", "M": "Medium (50–250)", "L": "Large (250+)"}
SIZE_OPTS = ["S", "M", "L"]
EMP_MAP   = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
EMP_OPTS  = ["FT", "PT", "CT", "FL"]
REMOTE    = {0: "On-site (0%)", 50: "Hybrid (50%)", 100: "Fully Remote (100%)"}

CAREER_STAGE_KEYS   = ["Entry", "Mid", "Senior", "Executive"]
CAREER_STAGE_LABELS = ["Entry (EN)", "Mid (MI)", "Senior (SE)", "Executive (EX)"]

TOP_JOB_TITLES = [
    "Data Scientist", "Machine Learning Engineer", "AI Research Scientist",
    "Data Engineer", "Data Analyst", "NLP Engineer", "Computer Vision Engineer",
    "Deep Learning Engineer", "MLOps Engineer", "Data Science Manager",
    "Applied Scientist", "Research Engineer", "AI Product Manager",
    "Quantitative Analyst", "BI Engineer", "AI Architect",
]
TOP_LOCATIONS = [
    "United States", "United Kingdom", "Canada", "Germany", "France",
    "India", "Australia", "Netherlands", "Spain", "Brazil",
    "Singapore", "Japan", "Switzerland", "Sweden", "Norway",
    "Denmark", "Ireland", "Poland", "Italy", "Portugal",
]
TOP_INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "E-commerce", "Manufacturing",
    "Consulting", "Education", "Media", "Telecommunications", "Government",
    "Research", "Retail", "Energy", "Transportation", "Real Estate",
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — no sidebar needed, so no sidebar CSS complexity
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main .block-container {{
    background-color: {C["bg"]} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: {C["text"]};
    padding-top: 0.5rem !important;
}}

#  FIXED CODE:
/* hide Streamlit chrome safely without breaking slider components */
[data-testid="stHeader"]    {{ background: transparent !important; height: 0px !important; min-height: 0px !important; overflow: hidden !important; }}
[data-testid="stToolbar"]   {{ display: none !important; }}
[data-testid="stDecoration"]{{ display: none !important; }}
footer                      {{ display: none !important; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{
    gap:            4px;
    background:     {C["card"]} !important;
    border-radius:  12px !important;
    padding:        4px !important;
    border:         1px solid {C["border"]} !important;
    margin-bottom:  1.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:  8px !important;
    color:          {C["subtle"]} !important;
    font-weight:    500 !important;
    font-size:      0.88rem !important;
    padding:        0.5rem 1.4rem !important;
    background:     transparent !important;
    border:         none !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {C["card2"]} !important;
    color:      {C["text"]} !important;
}}
.stTabs [aria-selected="true"] {{
    background: {C["card2"]} !important;
    color:      {C["text"]} !important;
    font-weight:700 !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* ── inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div {{
    background-color: {C["card2"]} !important;
    border:           1px solid {C["border"]} !important;
    border-radius:    8px !important;
}}
div[data-baseweb="select"] span,
div[data-baseweb="input"] input {{ color: {C["text"]} !important; }}
div[data-baseweb="popover"] *, li[role="option"] {{
    background-color: {C["card"]} !important;
    color:            {C["text"]} !important;
}}
div[data-testid="stSlider"] > div > div > div {{ background: {C["blue"]} !important; }}
div[data-testid="stSlider"] div[role="slider"] {{
    background:   {C["blue"]} !important;
    border-color: {C["blue"]} !important;
    box-shadow:   0 0 8px rgba(59,130,246,0.5) !important;
}}

/* ── button ── */
.stButton > button {{
    background:     linear-gradient(135deg, {C["blue"]} 0%, {C["indigo"]} 100%) !important;
    color:          #fff !important;
    border:         none !important;
    border-radius:  10px !important;
    padding:        0.65rem 1.8rem !important;
    font-weight:    600 !important;
    font-size:      0.92rem !important;
    letter-spacing: 0.03em !important;
    box-shadow:     0 4px 18px rgba(59,130,246,0.4) !important;
    transition:     transform 0.12s, box-shadow 0.12s !important;
    width:          100% !important;
}}
.stButton > button:hover {{
    transform:  translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(59,130,246,0.55) !important;
}}

/* ── metric cards ── */
[data-testid="metric-container"] {{
    background:    {C["card"]} !important;
    border:        1px solid {C["border"]} !important;
    border-radius: 12px !important;
    padding:       1rem 1.2rem !important;
}}
[data-testid="stMetricValue"] {{
    color:       {C["gold"]} !important;
    font-weight: 700 !important;
    font-size:   1.5rem !important;
}}
[data-testid="stMetricLabel"] {{
    color:          {C["subtle"]} !important;
    font-size:      0.76rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}}

/* ── labels ── */
label, [data-testid="stWidgetLabel"] {{
    color:          {C["subtle"]} !important;
    font-size:      0.76rem !important;
    font-weight:    600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}

/* ── expander ── */
        [data-testid="stExpander"] {{
            background:    {C["card"]} !important;
            border:        1px solid {C["border"]} !important;
            border-radius: 10px !important;
        }}

        /* ── caption / info ── */
        [data-testid="stCaptionContainer"] {{ color: {C["muted"]} !important; font-size: 0.74rem !important; }}

        /* ── STREAMLIT SLIDER LABELS PERMANENT VISIBILITY FIX ── */
        div[data-testid="stSlider"] {{
            color: {C["text"]} !important;
        }}

        /* Forces the dynamic numeric step/tick values below the slider track to show up clearly */
        div[data-testid="stSlider"] div[data-presentation="slider"] span,
        div[data-testid="stSlider"] div[role="presentation"] span,
        div[data-testid="stSlider"] span {{
            color: {C["text"]} !important;
            opacity: 1 !important;
        }}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS  (sklearn-optional)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading artefacts…")
def load_artefacts():
    base = ARTEFACT_DIR
    if not os.path.isdir(base):
        st.error(f"**`{base}/` folder not found.** Run notebook Stage 6.1 first.")
        st.stop()

    def safe_load(filename):
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    freq_maps   = safe_load("frequency_maps.joblib")
    feat_names  = safe_load("feature_names.joblib")
    skill_feats = safe_load("skill_feature_names.joblib")
    top5        = safe_load("top5_skill_strings.joblib")
    dash_data   = safe_load("dashboard_data.joblib")
    market_agg  = safe_load("market_aggregates.joblib")

    if any(v is None for v in [freq_maps, feat_names, skill_feats, top5, dash_data, market_agg]):
        st.error("**Core artefacts missing.** Re-run notebook Stage 6.1.")
        st.stop()

    model   = safe_load("tuned_gbr_model.joblib")
    ord_enc = safe_load("ordinal_encoders.joblib")
    ml_mode = model is not None and ord_enc is not None

    return {
        "model": model, "ord_enc": ord_enc,
        "freq_maps": freq_maps, "feat_names": feat_names,
        "skill_feats": skill_feats, "top5_skills": top5,
        "dash_data": dash_data, "market_agg": market_agg,
        "ml_mode": ml_mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_salary_ml(inputs, art):
    freq  = art["freq_maps"];  oe = art["ord_enc"]
    feats = art["feat_names"]; top5 = art["top5_skills"]; model = art["model"]
    def oe_t(col, val): return float(oe[col].transform([[str(val)]])[0][0])
    num = {"remote_ratio": float(inputs["remote_ratio"]),
           "years_experience": float(inputs["years_experience"]),
           "job_description_length": float(inputs.get("jd_len", 500)),
           "benefits_score": float(inputs.get("benefits_score", 60.0)),
           "n_skills": float(inputs.get("n_skills", 3))}
    ord_ = {"experience_level_ord":   oe_t("experience_level",   inputs["experience_level"]),
            "education_required_ord": oe_t("education_required", inputs["education_required"]),
            "company_size_ord":       oe_t("company_size",       inputs["company_size"])}
    et  = inputs["employment_type"]
    ohe = {"employment_type_CT": 1.0 if et=="CT" else 0.0,
           "employment_type_FL": 1.0 if et=="FL" else 0.0,
           "employment_type_FT": 1.0 if et=="FT" else 0.0,
           "employment_type_PT": 1.0 if et=="PT" else 0.0}
    frq = {"job_title_freq":          freq["job_title"].get(inputs["job_title"], 0.0),
           "company_location_freq":   freq["company_location"].get(inputs["company_location"], 0.0),
           "employee_residence_freq": freq["employee_residence"].get(inputs["employee_residence"], 0.0),
           "industry_freq":           freq["industry"].get(inputs["industry"], 0.0)}
    user_skills = set(inputs.get("skills", []))
    skl = {f"has_{s.replace(' ','_')}": 1.0 if s in user_skills else 0.0 for s in top5}
    all_vals = {**num, **ord_, **ohe, **frq, **skl}
    X = np.array([all_vals.get(f, 0.0) for f in feats], dtype=float).reshape(1, -1)
    return float(np.expm1(model.predict(X)[0]))

def predict_salary_market(inputs, m):
    ladder    = m.get("career_ladder",  {})
    countries = m.get("country_median", {})
    rem_med   = m.get("remote_median",  {})
    skill_med = m.get("skill_premium",  {})
    gmed      = m.get("global_median",  120_000)
    exp_key   = EXP_MAP[inputs["experience_level"]]
    base      = float(ladder.get(exp_key, gmed))
    loc = inputs["company_location"]
    if loc in countries:
        base *= (0.5 + 0.5 * countries[loc] / gmed)
    rkey = {0: "On-site", 50: "Hybrid", 100: "Fully remote"}.get(inputs["remote_ratio"], "On-site")
    if rkey in rem_med:
        base += (rem_med[rkey] - gmed) * 0.3
    edu_delta = {"Associate": -9_000, "Bachelor": 0, "Master": 12_000, "PhD": 22_000}
    base += edu_delta.get(inputs["education_required"], 0)
    for skill in inputs.get("skills", []):
        if skill in skill_med:
            base += (skill_med[skill] - gmed) * 0.22
    base += np.log1p(float(inputs.get("years_experience", 5))) * 2_800
    base += {"S": -5_000, "M": 0, "L": 8_000}.get(inputs["company_size"], 0)
    base += min(int(inputs.get("n_skills", 3)) * 1_200, 12_000)
    return float(max(base, 20_000))

def predict_salary(inputs, art):
    return (predict_salary_ml(inputs, art) if art["ml_mode"]
            else predict_salary_market(inputs, art["market_agg"]))


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def pill(label, color):
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
            f'border-radius:999px;padding:0.18rem 0.6rem;font-size:0.72rem;'
            f'font-weight:600;letter-spacing:0.05em;">{label}</span>')

def merge_opts(curated, freq_maps, freq_key, top_n=40):
    keys   = sorted(freq_maps.get(freq_key, {}).keys(),
                    key=lambda k: -freq_maps[freq_key][k])[:top_n]
    extras = [k for k in keys if k not in curated]
    return curated + extras


# ─────────────────────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header(art):
    m       = art["market_agg"]
    metrics = m.get("metrics", {})
    r2      = metrics.get("r2",   0.903)
    mae     = metrics.get("mae",  16426)
    mape    = metrics.get("mape", 13.45)
    naive   = metrics.get("naive_mae", 49661)
    ml_mode = art["ml_mode"]
    mode_color = C["green"] if ml_mode else C["amber"]
    mode_label = "GBR ML Model" if ml_mode else "Market Estimate"
    top5    = art["top5_skills"]

    st.markdown(f"""
<div style="background:linear-gradient(135deg,{C['card']} 0%,{C['card2']} 100%);
            border:1px solid {C['border']};border-radius:16px;
            padding:1.5rem 2rem;margin-bottom:1.25rem;
            border-left:4px solid {C['gold']};">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.35rem;">
        <span style="font-size:1.8rem;filter:drop-shadow(0 0 10px rgba(245,158,11,0.4));">⚖️</span>
        <span style="font-size:1.5rem;font-weight:700;color:{C['text']};
                     letter-spacing:-0.02em;">FairPay Validator</span>
      </div>
      <div style="font-size:0.82rem;color:{C['muted']};">
        WQD7003 Data Analytics &nbsp;·&nbsp; Group 12 &nbsp;·&nbsp;
        AI Salary Benchmarking &nbsp;&nbsp;
        {pill("WQD7003", C['indigo'])} {pill("Group 12", C['muted'])}
        &nbsp; <span style="background:{mode_color}22;color:{mode_color};
                            border:1px solid {mode_color}44;
                            border-radius:999px;padding:0.18rem 0.6rem;
                            font-size:0.72rem;font-weight:600;">⚡ {mode_label}</span>
      </div>
    </div>
    <div style="font-size:0.75rem;color:{C['muted']};text-align:right;line-height:1.8;">
      15,000-row AI job postings dataset &nbsp;·&nbsp; Semester 2, 2025/2026<br>
      Top-5 model skills:&nbsp;
      {'&nbsp; '.join(f'<span style="color:{C["blue"]};font-weight:600;">{s}</span>' for s in top5)}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",        f"{r2:.3f}",              help="Explained variance on held-out test set")
    c2.metric("MAE",             f"${mae:,}",              help="Mean absolute error in USD")
    c3.metric("MAPE",            f"{mape:.2f}%",           help="Mean absolute percentage error")
    c4.metric("vs Naive Baseline", f"-{(1-mae/naive)*100:.0f}% error", help=f"Naive MAE was ${naive:,}")

    if not ml_mode:
        st.markdown(f"""
<div style="background:{C['amber']}15;border:1px solid {C['amber']}44;
            border-left:3px solid {C['amber']};border-radius:0 10px 10px 0;
            padding:0.6rem 1rem;margin-top:0.75rem;
            display:flex;align-items:center;gap:0.75rem;">
  <span style="font-size:1rem;">⚠️</span>
  <div style="font-size:0.8rem;color:{C['subtle']};">
    <strong style="color:{C['amber']};">Market Estimate Mode</strong> —
    scikit-learn is not compatible with your Python version.
    Module A uses market-median statistics. Modules B & C are fully functional.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"<div style='height:0.25rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE A  —  SALARY ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def module_a(art):
    m     = art["market_agg"]
    mae   = m.get("metrics", {}).get("mae", 16426)
    gmed  = m.get("global_median", 120_000)
    quant = m.get("salary_quantiles", {})
    top5  = art["top5_skills"]
    fm    = art["freq_maps"]

    jt_opts  = merge_opts(TOP_JOB_TITLES, fm, "job_title",          50)
    loc_opts = merge_opts(TOP_LOCATIONS,  fm, "company_location",   50)
    res_opts = merge_opts(TOP_LOCATIONS,  fm, "employee_residence", 50)
    ind_opts = merge_opts(TOP_INDUSTRIES, fm, "industry",           50)

    col_form, _, col_result = st.columns([5, 0.4, 4.6])

    with col_form:
        st.markdown(f"""<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.1em;color:{C['text']};margin-bottom:0.85rem;">📋 Job Configuration Details</div>""",
            unsafe_allow_html=True)

        r1a, r1b = st.columns(2)
        with r1a: exp  = st.selectbox("Experience Level", EXP_OPTS, format_func=lambda x: f"{EXP_MAP[x]} ({x})")
        with r1b: edu  = st.selectbox("Education Required", EDU_OPTS, index=2)
        r2a, r2b = st.columns(2)
        with r2a: size = st.selectbox("Company Size", SIZE_OPTS, format_func=lambda x: SIZE_MAP[x], index=1)
        with r2b: emp  = st.selectbox("Employment Type", EMP_OPTS, format_func=lambda x: EMP_MAP[x])
        r3a, r3b = st.columns(2)
        with r3a: remote = st.select_slider("Work Arrangement", options=[0,50,100], value=0, format_func=lambda x: REMOTE[x])
        with r3b: yoe    = st.slider("Years of Experience", 0, 25, 5)
        r4a, r4b = st.columns(2)
        with r4a: jt  = st.selectbox("Job Title", jt_opts)
        with r4b: loc = st.selectbox("Company Location", loc_opts)
        r5a, r5b = st.columns(2)
        with r5a: res = st.selectbox("Employee Residence", res_opts)
        with r5b: ind = st.selectbox("Industry", ind_opts)

        sel_skills = st.multiselect("Top-5 Model Skills (select those required)", options=top5, default=[])
        n_extra    = st.slider("Additional skills (not in top-5)", 0, 15, 2)

        with st.expander("⚙️  Advanced options"):
            jd_len    = st.slider("Job Description Length (chars)", 100, 3000, 500, 50)
            ben_score = st.slider("Benefits Score", 0.0, 100.0, 60.0, 5.0)

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Calculate Salary Estimate", use_container_width=True)

    with col_result:
        if not predict_btn:
            st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:16px;
            padding:3rem 2rem;text-align:center;min-height:440px;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
  <div style="font-size:3.5rem;margin-bottom:1.2rem;opacity:0.3;">⚖️</div>
  <div style="font-size:0.9rem;color:{C['muted']};line-height:1.7;max-width:220px;">
    Fill in the job details and click
    <strong style='color:{C["blue"]};'>Calculate Salary Estimate</strong>.
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            inputs = {
                "experience_level": exp,   "education_required": edu,
                "company_size": size,      "employment_type": emp,
                "remote_ratio": remote,    "years_experience": yoe,
                "job_title": jt,           "company_location": loc,
                "employee_residence": res, "industry": ind,
                "skills": sel_skills,
                "n_skills": len(sel_skills) + n_extra,
                "jd_len": jd_len,          "benefits_score": ben_score,
            }
            with st.spinner("Calculating…"):
                try:
                    pred = predict_salary(inputs, art)
                except Exception as e:
                    st.error(f"Prediction failed: {e}"); return

            lo  = max(pred - mae, 0);  hi = pred + mae
            pct = float((art["dash_data"]["salary_usd"] < pred).mean() * 100)
            bar_col  = C["green"] if pct >= 50 else C["gold"]
            zone     = "Above market" if pred >= gmed else "Below market"
            zone_col = C["green"]     if pred >= gmed else C["red"]
            q10 = quant.get("0.1", 60_000);  q90 = quant.get("0.9", 200_000)
            method_tag = "GBR ML Model" if art["ml_mode"] else "Market Estimate"
            method_col = C["blue"]     if art["ml_mode"] else C["amber"]

            st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};
            border-radius:16px;padding:1.75rem;border-top:3px solid {C['gold']};">
  <div style="text-align:center;padding:0.5rem 0 1.25rem;">
    <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.12em;color:{C['muted']};margin-bottom:0.4rem;">Predicted Annual Salary</div>
    <div style="font-size:3.2rem;font-weight:700;color:{C['gold']};letter-spacing:-0.03em;
                line-height:1;text-shadow:0 0 40px rgba(245,158,11,0.25);">${pred:,.0f}</div>
    <div style="font-size:0.8rem;color:{C['muted']};margin-top:0.3rem;">USD / year</div>
    <div style="margin-top:0.55rem;display:flex;justify-content:center;gap:0.5rem;">
      <span style="background:{zone_col}22;color:{zone_col};border:1px solid {zone_col}44;
                   border-radius:999px;padding:0.15rem 0.7rem;font-size:0.72rem;font-weight:600;">{zone}</span>
      <span style="background:{method_col}22;color:{method_col};border:1px solid {method_col}44;
                   border-radius:999px;padding:0.15rem 0.7rem;font-size:0.72rem;font-weight:600;">{method_tag}</span>
    </div>
  </div>
  <div style="background:{C['card2']};border:1px solid {C['border']};border-radius:10px;
              padding:0.8rem 1.1rem;margin-bottom:1rem;text-align:center;">
    <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                color:{C['muted']};margin-bottom:0.3rem;">Estimate Range (± ${mae:,})</div>
    <div style="font-size:1.05rem;font-weight:600;color:{C['text']};">
      ${lo:,.0f} <span style="color:{C['muted']};font-size:0.85rem;margin:0 0.35rem;">–</span> ${hi:,.0f}
    </div>
  </div>
  <div style="margin-bottom:1.2rem;">
    <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
      <span style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
                   letter-spacing:0.1em;color:{C['muted']};">Market Percentile</span>
      <span style="font-size:0.78rem;font-weight:600;color:{bar_col};">Top {100-pct:.0f}%</span>
    </div>
    <div style="background:{C['border']};border-radius:999px;height:7px;overflow:hidden;">
      <div style="background:linear-gradient(90deg,{C['blue']},{C['gold']});
                  height:7px;width:{min(pct,100):.0f}%;border-radius:999px;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.67rem;
                color:{C['muted']};margin-top:0.3rem;">
      <span>P10 ${q10:,}</span><span>Median ${gmed:,}</span><span>P90 ${q90:,}</span>
    </div>
  </div>
  <div style="border-top:1px solid {C['border']};padding-top:1rem;">
    <div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:{C['muted']};margin-bottom:0.65rem;">Role Summary</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.32rem 0.5rem;font-size:0.8rem;">
      <div style="color:{C['muted']};">Experience</div><div style="color:{C['text']};font-weight:500;">{EXP_MAP[exp]}</div>
      <div style="color:{C['muted']};">Education</div><div style="color:{C['text']};font-weight:500;">{edu}</div>
      <div style="color:{C['muted']};">Location</div><div style="color:{C['text']};font-weight:500;">{loc}</div>
      <div style="color:{C['muted']};">Work Mode</div><div style="color:{C['text']};font-weight:500;">{REMOTE[remote].split("(")[0].strip()}</div>
      <div style="color:{C['muted']};">Industry</div><div style="color:{C['text']};font-weight:500;">{ind}</div>
      <div style="color:{C['muted']};">Total Skills</div><div style="color:{C['text']};font-weight:500;">{len(sel_skills)+n_extra}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            note = (f"±${mae:,} = test-set MAE." if art["ml_mode"]
                    else "Market Estimate: range reflects dataset spread.")
            st.caption(f"⚠️  {note}  Use as first-round guidance only.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE B  —  MARKET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def module_b(art):
    dash = art["dash_data"]
    m    = art["market_agg"]
    gmed = m.get("global_median", 120_000)

    st.markdown(f"""<div style="font-size:0.88rem;font-weight:600;color:{C['subtle']};
        margin-bottom:0.75rem;">🌍  Geographic Salary Distribution</div>""",
        unsafe_allow_html=True)

    all_countries = dash["company_location"].value_counts().head(20).index.tolist()
    sel_countries = st.multiselect("Select countries to compare",
                                   options=all_countries, default=all_countries[:9])
    if sel_countries:
        sub   = dash[dash["company_location"].isin(sel_countries)]
        order = (sub.groupby("company_location")["salary_usd"]
                    .median().sort_values(ascending=False).index.tolist())
        fig_geo = px.box(sub, x="company_location", y="salary_usd",
                         color="company_location",
                         category_orders={"company_location": order},
                         color_discrete_sequence=[C["blue"],C["indigo"],C["green"],C["gold"],
                             "#8B5CF6","#EC4899","#14B8A6","#F97316","#06B6D4","#84CC16"],
                         labels={"company_location": "", "salary_usd": "Annual Salary (USD)"})
        fig_geo.add_hline(y=gmed, line_dash="dash", line_color=C["gold"], line_width=1.5,
                          annotation_text=f" Global Median  ${gmed:,}",
                          annotation_font=dict(color=C["gold"], size=11),
                          annotation_position="top right")
        #  THE FIXED CODE (Option B):
        fig_geo.update_layout(
            template=PLY.get("template"),
            paper_bgcolor=PLY.get("paper_bgcolor"),
            plot_bgcolor=PLY.get("plot_bgcolor"),
            font=PLY.get("font"),
            margin=PLY.get("margin"),
            title_font=PLY.get("title_font"),
            title="Salary Distribution by Country (ranked by median)",
            showlegend=False, 
            height=430,
            yaxis=dict(gridcolor=C["border"], zeroline=False, tickprefix="$", tickformat=","),
            xaxis=dict(gridcolor=C["border"], zeroline=False, tickangle=-25)
        )
        st.plotly_chart(fig_geo, use_container_width=True)
    else:
        st.info("Select at least one country.")

    st.markdown(f"<hr style='border-color:{C['border']};margin:1.75rem 0;'/>",
                unsafe_allow_html=True)
    st.markdown(f"""<div style="font-size:0.88rem;font-weight:600;color:{C['subtle']};
        margin-bottom:0.75rem;">💡  Skill-to-Salary Premium (Top-15 Market Skills)</div>""",
        unsafe_allow_html=True)

    sp_raw = m.get("skill_premium", {})
    if sp_raw:
        rows  = [{"Skill": s, "Median": v, "Premium": v - gmed} for s, v in sp_raw.items()]
        sk_df = pd.DataFrame(rows).sort_values("Premium", ascending=True)
        col_b1, col_b2 = st.columns([3, 2], gap="large")
        with col_b1:
            bar_cols  = [C["green"] if p >= 0 else C["red"] for p in sk_df["Premium"]]
            fig_skill = go.Figure()
            fig_skill.add_trace(go.Bar(
                x=sk_df["Premium"], y=sk_df["Skill"], orientation="h",
                marker=dict(color=bar_cols, line=dict(color="rgba(0,0,0,0)", width=0)),
                text=[f"${p:+,.0f}" for p in sk_df["Premium"]],
                textposition="outside",
                textfont=dict(color=C["text"], size=11, family="Inter"), width=0.6))
            fig_skill.add_vline(x=0, line_color=C["border"], line_width=1)
            fig_skill.update_layout(PLY, height=420, title="Salary Premium vs. Market Median")
            fig_skill.update_xaxes(tickprefix="$", tickformat=",", title="Premium (USD)")
            fig_skill.update_yaxes(title="")
            st.plotly_chart(fig_skill, use_container_width=True)
        with col_b2:
            st.markdown(f"""<div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.1em;color:{C['muted']};margin-bottom:0.65rem;margin-top:0.3rem;">
                Skill Breakdown</div>""", unsafe_allow_html=True)
            for _, row in sk_df.sort_values("Premium", ascending=False).iterrows():
                sign    = "+" if row["Premium"] >= 0 else ""
                pcol    = C["green"] if row["Premium"] >= 0 else C["red"]
                pct_bar = min(abs(row["Premium"]) / 25_000 * 100, 100)
                st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:10px;
            padding:0.6rem 0.9rem;margin-bottom:0.4rem;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.28rem;">
    <span style="font-size:0.82rem;color:{C['text']};font-weight:600;">{row['Skill']}</span>
    <span style="font-size:0.79rem;color:{pcol};font-weight:700;">{sign}${row['Premium']:,.0f}</span>
  </div>
  <div style="background:{C['border']};border-radius:999px;height:4px;overflow:hidden;">
    <div style="background:{pcol};height:4px;width:{pct_bar:.0f}%;border-radius:999px;"></div>
  </div>
  <div style="font-size:0.69rem;color:{C['muted']};margin-top:0.25rem;">Median: ${row['Median']:,}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE C  —  CAREER LADDER
# ─────────────────────────────────────────────────────────────────────────────
def module_c(art):
    m             = art["market_agg"]
    cl_raw        = m.get("career_ladder", {})
    remote_median = m.get("remote_median", {})
    gmed          = m.get("global_median", 120_000)

    medians = [cl_raw.get(k, 0) for k in CAREER_STAGE_KEYS]
    stages  = CAREER_STAGE_LABELS;  n = len(stages)

    cs1, cs2, _ = st.columns([2, 2, 3])
    with cs1: from_stage = st.selectbox("Current Stage", stages, index=0)
    with cs2: to_stage   = st.selectbox("Target Stage",  stages, index=2)

    fi = stages.index(from_stage);  ti = stages.index(to_stage)
    if ti <= fi: ti = min(fi + 1, n - 1)
    from_sal = medians[fi];  to_sal = medians[ti]
    uplift   = to_sal - from_sal
    uplift_pct = uplift / from_sal * 100 if from_sal else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Salary", f"${from_sal:,}")
    k2.metric("Target Salary",  f"${to_sal:,}")
    k3.metric("Total Uplift",   f"+${uplift:,}")
    k4.metric("% Increase",     f"+{uplift_pct:.0f}%")
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    bar_cols = [C["blue"] if i==fi else C["gold"] if i==ti
                else C["indigo"] if fi<i<ti else C["border"]
                for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stages, y=medians,
                         marker=dict(color=bar_cols, line=dict(color="rgba(0,0,0,0)", width=0)),
                         text=[f"${v//1000}k" for v in medians],
                         textposition="outside",
                         textfont=dict(size=14, color=C["text"], family="Inter"),
                         width=0.5))
    fig.add_trace(go.Scatter(x=stages[fi:ti+1], y=medians[fi:ti+1],
                             mode="lines+markers",
                             line=dict(color=C["gold"], width=2.5, dash="dot"),
                             marker=dict(size=10, color=C["gold"],
                                         line=dict(color=C["bg"], width=2)),
                             showlegend=False))
    for idx, lbl, col_ in [(fi,"Current",C["blue"]),(ti,"Target",C["gold"])]:
        fig.add_annotation(x=stages[idx], y=medians[idx], yshift=44,
                           text=f"<b>{lbl}</b>", showarrow=False,
                           font=dict(color=col_, size=11, family="Inter"),
                           bgcolor=C["card"], bordercolor=col_, borderwidth=1, borderpad=4)
    fig.update_layout(
        PLY, 
        showlegend=False, 
        height=430,
        title="Salary Distribution by Career Stage",
        margin=dict(l=50, r=20, t=50, b=50)
    )
    fig.update_yaxes(tickprefix="", tickformat="", title="")
    fig.update_xaxes(tickprefix="$", tickformat=",", tickangle=-25, title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
        letter-spacing:0.1em;color:{C['muted']};margin-bottom:0.75rem;margin-top:0.2rem;">
        Stage-by-Stage Jumps</div>""", unsafe_allow_html=True)
    jcols = st.columns(n - 1)
    for i in range(n - 1):
        jump = medians[i+1] - medians[i];  jp = jump/medians[i]*100 if medians[i] else 0
        in_range = fi <= i < ti
        with jcols[i]:
            st.markdown(f"""
<div style="background:{C['card']};border:1.5px solid {C['gold'] if in_range else C['border']};
            border-radius:12px;padding:0.9rem;text-align:center;">
  <div style="font-size:0.63rem;color:{C['muted']};text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:0.35rem;">{CAREER_STAGE_KEYS[i]} → {CAREER_STAGE_KEYS[i+1]}</div>
  <div style="font-size:1.35rem;font-weight:700;color:{C['green'] if in_range else C['subtle']};">+${jump//1000:.0f}k</div>
  <div style="font-size:0.73rem;color:{C['muted']};margin-top:0.15rem;">+{jp:.0f}%</div>
</div>
""", unsafe_allow_html=True)

    if remote_median:
        st.markdown(f"""<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.1em;color:{C['muted']};margin:1.25rem 0 0.75rem;">
            Salary by Work Arrangement</div>""", unsafe_allow_html=True)
        rcols = st.columns(len(remote_median))
        for col_w, (arr, sal) in zip(rcols, remote_median.items()):
            diff = sal - gmed;  dcol = C["green"] if diff >= 0 else C["red"]
            with col_w:
                st.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['border']};border-radius:12px;
            padding:0.9rem;text-align:center;">
  <div style="font-size:0.7rem;color:{C['muted']};text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:0.4rem;">{arr}</div>
  <div style="font-size:1.25rem;font-weight:700;color:{C['gold']};">${sal:,}</div>
  <div style="font-size:0.73rem;color:{dcol};margin-top:0.2rem;">{"+" if diff>=0 else ""}${diff:,} vs global</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="background:{C['card2']};border:1px solid {C['border']};
            border-left:3px solid {C['indigo']};border-radius:0 12px 12px 0;
            padding:1rem 1.2rem;margin-top:1.1rem;">
  <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
              letter-spacing:0.1em;color:{C['muted']};margin-bottom:0.3rem;">📋  Business Case Snippet</div>
  <div style="font-size:0.87rem;color:{C['subtle']};line-height:1.65;">
    Promoting from <strong style='color:{C["text"]};'>{stages[fi]}</strong> to
    <strong style='color:{C["text"]};'>{stages[ti]}</strong> represents a median investment of
    <strong style='color:{C["gold"]};'>+${uplift:,} (+{uplift_pct:.0f}%)</strong>,
    growing from <strong style='color:{C["blue"]};'>${from_sal:,}</strong> to
    <strong style='color:{C["gold"]};'>${to_sal:,}</strong>/year.
    Based on 15,000 AI job postings. Adjust for local cost-of-living before use in offer letters.
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    art = load_artefacts()
    render_header(art)

    tab1, tab2, tab3 = st.tabs([
        "📊  Salary Estimator",
        "🌍  Market Explorer",
        "🪜  Career Ladder",
    ])
    with tab1: module_a(art)
    with tab2: module_b(art)
    with tab3: module_c(art)

if __name__ == "__main__":
    main()
