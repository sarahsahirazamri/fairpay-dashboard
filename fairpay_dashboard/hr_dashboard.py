"""
FairPay Validator: a Streamlit dashboard for the WQD7003 project (Group 12).

This app loads the nine .joblib artefacts your notebook writes to
`deployment_artefacts/` and serves the three modules from Stage 6.3:

    Module A  Salary Estimator        (Objective 4)
    Module B  Geographic and Skills   (Objectives 2 and 3)
    Module C  Career Progression      (Objective 1)

Navigation is by top tabs, not a sidebar. The estimator reproduces the exact
preprocessing from the notebook, encodes the inputs in the saved FEATURES
order, calls the tuned Gradient Boosting model, and converts the log
prediction back to US dollars with np.expm1.

Run locally:   streamlit run hr_dashboard.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants and theme
# --------------------------------------------------------------------------- #
ART_DIR = Path(__file__).parent / "deployment_artefacts"

INK = "#0E1F33"
EMERALD = "#0FA968"
GOLD = "#D99A06"
STEEL = "#5E7088"
LINE = "#E1E7F0"
SURFACE = "#FFFFFF"

# Human labels mapped to the codes the model was trained on
EXP_LABEL_TO_CODE = {"Entry": "EN", "Mid": "MI", "Senior": "SE", "Executive": "EX"}
EXP_CODE_TO_LABEL = {v: k for k, v in EXP_LABEL_TO_CODE.items()}
SIZE_LABEL_TO_CODE = {"Small": "S", "Medium": "M", "Large": "L"}
EMP_LABEL_TO_CODE = {"Full-time": "FT", "Part-time": "PT", "Contract": "CT", "Freelance": "FL"}
REMOTE_LABEL_TO_VALUE = {"On-site": 0, "Hybrid": 50, "Fully remote": 100}


# --------------------------------------------------------------------------- #
# Artefact loading (cached so it runs once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artefacts():
    """Load every artefact the notebook serialised. Returns a dict or raises."""
    needed = {
        "model": "tuned_gbr_model.joblib",
        "ordinal_encoders": "ordinal_encoders.joblib",
        "frequency_maps": "frequency_maps.joblib",
        "feature_names": "feature_names.joblib",
        "skill_feature_names": "skill_feature_names.joblib",
        "top5_skill_strings": "top5_skill_strings.joblib",
        "dashboard_data": "dashboard_data.joblib",
        "market_aggregates": "market_aggregates.joblib",
    }
    art = {}
    missing = []
    for key, fname in needed.items():
        path = ART_DIR / fname
        if not path.exists():
            missing.append(fname)
            continue
        art[key] = joblib.load(path)
    if missing:
        raise FileNotFoundError(missing)
    return art


# --------------------------------------------------------------------------- #
# Pure inference logic (no Streamlit calls, so it is easy to test)
# --------------------------------------------------------------------------- #
def build_feature_row(inp, art):
    """Encode a single set of inputs into a list ordered exactly like FEATURES."""
    oe = art["ordinal_encoders"]
    fm = art["frequency_maps"]
    feats = art["feature_names"]
    top5 = art["top5_skill_strings"]
    skill_feats = art["skill_feature_names"]

    row = {
        # numeric features
        "remote_ratio": float(inp["remote_ratio"]),
        "years_experience": float(inp["years_experience"]),
        "job_description_length": float(inp["job_description_length"]),
        "benefits_score": float(inp["benefits_score"]),
        "n_skills": float(inp["n_skills"]),
        # ordinal features (use the encoders fitted in the notebook)
        "experience_level_ord": float(oe["experience_level"].transform([[inp["experience_level"]]])[0][0]),
        "education_required_ord": float(oe["education_required"].transform([[inp["education"]]])[0][0]),
        "company_size_ord": float(oe["company_size"].transform([[inp["company_size"]]])[0][0]),
        # frequency features (unseen category falls back to 0.0, as in training)
        "job_title_freq": float(fm["job_title"].get(inp["job_title"], 0.0)),
        "company_location_freq": float(fm["company_location"].get(inp["company_location"], 0.0)),
        "employee_residence_freq": float(fm["employee_residence"].get(inp["employee_residence"], 0.0)),
        "industry_freq": float(fm["industry"].get(inp["industry"], 0.0)),
    }

    # one-hot employment type
    for code in ["CT", "FL", "FT", "PT"]:
        row[f"employment_type_{code}"] = 1.0 if inp["employment_type"] == code else 0.0

    # binary skill flags, aligned 1:1 with the saved skill strings
    selected = set(inp["skills"])
    for raw, flag in zip(top5, skill_feats):
        row[flag] = 1.0 if raw in selected else 0.0

    # assemble strictly in the order the model expects
    return [float(row[f]) for f in feats]


def predict_salary(inp, art):
    """Return the point estimate and the plus/minus MAE band in US dollars."""
    row = build_feature_row(inp, art)
    X = np.array([row], dtype=float)               # numpy array, matching training
    log_pred = float(art["model"].predict(X)[0])   # model predicts log1p(salary)
    usd = float(np.expm1(log_pred))                 # back to dollars
    mae = float(art["market_aggregates"]["metrics"]["mae"])
    return {"point": usd, "low": max(0.0, usd - mae), "high": usd + mae, "mae": mae}


def percentile_of(value, quantiles):
    """Approximate where a salary sits on the training distribution (0 to 100)."""
    qs = sorted((float(k), float(v)) for k, v in quantiles.items())
    probs = [q for q, _ in qs]
    vals = [v for _, v in qs]
    if value <= vals[0]:
        return max(1.0, probs[0] * 100 * value / max(vals[0], 1))
    if value >= vals[-1]:
        return min(99.0, probs[-1] * 100)
    return float(np.interp(value, vals, probs)) * 100


# --------------------------------------------------------------------------- #
# Small formatting and chart helpers
# --------------------------------------------------------------------------- #
def usd(n):
    return f"${n:,.0f}"


def usd_k(n):
    return f"${n/1000:,.0f}k"


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        hoverlabel=dict(font_size=13, font_family="Inter, sans-serif"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Page styling
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');
          html, body, [class*="css"] { font-family:'Inter', system-ui, sans-serif; }
          h1, h2, h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:-.01em; }
          .block-container { padding-top:4.75rem; max-width:1180px; }
          header[data-testid="stHeader"] { background:rgba(244,246,249,.85); backdrop-filter:blur(8px); }

          .fp-hero { background:linear-gradient(157deg,#0E1F33,#13283F 60%,#0d2138);
                     border-radius:18px; padding:26px 30px; color:#fff;
                     box-shadow:0 30px 70px -34px rgba(14,31,51,.5); margin-bottom:8px; }
          .fp-hero h1 { color:#fff; font-size:28px; margin:0 0 6px; }
          .fp-hero p { color:#C9D6E5; margin:0; font-size:14.5px; max-width:62ch; }
          .fp-kicker { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.16em;
                       text-transform:uppercase; color:#0FA968; font-weight:600; margin-bottom:10px; }

          .fp-result { background:linear-gradient(157deg,#0E1F33,#13283F 60%,#0d2138);
                       border-radius:18px; padding:26px 28px; color:#fff;
                       box-shadow:0 30px 70px -34px rgba(14,31,51,.5); }
          .fp-result .lab { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.12em;
                            text-transform:uppercase; color:#8295AC; }
          .fp-result .fig { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:52px;
                            line-height:1; margin:6px 0 2px; }
          .fp-result .fig .cur { color:#8295AC; font-size:26px; font-weight:500; vertical-align:top; }
          .fp-band { display:flex; justify-content:space-between; font-family:'JetBrains Mono',monospace;
                     font-size:13px; color:#C9D6E5; margin-top:18px; padding-top:16px;
                     border-top:1px solid rgba(255,255,255,.1); }
          .fp-band b { color:#fff; }
          .fp-band .mid b { color:#0FA968; }

          .fp-chip { display:inline-flex; align-items:center; gap:7px; background:#fff; border:1px solid #E1E7F0;
                     border-radius:99px; padding:7px 14px; font-size:13px; color:#5E7088; margin-right:8px; }
          .fp-chip b { font-family:'JetBrains Mono',monospace; color:#0E1F33; }
          .fp-chip .dot { width:7px; height:7px; border-radius:50%; background:#0FA968; }

          .fp-note { background:#FBF6E9; border:1px solid rgba(217,154,6,.3); border-radius:11px;
                     padding:13px 15px; font-size:13px; color:#6b5410; margin-top:14px; }
          /* top header bar (replaces the sidebar) */
          .fp-topbar { display:flex; align-items:center; justify-content:space-between; gap:16px;
                       flex-wrap:wrap; padding-bottom:16px; margin-bottom:4px; border-bottom:1px solid #E1E7F0; }
          .fp-brandname { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:20px;
                          color:#0E1F33; letter-spacing:-.02em; line-height:1; }
          .fp-brandsub { font-size:12.5px; color:#5E7088; margin-top:3px; }
          .fp-chips { display:flex; flex-wrap:wrap; align-items:center; }

          /* top tabs styling */
          .stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid #E1E7F0; }
          .stTabs [data-baseweb="tab"] { font-family:'Space Grotesk',sans-serif; font-weight:500;
                                         font-size:14.5px; padding-top:8px; padding-bottom:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Module A: Salary Estimator
# --------------------------------------------------------------------------- #
def module_a(art):
    agg = art["market_aggregates"]
    fm = art["frequency_maps"]
    metrics = agg["metrics"]
    quants = agg["salary_quantiles"]
    global_median = agg["global_median"]
    ladder = agg["career_ladder"]
    num_med = agg["num_medians"]

    st.markdown(
        f"""
        <div class="fp-hero">
          <div class="fp-kicker">Module A &middot; Objective 4</div>
          <h1>Salary Estimator</h1>
          <p>Enter a role and the tuned Gradient Boosting model returns a market benchmark, with a
          confidence band equal to the model's mean absolute error. This is decision support, not an
          automatic salary-setting engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    titles = sorted(fm["job_title"].keys())
    locations = sorted(fm["company_location"].keys())
    residences = sorted(fm["employee_residence"].keys())
    industries = sorted(fm["industry"].keys())
    skill_options = sorted(set(list(agg["skill_premium"].keys()) + list(art["top5_skill_strings"])))

    def default_idx(options, preferred):
        return options.index(preferred) if preferred in options else 0

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        c1, c2 = st.columns(2)
        exp_label = c1.selectbox("Career level", list(EXP_LABEL_TO_CODE.keys()), index=2)
        years = c2.slider("Years of experience", 0, 19, 9)

        job_title = st.selectbox("Job title", titles, index=default_idx(titles, "Data Scientist"))

        c3, c4 = st.columns(2)
        company_location = c3.selectbox("Company location", locations,
                                        index=default_idx(locations, "United States"))
        employee_residence = c4.selectbox("Employee residence", residences,
                                          index=default_idx(residences, "United States"))

        c5, c6 = st.columns(2)
        industry = c5.selectbox("Industry", industries, index=0)
        size_label = c6.selectbox("Company size", list(SIZE_LABEL_TO_CODE.keys()), index=1)

        c7, c8 = st.columns(2)
        edu = c7.selectbox("Education", ["Associate", "Bachelor", "Master", "PhD"], index=1)
        emp_label = c8.selectbox("Employment type", list(EMP_LABEL_TO_CODE.keys()), index=0)

        remote_label = st.radio("Work arrangement", list(REMOTE_LABEL_TO_VALUE.keys()),
                                index=1, horizontal=True)

        skills = st.multiselect(
            "Required skills (the five flagged skills carry extra model weight)",
            skill_options, default=["Python"],
        )

        with st.expander("Advanced inputs (optional)"):
            jd_len = st.slider("Job description length (characters)", 500, 2500,
                               int(round(num_med["job_description_length"])))
            benefits = st.slider("Benefits score", 5.0, 10.0,
                                 float(round(num_med["benefits_score"], 1)), 0.1)

    inp = {
        "experience_level": EXP_LABEL_TO_CODE[exp_label],
        "years_experience": years,
        "job_title": job_title,
        "company_location": company_location,
        "employee_residence": employee_residence,
        "industry": industry,
        "company_size": SIZE_LABEL_TO_CODE[size_label],
        "education": edu,
        "employment_type": EMP_LABEL_TO_CODE[emp_label],
        "remote_ratio": REMOTE_LABEL_TO_VALUE[remote_label],
        "skills": skills,
        "n_skills": max(len(skills), 1),
        "job_description_length": jd_len,
        "benefits_score": benefits,
    }

    result = predict_salary(inp, art)
    point, low, high = result["point"], result["low"], result["high"]
    pct = percentile_of(point, quants)
    vs_global = (point / global_median - 1) * 100
    level_median = ladder.get(exp_label, global_median)
    vs_level = (point / level_median - 1) * 100

    with right:
        st.markdown(
            f"""
            <div class="fp-result">
              <div class="lab">{exp_label} benchmark &middot; confidence &plusmn;{usd_k(result['mae'])}</div>
              <div class="fig"><span class="cur">$</span>{point:,.0f}</div>
              <div class="lab">expected annual salary, US dollars</div>
              <div class="fp-band">
                <span>low <b>{usd_k(low)}</b></span>
                <span class="mid">expected <b>{usd_k(point)}</b></span>
                <span>high <b>{usd_k(high)}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # gauge against the salary distribution
        q = {float(k): float(v) for k, v in quants.items()}
        axis_max = max(q[0.9] * 1.05, high * 1.05)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=point,
            number={"prefix": "$", "valueformat": ",.0f",
                    "font": {"family": "JetBrains Mono", "size": 26, "color": INK}},
            delta={"reference": global_median, "valueformat": ",.0f",
                   "increasing": {"color": EMERALD}, "decreasing": {"color": GOLD}},
            title={"text": f"vs global median ({usd_k(global_median)})",
                   "font": {"size": 12, "color": STEEL}},
            gauge={
                "axis": {"range": [q[0.1], axis_max], "tickformat": "$,.0s",
                         "tickfont": {"size": 10, "color": STEEL}},
                "bar": {"color": EMERALD, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [q[0.1], q[0.25]], "color": "#EEF1F6"},
                    {"range": [q[0.25], q[0.5]], "color": "#DDE4EE"},
                    {"range": [q[0.5], q[0.75]], "color": "#EEF1F6"},
                    {"range": [q[0.75], axis_max], "color": "#DDE4EE"},
                ],
                "threshold": {"line": {"color": GOLD, "width": 3}, "thickness": 0.85,
                              "value": global_median},
            },
        ))
        gauge.update_layout(height=240, margin=dict(l=20, r=20, t=44, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif", color=INK))
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        cc1, cc2 = st.columns(2)
        cc1.metric("Distribution position", f"{pct:.0f}th pct")
        cc2.metric("Versus this level's median", f"{vs_level:+.0f}%")

    st.markdown(
        f"""
        <div class="fp-note">
          The band of {usd_k(low)} to {usd_k(high)} reflects the production model's mean absolute error
          of {usd(result['mae'])} (tuned MAPE 13.45 percent, R squared 0.90 on the log scale). Treat the
          number as a midpoint for negotiation, then layer in internal equity and budget. All figures are
          nominal US dollars and are not adjusted for local cost of living.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Module B: Geographic and Skill Premium Explorer
# --------------------------------------------------------------------------- #
def module_b(art):
    agg = art["market_aggregates"]
    data = art["dashboard_data"]
    global_median = agg["global_median"]

    st.markdown(
        """
        <div class="fp-hero">
          <div class="fp-kicker">Module B &middot; Objectives 2 and 3</div>
          <h1>Geographic and Skill Premiums</h1>
          <p>Compare pay across the top hiring countries and skills, and judge whether a cross-border
          offer needs a location adjustment. The dashed line is the global median.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_geo, tab_skill, tab_remote = st.tabs(["Country pay", "Skill premiums", "Work arrangement"])

    with tab_geo:
        cm = agg["country_median"]
        countries = list(cm.keys())
        names = countries[::-1]
        vals = [cm[c] for c in names]
        colors = [EMERALD if v >= global_median else "#2a577f" for v in vals]
        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h", marker_color=colors,
            text=[usd_k(v) for v in vals], textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
            hovertemplate="%{y}<br>median %{x:$,.0f}<extra></extra>",
        ))
        fig.add_vline(x=global_median, line_dash="dash", line_color=GOLD,
                      annotation_text=f"global median {usd_k(global_median)}",
                      annotation_position="top")
        style_fig(fig, height=440)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("**Distribution for selected countries**")
        pick = st.multiselect("Countries to compare", countries,
                              default=countries[:3], key="geo_pick")
        if pick and "company_location" in data.columns:
            box = go.Figure()
            for c in pick:
                vals_c = data.loc[data["company_location"] == c, "salary_usd"].dropna()
                box.add_trace(go.Box(y=vals_c, name=c, marker_color=EMERALD,
                                     line_color=INK, boxmean=True))
            box.add_hline(y=global_median, line_dash="dash", line_color=GOLD)
            style_fig(box, height=380)
            box.update_layout(showlegend=False)
            st.plotly_chart(box, use_container_width=True, config={"displayModeBar": False})

        df_geo = pd.DataFrame({"country": list(cm.keys()), "median_salary_usd": list(cm.values())})
        st.download_button("Download country medians (CSV)",
                           df_geo.to_csv(index=False).encode("utf-8"),
                           "country_medians.csv", "text/csv")

    with tab_skill:
        sp = agg["skill_premium"]
        names = list(sp.keys())[::-1]
        vals = [sp[s] for s in names]
        colors = [EMERALD if v >= global_median else "#2a577f" for v in vals]
        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation="h", marker_color=colors,
            text=[usd_k(v) for v in vals], textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11),
            hovertemplate="%{y}<br>median %{x:$,.0f}<extra></extra>",
        ))
        fig.add_vline(x=global_median, line_dash="dash", line_color=GOLD,
                      annotation_text=f"global median {usd_k(global_median)}",
                      annotation_position="top")
        style_fig(fig, height=440)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Skill premiums sit in a narrow band, so individual skills move pay only modestly "
                   "compared with career level and location.")

        df_sk = pd.DataFrame({"skill": list(sp.keys()), "median_salary_usd": list(sp.values())})
        st.download_button("Download skill premiums (CSV)",
                           df_sk.to_csv(index=False).encode("utf-8"),
                           "skill_premiums.csv", "text/csv")

    with tab_remote:
        rm = agg["remote_median"]
        names = list(rm.keys())
        vals = [rm[k] for k in names]
        fig = go.Figure(go.Bar(
            x=names, y=vals, marker_color=[INK, EMERALD, STEEL],
            text=[usd_k(v) for v in vals], textposition="outside",
            textfont=dict(family="JetBrains Mono", size=13),
            hovertemplate="%{x}<br>median %{y:$,.0f}<extra></extra>",
        ))
        style_fig(fig, height=360)
        fig.update_yaxes(range=[0, max(vals) * 1.18])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        spread = (max(vals) - min(vals)) / min(vals) * 100
        st.markdown(
            f"""<div class="fp-note">On-site, hybrid, and fully remote roles differ by only
            {spread:.1f} percent in median pay. Work arrangement is not a meaningful salary driver in
            this data, so remote policy can be set on culture and retention grounds.</div>""",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Module C: Career Progression Ladder
# --------------------------------------------------------------------------- #
def module_c(art):
    agg = art["market_aggregates"]
    ladder = agg["career_ladder"]
    order = ["Entry", "Mid", "Senior", "Executive"]

    st.markdown(
        """
        <div class="fp-hero">
          <div class="fp-kicker">Module C &middot; Objective 1</div>
          <h1>Career Progression Ladder</h1>
          <p>Median pay at each stage, and the raise an organisation should expect at a promotion.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    vals = [ladder[k] for k in order]
    colors = [INK, "#1d3a5c", EMERALD, GOLD]
    fig = go.Figure(go.Bar(
        x=order, y=vals, marker_color=colors,
        text=[usd(v) for v in vals], textposition="outside",
        textfont=dict(family="JetBrains Mono", size=14),
        hovertemplate="%{x}<br>median %{y:$,.0f}<extra></extra>",
    ))
    for i in range(1, len(order)):
        jump = (vals[i] / vals[i - 1] - 1) * 100
        fig.add_annotation(x=i - 0.5, y=max(vals) * 1.02, text=f"+{jump:.0f}%",
                           showarrow=False, font=dict(family="JetBrains Mono",
                           size=12, color=EMERALD))
    style_fig(fig, height=420)
    fig.update_yaxes(range=[0, max(vals) * 1.18])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("**Promotion uplift calculator**")
    c1, c2 = st.columns(2)
    cur = c1.selectbox("Current level", order, index=1)
    tgt = c2.selectbox("Target level", order, index=2)
    if order.index(tgt) > order.index(cur):
        uplift = ladder[tgt] - ladder[cur]
        pct = (ladder[tgt] / ladder[cur] - 1) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{cur} median", usd(ladder[cur]))
        m2.metric(f"{tgt} median", usd(ladder[tgt]))
        m3.metric("Expected uplift", usd(uplift), f"{pct:+.0f}%")
    else:
        st.info("Pick a target level above the current level to see the uplift.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(
        page_title="FairPay Validator \u00b7 AI Salary Benchmarking",
        page_icon="\u2696\ufe0f",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    try:
        art = load_artefacts()
    except FileNotFoundError as exc:
        st.error("Some model artefacts are missing from the deployment_artefacts folder.")
        st.write("The app could not find these files:")
        for f in exc.args[0]:
            st.write(f"- `{f}`")
        st.info(
            "Run Section 6.1 of the notebook to write the nine .joblib files into "
            "`deployment_artefacts/`, then place that folder next to hr_dashboard.py and "
            "commit it to your repository."
        )
        st.stop()
    except Exception as exc:  # most often a scikit-learn version mismatch
        st.error("The artefacts were found but could not be loaded.")
        st.exception(exc)
        st.info(
            "This is usually a scikit-learn version mismatch. Pin scikit-learn in "
            "requirements.txt to the same version used to train the model in Colab. "
            "See the README for how to check it."
        )
        st.stop()

    metrics = art["market_aggregates"]["metrics"]

    # Top header bar with the model metrics (replaces the sidebar)
    st.markdown(
        f"""
        <div class="fp-topbar">
          <div>
            <div class="fp-brandname">FairPay Validator</div>
            <div class="fp-brandsub">AI talent compensation intelligence &middot; WQD7003 Group 12</div>
          </div>
          <div class="fp-chips">
            <span class="fp-chip"><span class="dot"></span> Gradient Boosting</span>
            <span class="fp-chip">R squared <b>{metrics['r2']:.2f}</b></span>
            <span class="fp-chip">MAE <b>{usd(metrics['mae'])}</b></span>
            <span class="fp-chip">MAPE <b>{metrics['mape']:.2f}%</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Module selection by top tabs
    tab_a, tab_b, tab_c = st.tabs([
        "A \u2014 Salary Estimator",
        "B \u00b7 Geography and Skills",
        "C \u00b7 Career Ladder",
    ])
    with tab_a:
        module_a(art)
    with tab_b:
        module_b(art)
    with tab_c:
        module_c(art)


if __name__ == "__main__":
    main()
