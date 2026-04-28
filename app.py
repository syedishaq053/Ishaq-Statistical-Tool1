"""
IshaKrishnan Statistical Analysis Suite
========================================
Multi-tool hub: after registration/access control, users land on a
tool-selector page with two cards:
  1. Statistical Analysis Tool  (the existing full stats engine)
  2. Sample Size Calculator     (embedded standalone HTML tool)

Phase flow:
  Phase 0 → Access wall  (non-IN users without active trial)
  Phase HUB → Tool selector (choose which tool to open)
  Phase STATS → The full stats app (existing logic, unchanged)
  Phase CALC  → Embedded sample size calculator HTML
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
import itertools

from scipy.stats import (
    shapiro, ttest_ind, ttest_rel, mannwhitneyu,
    wilcoxon, pearsonr, spearmanr, f_oneway, kruskal,
    chi2_contingency, friedmanchisquare,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
import streamlit.components.v1 as components
import requests
try:
    from supabase import create_client, Client
    _supabase_available = True
except ImportError:
    _supabase_available = False
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    layout="wide",
    page_title="IshaQ Stats Suite",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Supabase setup
supabase_url = ""
supabase_key = ""
try:
    supabase_url = st.secrets.get("SUPABASE_URL", "")
    supabase_key = st.secrets.get("SUPABASE_KEY", "")
except Exception:
    pass

supabase = None
if _supabase_available and supabase_url and supabase_key:
    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception:
        supabase = None


def get_country():
    try:
        return requests.get("https://ipapi.co/country/", timeout=4).text.strip()
    except Exception:
        return "Unknown"


if "user_country" not in st.session_state:
    st.session_state.user_country = get_country()


def is_trial_active(email):
    if supabase is None:
        return False
    try:
        result = (
            supabase.table("users").select("trial_start").eq("email", email).execute()
        )
        if not getattr(result, "data", None):
            return False
        trial_start = datetime.fromisoformat(
            result.data[0]["trial_start"].replace("Z", "+00:00")
        )
        return (datetime.now() - trial_start).days < 90
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# SESSION STATE — initialise all keys once
# ═══════════════════════════════════════════════════════════
_ss_defaults = {
    # navigation
    "active_tool": None,          # None → hub, "stats" → stats tool, "calc" → sample size
    "trial_active": False,
    # stats tool state
    "df_loaded": None,
    "guide_key": "default",
    "override_nominal": [],
    "override_ordinal": [],
    "analysis_done": False,
    "res_unified_df": None,
    "res_posthoc_df": None,
    "res_desc_df": None,
    "res_outlier_df": None,
    "res_pearson_corr": None,
    "res_spearman_corr": None,
    "res_corr_fig_bytes": None,
    "res_bp_figs": [],
    "res_extra_figs": [],
    "res_chi2_results": [],
    "res_friedman_results": [],
    "res_twoway_results": [],
    "res_mlr_results": [],
    "res_mlr_fig_bytes": None,
    "res_mlr_txt": "",
    "res_desc_fig_bytes": None,
    "res_col_subtype": {},
    "res_num_cols": [],
    "res_vars_desc": [],
    "res_corr_vars": [],
    "res_do_corr": True,
    "res_do_chi2": False,
    "res_do_friedman": False,
    "res_do_twoway": False,
    "res_do_mlr": False,
    "res_chi2_vars": None,
    "res_mlr_outcome": None,
    "res_mlr_predictors": [],
    "res_wide_mode": False,
    "res_group_cols_wide": [],
    "res_df_clean": None,
}
for _k, _v in _ss_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═══════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.markdown(
    """
<style>
  .main .block-container { max-width:100%!important; padding:1rem!important; }
  [data-testid="stSidebar"] { min-width:340px!important; max-width:420px!important; }
  .stButton>button { width:100%; border-radius:6px; font-weight:bold; }
  .stDownloadButton>button { width:100%; margin-top:3px; }
  .stTabs [data-baseweb="tab"] { padding:8px 14px; background:#f0f2f6; border-radius:5px; }
  div[data-testid="metric-container"] { background:#f8f9fa; border-radius:8px; padding:8px; }

  /* ── HUB CARDS ─────────────────────────────────────────── */
  .hub-wrapper {
    display: flex; gap: 28px; justify-content: center;
    flex-wrap: wrap; margin: 2.5rem 0;
  }
  .hub-card {
    width: 320px; border-radius: 18px; padding: 32px 28px 28px;
    border: 2px solid #e0e7ff; background: #f8faff;
    box-shadow: 0 4px 28px rgba(67,97,238,.09);
    display: flex; flex-direction: column; align-items: center;
    text-align: center; transition: transform .18s, box-shadow .18s;
    cursor: pointer;
  }
  .hub-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 40px rgba(67,97,238,.18);
  }
  .hub-card .hc-icon  { font-size: 3rem; margin-bottom: 14px; }
  .hub-card .hc-title {
    font-size: 1.25rem; font-weight: 800; color: #1e3a8a;
    margin-bottom: 8px; letter-spacing: -.3px;
  }
  .hub-card .hc-desc  { font-size: 0.88rem; color: #475569; line-height: 1.65; }
  .hub-card .hc-tags  { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 14px; }
  .hub-card .hc-tag   {
    background: #e0e7ff; color: #3730a3; border-radius: 20px;
    padding: 3px 11px; font-size: 0.75rem; font-weight: 600;
  }
  .hub-card .hc-btn {
    margin-top: 20px; background: #3730a3; color: #fff;
    border: none; border-radius: 9px; padding: 11px 28px;
    font-size: 0.92rem; font-weight: 700; cursor: pointer;
    width: 100%; letter-spacing: .3px;
    transition: background .15s;
  }
  /* Hub wrapper – flex container for cards */
.hub-wrapper {
  display: flex;
  justify-content: center;
  align-items: stretch;
  gap: 28px;
  flex-wrap: wrap;
  margin: 2.5rem 0;
}
/* Hub card style */
.hub-card {
  background: #f8faff;
  border: 2px solid #e0e7ff;
  border-radius: 18px;
  padding: 32px 28px 28px;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  transition: transform 0.18s, box-shadow 0.18s;
  height: 100%;
  width: 360px;
  max-width: 90%;
  margin: 0 auto;
}
.hub-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 40px rgba(67,97,238,.18);
}
.hc-icon { font-size: 3rem; margin-bottom: 14px; }
.hc-title { font-size: 1.25rem; font-weight: 800; color: #1e3a8a; margin-bottom: 8px; }
.hc-desc { font-size: 0.88rem; color: #475569; line-height: 1.65; margin-bottom: 16px; }
.hc-tags { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-top: 8px; }
.hc-tag {
  background: #e0e7ff;
  color: #3730a3;
  border-radius: 20px;
  padding: 3px 11px;
  font-size: 0.75rem;
  font-weight: 600;
}
/* Make columns (if used) same height; but here we use flex container */
.stButton button {
  margin-top: auto;
}
@media (max-width: 800px) {
  .hub-card {
    width: 100%;
  }
  .hub-wrapper {
    gap: 20px;
  }
}
  .hub-card .hc-btn:hover { background: #1e3a8a; }

  /* animated title */
  #anim-title {
    text-align:center; font-size:2rem; font-weight:900;
    font-family:'Segoe UI',sans-serif; letter-spacing:1px;
    margin-bottom:0.15rem; color:#1a1a2e;
  }
  .kq-span  { color:#e63946; }
  .title-sub {
    text-align:center; font-size:0.9rem; color:#666;
    margin-bottom:0.8rem; letter-spacing:0.4px;
  }

  /* guide / config boxes */
  .guide-box {
    background:#f0f4ff!important; color:#1a1a2e!important;
    border-left:4px solid #4361ee!important; border-radius:8px;
    padding:14px 16px; margin-bottom:10px;
    font-size:0.87rem; line-height:1.65;
  }
  .guide-box h4 { margin:0 0 6px 0; color:#1e3a8a!important; font-size:1rem; }
  .guide-box code { background:#e0e7ff!important; color:#1e293b!important;
    border-radius:3px; padding:1px 5px; font-size:0.82rem; }

  .config-card {
    background:#f0f4ff!important; border:1px solid #c7d2fe!important;
    border-radius:10px; padding:16px 18px; margin-bottom:12px; color:#1a1a2e!important;
  }
  .config-card h4 { margin:0 0 8px 0; color:#3730a3!important; font-size:1rem; }
  .config-tag      { background:#e0e7ff!important; color:#3730a3!important;
    border-radius:20px; padding:2px 10px; font-size:.78rem; font-weight:600; }
  .config-tag-green  { background:#d1fae5!important; color:#065f46!important; }
  .config-tag-yellow { background:#fef9c3!important; color:#713f12!important; }
  .config-tag-red    { background:#fee2e2!important; color:#991b1b!important; }

  .badge-ordinal { background:#fff3cd; color:#856404; border-radius:4px; padding:2px 7px; font-size:.78rem; font-weight:600; }
  .badge-nominal { background:#d1e7dd; color:#0f5132; border-radius:4px; padding:2px 7px; font-size:.78rem; font-weight:600; }

  .linkedin-bar {
    text-align:center; padding:9px; background:#f0f4ff;
    border-radius:8px; margin-top:16px; font-size:0.84rem;
  }
  .linkedin-bar a { color:#0a66c2; font-weight:bold; text-decoration:none; }
  .linkedin-bar a:hover { text-decoration:underline; }

  /* back button */
  .back-btn-wrap { margin-bottom: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════
# Google Analytics
# ═══════════════════════════════════════════════════════════
ga_code = """
<script async src="https://www.googletagmanager.com/gtag/js?id=G-4L6B7JX151"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-4L6B7JX151');
</script>
"""
components.html(ga_code, height=0)

# ═══════════════════════════════════════════════════════════
# ACCESS CONTROL
# ═══════════════════════════════════════════════════════════
_country = st.session_state.user_country
_trial   = st.session_state.get("trial_active", False)

if _country != "IN" and not _trial:
    st.markdown(
        """
<div style="max-width:480px;margin:60px auto;padding:32px 36px;
     background:#f8faff;border:1px solid #c7d2fe;border-radius:14px;
     box-shadow:0 4px 24px rgba(67,97,238,.10);">
  <h2 style="text-align:center;color:#3730a3;margin-bottom:4px;">
    📊 IshaQ‑Krishnan Stats Suite
  </h2>
  <p style="text-align:center;color:#666;font-size:0.9rem;margin-bottom:20px;">
    Comprehensive statistical analysis · Free 90‑day trial for new users
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.info("🌍 Enter your email below to start your **free 90‑day trial**.")
    reg_email = st.text_input(
        "Email address", placeholder="you@example.com", key="reg_email"
    )

    if st.button("🚀 Start Free Trial", type="primary"):
        if not reg_email or "@" not in reg_email:
            st.warning("Please enter a valid email address.")
        elif supabase is None:
            st.success("✅ Trial started (local mode).")
            st.session_state["trial_active"] = True
            st.rerun()
        else:
            try:
                existing = (
                    supabase.table("users")
                    .select("trial_start,email")
                    .eq("email", reg_email)
                    .execute()
                )
                if not getattr(existing, "data", None):
                    supabase.table("users").insert(
                        {"email": reg_email, "country": _country}
                    ).execute()
                    st.success("✅ Trial started! You have 90 days of free access.")
                    st.session_state["trial_active"] = True
                    st.rerun()
                else:
                    trial_start = datetime.fromisoformat(
                        existing.data[0]["trial_start"].replace("Z", "+00:00")
                    )
                    days_used = (datetime.now().astimezone() - trial_start).days
                    if days_used < 90:
                        st.success(
                            f"✅ Welcome back! You have **{90 - days_used} days** left."
                        )
                        st.session_state["trial_active"] = True
                        st.rerun()
                    else:
                        st.error(
                            "❌ Your 90‑day trial has ended. "
                            "Please subscribe to continue."
                        )
            except Exception as e:
                st.error(f"Registration failed: {e}")
    st.stop()

elif _country == "IN":
    st.success("🇮🇳 Free access for India")

# ═══════════════════════════════════════════════════════════
# ANIMATED TITLE (shared across hub and both tools)
# ═══════════════════════════════════════════════════════════
st.markdown(
    """
<div id="anim-title">
  Isha<span id="kq-span" class="kq-span">Q</span>rishnan
  &nbsp;Statistical Analysis Suite
</div>
<div class="title-sub">
  Comprehensive · Parametric &amp; Non-parametric · Wide &amp; Long format
</div>
""",
    unsafe_allow_html=True,
)
components.html(
    """
<script>
(function(){
  var forms = ['Q','K'];
  var i = 0;
  function tick(){
    var el = window.parent.document.getElementById('kq-span');
    if(el){ i=(i+1)%2; el.textContent=forms[i]; }
  }
  setInterval(tick, 1000);
})();
</script>
""",
    height=0,
)

# ═══════════════════════════════════════════════════════════
# ══════════  HUB — TOOL SELECTOR  ══════════════════════════
# ═══════════════════════════════════════════════════════════
if st.session_state["active_tool"] is None:

    st.markdown(
        """
<h2 style="text-align:center;margin:1.5rem 0 0.3rem;font-size:1.5rem;
   font-weight:800;color:#1e3a8a;letter-spacing:-.4px;">
  Choose your tool
</h2>
<p style="text-align:center;color:#64748b;font-size:0.93rem;margin-bottom:0;">
  Both tools are fully free for Indian users · 90-day trial for all others
</p>
""",
        unsafe_allow_html=True,
    )

    # Use a simple flex wrapper – no Streamlit columns
    st.markdown('<div class="hub-wrapper">', unsafe_allow_html=True)

    # --- Left card (Statistical Analysis Tool) ---
    st.markdown(
        """
<div class="hub-card">
  <div class="hc-icon">🔬</div>
  <div class="hc-title">Statistical Analysis Tool</div>
  <div class="hc-desc">
    Full-featured parametric &amp; non-parametric tests.
    Upload your Excel or CSV data and get instant results
    with effect sizes, post-hoc tests, plots, and downloads.
  </div>
  <div class="hc-tags">
    <span class="hc-tag">t-test</span>
    <span class="hc-tag">ANOVA</span>
    <span class="hc-tag">Mann-Whitney</span>
    <span class="hc-tag">Kruskal-Wallis</span>
    <span class="hc-tag">Chi-square</span>
    <span class="hc-tag">Regression</span>
    <span class="hc-tag">Correlation</span>
    <span class="hc-tag">8 Plot types</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    # Button (outside the HTML so it works with Streamlit)
    if st.button("🔬 Open Statistical Analysis Tool", key="goto_stats", type="primary", use_container_width=True):
        st.session_state["active_tool"] = "stats"
        st.rerun()

    # --- Right card (Sample Size Calculator) ---
    st.markdown(
        """
<div class="hub-card">
  <div class="hc-icon">📐</div>
  <div class="hc-title">Sample Size Calculator</div>
  <div class="hc-desc">
    Upload your data or enter summary statistics.
    The tool auto-detects the right test, computes effect size,
    and tells you exactly how many observations you need —
    with a colour-coded sufficiency verdict.
  </div>
  <div class="hc-tags">
    <span class="hc-tag">Independent t-test</span>
    <span class="hc-tag">Paired t-test</span>
    <span class="hc-tag">One-way ANOVA</span>
    <span class="hc-tag">Correlation</span>
    <span class="hc-tag">Two proportions</span>
    <span class="hc-tag">Power curve</span>
    <span class="hc-tag">Cohen's d / f / h</span>
    <span class="hc-tag">Excel / CSV upload</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("📐 Open Sample Size Calculator", key="goto_calc", type="primary", use_container_width=True):
        st.session_state["active_tool"] = "calc"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)   # close hub-wrapper

    # Comparison table
    st.markdown("---")
    st.markdown(
        """
<h3 style="text-align:center;font-size:1.1rem;color:#1e3a8a;margin-bottom:.7rem;">
  Quick comparison
</h3>
""",
        unsafe_allow_html=True,
    )
    comp_df = pd.DataFrame(
        {
            "Feature": [
                "Upload Excel / CSV",
                "Automatic test selection",
                "Effect sizes",
                "Post-hoc pairwise tests",
                "Correlation heatmaps",
                "8 extra plot types",
                "Required sample size (N)",
                "Power curve chart",
                "Green / Orange / Red verdict",
                "MLR & Two-way ANOVA",
            ],
            "🔬 Stats Tool": ["✅", "✅", "✅", "✅", "✅", "✅", "—", "—", "—", "✅"],
            "📐 Sample Size Calc": ["✅", "✅", "✅", "—", "—", "—", "✅", "✅", "✅", "—"],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
<div class="linkedin-bar" style="margin-top:20px;">
  ✉️ Contact / Collaborate:&nbsp;
  <a href="https://www.linkedin.com/in/syed-ishaq-893052285/" target="_blank">
    🔗 Syed Ishaq on LinkedIn
  </a>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()


# ═══════════════════════════════════════════════════════════
# BACK-TO-HUB BUTTON (shown at the top of both tools)
# ═══════════════════════════════════════════════════════════
def render_back_button():
    if st.button("← Back to tool selector", key="back_hub"):
        st.session_state["active_tool"] = None
        st.rerun()


# ═══════════════════════════════════════════════════════════
# ══════════  SAMPLE SIZE CALCULATOR TOOL  ══════════════════
# ═══════════════════════════════════════════════════════════
if st.session_state["active_tool"] == "calc":
    render_back_button()
    st.markdown(
        "<h2 style='text-align:center;color:#1e3a8a;margin-bottom:.3rem;'>"
        "📐 Sample Size Calculator</h2>",
        unsafe_allow_html=True,
    )

    # Load the external HTML file; if present render and stop.
    try:
        with open("samplesize.html", "r", encoding="utf-8") as f:
            calc_html = f.read()
        components.html(calc_html, height=1200, scrolling=True)
        st.stop()
    except FileNotFoundError:
        st.error("Calculator file 'samplesize.html' not found — rendering embedded fallback.")

    CALC_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sample Size Calculator</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'DM Sans',sans-serif;background:#f8faff;color:#1e293b;font-size:15px;line-height:1.6;}
  .app{max-width:880px;margin:0 auto;padding:1.5rem 1.25rem 4rem;}
  .section{margin-bottom:1.5rem;}
  .section-label{font-family:'DM Mono',monospace;font-size:10px;letter-spacing:.1em;color:#94a3b8;text-transform:uppercase;margin-bottom:.75rem;display:flex;align-items:center;gap:8px;}
  .section-label::after{content:'';flex:1;height:1px;background:#e2e8f0;}
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:1.375rem;}
  label{font-size:12px;color:#64748b;display:block;margin-bottom:5px;font-weight:500;letter-spacing:.02em;}
  select,input[type=number],input[type=text]{width:100%;background:#f8faff;border:1px solid #e2e8f0;border-radius:8px;padding:9px 12px;font-size:14px;font-family:'DM Sans',sans-serif;color:#1e293b;transition:border-color .15s;-webkit-appearance:none;appearance:none;}
  select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2394a3b8' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px;cursor:pointer;}
  select:focus,input:focus{outline:none;border-color:#6366f1;}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
  .grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
  .slider-wrap{display:flex;align-items:center;gap:10px;}
  input[type=range]{flex:1;-webkit-appearance:none;appearance:none;height:4px;background:#e2e8f0;border-radius:2px;outline:none;cursor:pointer;}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#6366f1;cursor:pointer;}
  .slider-val{font-family:'DM Mono',monospace;font-size:13px;color:#6366f1;min-width:38px;text-align:right;}
  .btn-run{width:100%;background:#3730a3;color:#fff;border:none;border-radius:9px;padding:12px 20px;font-size:14px;font-weight:700;font-family:'DM Sans',sans-serif;cursor:pointer;letter-spacing:.02em;transition:all .2s;margin-top:1.25rem;}
  .btn-run:hover{background:#1e3a8a;transform:translateY(-1px);}
  .upload-zone{border:1.5px dashed #c7d2fe;border-radius:14px;padding:2.25rem 2rem;text-align:center;cursor:pointer;transition:all .2s;}
  .upload-zone:hover,.upload-zone.drag-over{border-color:#6366f1;background:#eef2ff;}
  .upload-zone.has-file{border-style:solid;border-color:#a5b4fc;background:#eef2ff;}
  .upload-icon{font-size:2.25rem;margin-bottom:.75rem;}
  .upload-title{font-size:15px;font-weight:600;color:#1e293b;margin-bottom:4px;}
  .upload-sub{font-size:13px;color:#94a3b8;}
  .preview-wrap{overflow-x:auto;margin-top:1rem;border:1px solid #e2e8f0;border-radius:8px;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  thead th{padding:7px 11px;background:#f1f5f9;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#64748b;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap;}
  tbody td{padding:7px 11px;border-bottom:1px solid #f1f5f9;color:#475569;}
  tbody tr:last-child td{border-bottom:none;}
  .status-bar{border-radius:12px;padding:1.125rem 1.375rem;display:flex;align-items:flex-start;gap:14px;margin-bottom:1.25rem;border:1.5px solid;}
  .status-bar.green{background:#f0fdf4;border-color:#bbf7d0;}
  .status-bar.orange{background:#fffbeb;border-color:#fed7aa;}
  .status-bar.red{background:#fef2f2;border-color:#fecaca;}
  .status-dot{width:10px;height:10px;border-radius:50%;margin-top:5px;flex-shrink:0;}
  .green .status-dot{background:#22c55e;}
  .orange .status-dot{background:#f59e0b;}
  .red .status-dot{background:#ef4444;}
  .status-title{font-size:15px;font-weight:700;margin-bottom:2px;}
  .green .status-title{color:#15803d;}
  .orange .status-title{color:#b45309;}
  .red .status-title{color:#b91c1c;}
  .status-desc{font-size:13px;color:#64748b;}
  .metric-card{background:#f8faff;border:1px solid #e2e8f0;border-radius:10px;padding:1rem 1.125rem;}
  .metric-label{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px;}
  .metric-value{font-size:2rem;font-weight:700;color:#1e293b;line-height:1;margin-bottom:3px;}
  .metric-sub{font-size:11px;color:#94a3b8;}
  .power-bar-outer{height:7px;background:#e2e8f0;border-radius:4px;overflow:hidden;margin-top:8px;}
  .power-bar-inner{height:100%;border-radius:4px;transition:width .6s cubic-bezier(.34,1.56,.64,1);}
  .tabs{display:flex;gap:2px;margin-bottom:1.25rem;border-bottom:1.5px solid #e2e8f0;}
  .tab-btn{font-family:'DM Sans',sans-serif;font-size:13px;padding:8px 16px;background:none;border:none;color:#94a3b8;cursor:pointer;border-bottom:2.5px solid transparent;margin-bottom:-1.5px;transition:all .15s;border-radius:0;}
  .tab-btn.active{color:#3730a3;border-bottom-color:#3730a3;font-weight:600;}
  .chart-wrap{position:relative;width:100%;height:290px;}
  .interpret-box{background:#f0f4ff;border:1px solid #e0e7ff;border-left:4px solid #6366f1;border-radius:0 10px 10px 0;padding:.875rem 1.125rem;font-size:13px;color:#475569;margin-top:1rem;line-height:1.6;}
  .formula{font-family:'DM Mono',monospace;font-size:12px;color:#4f46e5;background:#eef2ff;border-radius:8px;padding:.6rem 1rem;margin-top:.75rem;letter-spacing:.03em;}
  .divider{border:none;border-top:1px solid #e2e8f0;margin:1.25rem 0;}
  .detail-table thead th{background:#f1f5f9;}
  .detail-table .highlight td{background:#eef2ff;}
  .detail-table .highlight td:first-child{color:#4f46e5;}
  .detail-table .highlight td:last-child{color:#4f46e5;font-weight:600;}
  @media(max-width:640px){.grid-2,.grid-4{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="app">

<div class="section">
  <div class="section-label">01 — Data source</div>
  <div class="card">
    <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
      <div class="upload-icon">📊</div>
      <div class="upload-title" id="uploadTitle">Drop your file here, or <span style="color:#6366f1">browse</span></div>
      <div class="upload-sub">Supports .xlsx · .xls · .csv</div>
    </div>
    <input type="file" id="fileInput" accept=".xlsx,.xls,.csv" style="display:none">
    <div id="previewSection" style="display:none">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:1rem;margin-bottom:.5rem;">
        <span style="font-size:13px;color:#64748b;" id="fileInfoText"></span>
      </div>
      <div class="preview-wrap" id="previewWrap"></div>
    </div>
  </div>
</div>

<div class="section" id="configSection" style="display:none">
  <div class="section-label">02 — Test configuration</div>
  <div class="card">
    <div style="margin-bottom:1.25rem;">
      <label>Statistical test</label>
      <select id="testType" onchange="onTestChange()">
        <option value="independent_t">Independent samples t-test (2 groups)</option>
        <option value="paired_t">Paired samples t-test (pre/post)</option>
        <option value="one_way_anova">One-way ANOVA (3+ groups)</option>
        <option value="correlation">Pearson correlation</option>
        <option value="two_prop">Two proportions z-test</option>
        <option value="one_sample_t">One-sample t-test (vs reference)</option>
      </select>
    </div>
    <div id="colSelectors"></div>
    <hr class="divider">
    <div class="grid-2" style="margin-bottom:1.25rem;">
      <div>
        <label>Effect size source</label>
        <select id="effectSource" onchange="onEffectSourceChange()">
          <option value="from_data">Compute from data (recommended)</option>
          <option value="manual">Enter manually</option>
          <option value="conventional">Cohen's conventional sizes</option>
        </select>
      </div>
      <div id="manualEffectDiv" style="display:none">
        <label>Effect size value</label>
        <input type="number" id="manualEffect" value="0.5" step="0.01" min="0.001" max="5">
      </div>
      <div id="conventionalDiv" style="display:none">
        <label>Convention</label>
        <select id="conventionSel">
          <option value="small">Small</option>
          <option value="medium" selected>Medium</option>
          <option value="large">Large</option>
        </select>
      </div>
    </div>
    <div class="grid-2" style="margin-bottom:1.25rem;">
      <div>
        <label>Significance level α — <span id="alphaDisplay" style="color:#6366f1;font-family:'DM Mono',monospace;">0.05</span></label>
        <div class="slider-wrap">
          <input type="range" id="alphaSlider" min="0.01" max="0.20" step="0.01" value="0.05"
            oninput="document.getElementById('alphaDisplay').textContent=parseFloat(this.value).toFixed(2)">
          <span class="slider-val" id="alphaVal">0.05</span>
        </div>
      </div>
      <div>
        <label>Desired power (1−β) — <span id="powerDisplay" style="color:#6366f1;font-family:'DM Mono',monospace;">0.80</span></label>
        <div class="slider-wrap">
          <input type="range" id="powerSlider" min="0.50" max="0.99" step="0.01" value="0.80"
            oninput="document.getElementById('powerDisplay').textContent=parseFloat(this.value).toFixed(2)">
          <span class="slider-val" id="powerVal">0.80</span>
        </div>
      </div>
    </div>
    <div class="grid-2" style="margin-bottom:.5rem;">
      <div>
        <label>Hypothesis tails</label>
        <select id="tailsSel">
          <option value="2">Two-tailed</option>
          <option value="1">One-tailed</option>
        </select>
      </div>
      <div id="muRefDiv" style="display:none">
        <label>Reference mean (μ₀)</label>
        <input type="number" id="muRef" value="0" step="0.01">
      </div>
    </div>
    <button class="btn-run" id="calcBtn" onclick="calculate()">⚡ Run power analysis</button>
  </div>
</div>

<div class="section" id="resultSection" style="display:none">
  <div class="section-label">03 — Results</div>
  <div id="statusBar"></div>
  <div class="card">
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('summary',this)">Summary</button>
      <button class="tab-btn" onclick="switchTab('power',this)">Power curve</button>
      <button class="tab-btn" onclick="switchTab('detail',this)">Full output</button>
    </div>
    <div id="tabSummary">
      <div class="grid-4" id="metricGrid" style="margin-bottom:1.25rem;"></div>
      <div id="powerBarSection"></div>
    </div>
    <div id="tabPower" style="display:none">
      <div id="chartLegend" style="display:flex;gap:20px;margin-bottom:1rem;flex-wrap:wrap;"></div>
      <div class="chart-wrap"><canvas id="powerChart" role="img" aria-label="Power curve"></canvas></div>
    </div>
    <div id="tabDetail" style="display:none;overflow-x:auto;" class="detail-table">
      <div class="preview-wrap" id="detailTableWrap"></div>
    </div>
  </div>
  <div id="formulaBox"></div>
  <div class="interpret-box" id="interpretBox"></div>
</div>
</div>

<script>
let parsedData=[],columns=[],chartInstance=null;
const fileInput=document.getElementById('fileInput');
const uploadZone=document.getElementById('uploadZone');
uploadZone.addEventListener('dragover',e=>{e.preventDefault();uploadZone.classList.add('drag-over');});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop',e=>{e.preventDefault();uploadZone.classList.remove('drag-over');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0]);});
fileInput.addEventListener('change',()=>{if(fileInput.files[0])handleFile(fileInput.files[0]);});
function handleFile(file){
  const reader=new FileReader();
  reader.onload=function(e){
    let wb;
    if(file.name.toLowerCase().endsWith('.csv')){const txt=new TextDecoder().decode(e.target.result);wb=XLSX.read(txt,{type:'string'});}
    else{wb=XLSX.read(e.target.result,{type:'array'});}
    const ws=wb.Sheets[wb.SheetNames[0]];
    const data=XLSX.utils.sheet_to_json(ws,{defval:null});
    if(!data||data.length===0){alert('No data found.');return;}
    parsedData=data;columns=Object.keys(data[0]);
    document.getElementById('uploadTitle').textContent='✓  '+file.name;
    document.getElementById('fileInfoText').textContent=`${data.length} rows · ${columns.length} columns`;
    uploadZone.classList.add('has-file');
    renderPreview(data,columns);
    document.getElementById('previewSection').style.display='';
    document.getElementById('configSection').style.display='';
    document.getElementById('resultSection').style.display='none';
    buildColSelectors();
  };
  reader.readAsArrayBuffer(file);
}
function renderPreview(data,cols){
  const rows=data.slice(0,5);
  let html=`<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(r=>{html+=`<tr>${cols.map(c=>`<td>${r[c]===null||r[c]===undefined?'—':r[c]}</td>`).join('')}</tr>`;});
  html+=`</tbody></table>`;
  document.getElementById('previewWrap').innerHTML=html;
}
function getNumericCols(){return columns.filter(c=>parsedData.some(r=>r[c]!==null&&r[c]!==''&&!isNaN(Number(r[c]))));}
function getCategoricalCols(){return columns.filter(c=>{const v=[...new Set(parsedData.map(r=>r[c]).filter(v=>v!==null&&v!==''))];return v.length>=2&&v.length<=30;});}
function buildColSelectors(){
  const test=document.getElementById('testType').value;
  const num=getNumericCols(),cat=getCategoricalCols();
  const numO=num.map(c=>`<option value="${c}">${c}</option>`).join('');
  const catO=cat.map(c=>`<option value="${c}">${c}</option>`).join('');
  const numO2=num.map((c,i)=>`<option value="${c}"${i===1?' selected':''}>${c}</option>`).join('');
  let html='';
  if(test==='independent_t'||test==='one_way_anova'){
    html=`<div class="grid-2" style="margin-bottom:1rem;"><div><label>Outcome (numeric)</label><select id="outcomeCol">${numO}</select></div><div><label>Grouping (categorical)</label><select id="groupCol">${catO}</select></div></div>`;
  }else if(test==='paired_t'){
    html=`<div class="grid-2" style="margin-bottom:1rem;"><div><label>Pre / Condition 1</label><select id="col1">${numO}</select></div><div><label>Post / Condition 2</label><select id="col2">${numO2}</select></div></div>`;
  }else if(test==='correlation'){
    html=`<div class="grid-2" style="margin-bottom:1rem;"><div><label>Variable X</label><select id="col1">${numO}</select></div><div><label>Variable Y</label><select id="col2">${numO2}</select></div></div>`;
  }else if(test==='two_prop'){
    html=`<div class="grid-2" style="margin-bottom:1rem;"><div><label>Binary outcome (0/1, yes/no)</label><select id="outcomeCol">${columns.map(c=>`<option>${c}</option>`).join('')}</select></div><div><label>Grouping (exactly 2 groups)</label><select id="groupCol">${catO}</select></div></div>`;
  }else if(test==='one_sample_t'){
    html=`<div style="margin-bottom:1rem;"><label>Variable (numeric)</label><select id="outcomeCol">${numO}</select></div>`;
    document.getElementById('muRefDiv').style.display='';
  }
  if(test!=='one_sample_t')document.getElementById('muRefDiv').style.display='none';
  document.getElementById('colSelectors').innerHTML=html;
}
function onTestChange(){buildColSelectors();}
function onEffectSourceChange(){
  const v=document.getElementById('effectSource').value;
  document.getElementById('manualEffectDiv').style.display=v==='manual'?'':'none';
  document.getElementById('conventionalDiv').style.display=v==='conventional'?'':'none';
}
document.getElementById('alphaSlider').addEventListener('input',function(){document.getElementById('alphaDisplay').textContent=parseFloat(this.value).toFixed(2);document.getElementById('alphaVal').textContent=parseFloat(this.value).toFixed(2);});
document.getElementById('powerSlider').addEventListener('input',function(){document.getElementById('powerDisplay').textContent=parseFloat(this.value).toFixed(2);document.getElementById('powerVal').textContent=parseFloat(this.value).toFixed(2);});
function normalCDF(z){const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;const sign=z<0?-1:1;z=Math.abs(z)/Math.sqrt(2);const t=1/(1+p*z);const y=1-(((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t)*Math.exp(-z*z);return 0.5*(1+sign*y);}
function invNorm(p){let lo=-10,hi=10,m=0;for(let i=0;i<80;i++){m=(lo+hi)/2;normalCDF(m)<p?lo=m:hi=m;}return m;}
function mean(a){return a.reduce((s,v)=>s+v,0)/a.length;}
function variance(a){const m=mean(a);return a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1);}
function std(a){return Math.sqrt(variance(a));}
function pearsonR(x,y){const mx=mean(x),my=mean(y);let num=0,dx2=0,dy2=0;for(let i=0;i<x.length;i++){num+=(x[i]-mx)*(y[i]-my);dx2+=(x[i]-mx)**2;dy2+=(y[i]-my)**2;}return num/Math.sqrt(dx2*dy2);}
function computePower(n,es,alpha,tails,test){
  const za=invNorm(1-alpha/tails);let ncp;
  if(test==='independent_t')ncp=es*Math.sqrt(n/2);
  else if(test==='paired_t'||test==='one_sample_t')ncp=es*Math.sqrt(n);
  else if(test==='one_way_anova')ncp=es*Math.sqrt(n);
  else if(test==='correlation'){const rz=0.5*Math.log((1+es)/(1-Math.min(es,0.9999)));ncp=Math.abs(rz)*Math.sqrt(n-3);}
  else if(test==='two_prop')ncp=es*Math.sqrt(n);
  else ncp=es*Math.sqrt(n/2);
  return 1-normalCDF(za-ncp)+normalCDF(-za-ncp);
}
function solveN(es,alpha,power,tails,test){
  let lo=2,hi=1000000;
  for(let i=0;i<80;i++){const mid=Math.ceil((lo+hi)/2);computePower(mid,es,alpha,tails,test)>=power?hi=mid:lo=mid;if(hi-lo<=1)break;}
  return hi;
}
function getConventional(test,size){
  const map={independent_t:{small:.20,medium:.50,large:.80},paired_t:{small:.20,medium:.50,large:.80},one_sample_t:{small:.20,medium:.50,large:.80},one_way_anova:{small:.10,medium:.25,large:.40},correlation:{small:.10,medium:.30,large:.50},two_prop:{small:.20,medium:.50,large:.80}};
  return(map[test]||map.independent_t)[size];
}
function getActualN(test){
  try{
    if(test==='independent_t'||test==='one_way_anova'||test==='two_prop'){const oC=document.getElementById('outcomeCol')?.value;return parsedData.filter(r=>r[oC]!==null&&r[oC]!==''&&!isNaN(Number(r[oC]))).length;}
    else if(test==='paired_t'||test==='correlation'){const c1=document.getElementById('col1')?.value,c2=document.getElementById('col2')?.value;return parsedData.filter(r=>r[c1]!==null&&!isNaN(Number(r[c1]))&&r[c2]!==null&&!isNaN(Number(r[c2]))).length;}
    else if(test==='one_sample_t'){const c=document.getElementById('outcomeCol')?.value;return parsedData.filter(r=>r[c]!==null&&!isNaN(Number(r[c]))).length;}
  }catch(e){return parsedData.length;}
  return parsedData.length;
}
function computeEffectFromData(test){
  if(test==='independent_t'||test==='one_way_anova'){
    const oC=document.getElementById('outcomeCol').value,gC=document.getElementById('groupCol').value;
    const rows=parsedData.filter(r=>r[oC]!==null&&r[oC]!==''&&!isNaN(Number(r[oC]))&&r[gC]!==null&&r[gC]!=='');
    if(!rows.length)throw new Error('No valid rows.');
    const grps={};rows.forEach(r=>{const g=String(r[gC]);if(!grps[g])grps[g]=[];grps[g].push(Number(r[oC]));});
    const gNames=Object.keys(grps);if(gNames.length<2)throw new Error('Need ≥2 groups.');
    if(test==='independent_t'&&gNames.length===2){
      const a=grps[gNames[0]],b=grps[gNames[1]];
      const pooled=Math.sqrt(((a.length-1)*variance(a)+(b.length-1)*variance(b))/(a.length+b.length-2));
      const d=Math.abs(mean(a)-mean(b))/pooled;
      return{effect:d,label:"Cohen's d",formula:`d=|μ₁−μ₂|/SD_pooled=${d.toFixed(3)}`,n:rows.length,groups:gNames,detail:{[`Group ${gNames[0]}`]:`n=${a.length}, mean=${mean(a).toFixed(3)}, SD=${std(a).toFixed(3)}`,[`Group ${gNames[1]}`]:`n=${b.length}, mean=${mean(b).toFixed(3)}, SD=${std(b).toFixed(3)}`,'Pooled SD':pooled.toFixed(4),"Cohen's d":d.toFixed(4),'Magnitude':d<0.2?'negligible':d<0.5?'small':d<0.8?'medium':'large'}};
    }else{
      const all=rows.map(r=>Number(r[oC])),gm=mean(all);
      const ssB=gNames.reduce((s,g)=>s+grps[g].length*(mean(grps[g])-gm)**2,0);
      const ssT=all.reduce((s,v)=>s+(v-gm)**2,0);
      const eta2=ssB/ssT,f=Math.sqrt(eta2/(1-eta2));
      const det={};gNames.forEach(g=>{det[`Group ${g}`]=`n=${grps[g].length}, mean=${mean(grps[g]).toFixed(3)}, SD=${std(grps[g]).toFixed(3)}`;});
      det['η²']=eta2.toFixed(4);det["Cohen's f"]=f.toFixed(4);det['Magnitude']=f<0.1?'small':f<0.25?'medium':'large';
      return{effect:f,label:"Cohen's f",formula:`f=√(η²/(1−η²))=${f.toFixed(3)}`,n:rows.length,groups:gNames,detail:det};
    }
  }
  if(test==='paired_t'){
    const c1=document.getElementById('col1').value,c2=document.getElementById('col2').value;
    const rows=parsedData.filter(r=>r[c1]!==null&&!isNaN(Number(r[c1]))&&r[c2]!==null&&!isNaN(Number(r[c2])));
    if(!rows.length)throw new Error('No valid paired rows.');
    const diffs=rows.map(r=>Number(r[c1])-Number(r[c2]));
    const md=mean(diffs),sd=std(diffs),dz=Math.abs(md)/sd;
    return{effect:dz,label:"Cohen's dz",formula:`dz=|mean(diff)|/SD(diff)=${dz.toFixed(3)}`,n:rows.length,detail:{'Mean difference':md.toFixed(4),'SD of differences':sd.toFixed(4),"Cohen's dz":dz.toFixed(4),'Magnitude':dz<0.2?'small':dz<0.5?'medium':'large'}};
  }
  if(test==='correlation'){
    const c1=document.getElementById('col1').value,c2=document.getElementById('col2').value;
    const rows=parsedData.filter(r=>r[c1]!==null&&!isNaN(Number(r[c1]))&&r[c2]!==null&&!isNaN(Number(r[c2])));
    if(!rows.length)throw new Error('No valid rows.');
    const x=rows.map(r=>Number(r[c1])),y=rows.map(r=>Number(r[c2]));
    const r=pearsonR(x,y);
    return{effect:Math.abs(r),label:'Pearson r',formula:`r=${r.toFixed(4)}, |r|=${Math.abs(r).toFixed(4)}`,n:rows.length,detail:{'Pearson r':r.toFixed(4),'|r|':Math.abs(r).toFixed(4),'r²':(r*r).toFixed(4),'Magnitude':Math.abs(r)<0.1?'negligible':Math.abs(r)<0.3?'small':Math.abs(r)<0.5?'medium':'large'}};
  }
  if(test==='two_prop'){
    const oC=document.getElementById('outcomeCol').value,gC=document.getElementById('groupCol').value;
    const rows=parsedData.filter(r=>r[gC]!==null&&r[gC]!=='');
    const grps={};rows.forEach(r=>{const g=String(r[gC]);if(!grps[g])grps[g]=[];const v=String(r[oC]).toLowerCase().trim();grps[g].push(v==='1'||v==='yes'||v==='true'?1:0);});
    const gNames=Object.keys(grps);if(gNames.length!==2)throw new Error('Need exactly 2 groups.');
    const p1=mean(grps[gNames[0]]),p2=mean(grps[gNames[1]]);
    const h=2*(Math.asin(Math.sqrt(Math.max(0,Math.min(1,p1))))-Math.asin(Math.sqrt(Math.max(0,Math.min(1,p2)))));
    return{effect:Math.abs(h),label:"Cohen's h",formula:`h=2·(arcsin(√p₁)−arcsin(√p₂))=${Math.abs(h).toFixed(3)}`,n:rows.length,groups:gNames,detail:{[gNames[0]+' prop']:p1.toFixed(4),[gNames[1]+' prop']:p2.toFixed(4),[gNames[0]+' n']:grps[gNames[0]].length,[gNames[1]+' n']:grps[gNames[1]].length,"Cohen's h":Math.abs(h).toFixed(4),'Magnitude':Math.abs(h)<0.2?'small':Math.abs(h)<0.5?'medium':'large'}};
  }
  if(test==='one_sample_t'){
    const oC=document.getElementById('outcomeCol').value;
    const mu0=parseFloat(document.getElementById('muRef').value)||0;
    const vals=parsedData.filter(r=>r[oC]!==null&&!isNaN(Number(r[oC]))).map(r=>Number(r[oC]));
    if(!vals.length)throw new Error('No valid values.');
    const m=mean(vals),s=std(vals),d=Math.abs(m-mu0)/s;
    return{effect:d,label:"Cohen's d",formula:`d=|x̄−μ₀|/SD=${d.toFixed(3)}`,n:vals.length,detail:{'Sample mean':m.toFixed(4),'Reference μ₀':mu0,'Sample SD':s.toFixed(4),"Cohen's d":d.toFixed(4),'Magnitude':d<0.2?'small':d<0.5?'medium':'large'}};
  }
  throw new Error('Unknown test.');
}
function calculate(){
  const test=document.getElementById('testType').value;
  const alpha=parseFloat(document.getElementById('alphaSlider').value);
  const power=parseFloat(document.getElementById('powerSlider').value);
  const tails=parseInt(document.getElementById('tailsSel').value);
  const src=document.getElementById('effectSource').value;
  let res;
  try{
    if(src==='from_data')res=computeEffectFromData(test);
    else if(src==='manual'){const es=parseFloat(document.getElementById('manualEffect').value);if(isNaN(es)||es<=0)throw new Error('Effect size must be positive.');res={effect:es,label:'Manual',formula:`User-specified effect size=${es}`,n:getActualN(test),detail:{'Effect size (manual)':es}};}
    else{const conv=document.getElementById('conventionSel').value;const es=getConventional(test,conv);res={effect:es,label:`Cohen's convention (${conv})`,formula:`Conventional ${conv} effect size=${es}`,n:getActualN(test),detail:{Convention:conv,'Effect size':es}};}
  }catch(err){alert('Error: '+err.message);return;}
  const requiredN=solveN(res.effect,alpha,power,tails,test);
  const achievedPower=computePower(res.n,res.effect,alpha,tails,test);
  let verdict=res.n>=requiredN?'green':res.n>=requiredN*0.9?'orange':'red';
  renderResults({...res,test,alpha,power,tails,requiredN,achievedPower,verdict});
}
const testNames={independent_t:'Independent t-test',paired_t:'Paired t-test',one_way_anova:'One-way ANOVA',correlation:'Pearson correlation',two_prop:'Two proportions z-test',one_sample_t:'One-sample t-test'};
function renderResults(p){
  document.getElementById('resultSection').style.display='';
  const msgs={green:{title:'Sample size is sufficient ✓',desc:`Current N=${p.n} meets required N=${p.requiredN}.`},orange:{title:'Sample size is borderline ⚠',desc:`Current N=${p.n} is within 10% of required N=${p.requiredN}. Consider ${p.requiredN-p.n} more observations.`},red:{title:'Sample size is insufficient ✗',desc:`Current N=${p.n} is below required N=${p.requiredN}. Need ${p.requiredN-p.n} more observations.`}};
  const m=msgs[p.verdict];
  document.getElementById('statusBar').innerHTML=`<div class="status-bar ${p.verdict}"><div class="status-dot"></div><div><div class="status-title">${m.title}</div><div class="status-desc">${m.desc}</div></div></div>`;
  const pct=(p.achievedPower*100).toFixed(1);
  const bc=p.verdict==='green'?'#22c55e':p.verdict==='orange'?'#f59e0b':'#ef4444';
  document.getElementById('metricGrid').innerHTML=`<div class="metric-card"><div class="metric-label">Effect size</div><div class="metric-value">${p.effect.toFixed(3)}</div><div class="metric-sub">${p.label}</div></div><div class="metric-card"><div class="metric-label">Current N</div><div class="metric-value">${p.n}</div><div class="metric-sub">in dataset</div></div><div class="metric-card"><div class="metric-label">Required N</div><div class="metric-value">${p.requiredN}</div><div class="metric-sub">α=${p.alpha}, 1−β=${p.power}</div></div><div class="metric-card"><div class="metric-label">Achieved power</div><div class="metric-value">${pct}%</div><div class="metric-sub">with current N</div></div>`;
  const bp=Math.min(100,Math.round(p.achievedPower*100));
  document.getElementById('powerBarSection').innerHTML=`<div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:5px;"><span>Achieved power</span><span>${bp}% / target ${Math.round(p.power*100)}%</span></div><div class="power-bar-outer"><div class="power-bar-inner" style="width:${bp}%;background:${bc};"></div></div>`;
  buildPowerChart(p);
  buildDetailTable(p);
  const interp={independent_t:"Cohen's d is the standardised mean difference. d≈0.2 small, d≈0.5 medium, d≈0.8 large. Power uses non-centrality δ=d·√(n/2).",paired_t:"Cohen's dz divides the mean of the differences by its SD, capturing within-subject correlation.",one_way_anova:"Cohen's f is derived from η² (variance explained). f=0.1 small, f=0.25 medium, f=0.40 large.",correlation:"Power uses the Fisher z-transformation. |r|=0.1 small, |r|=0.3 medium, |r|=0.5 large.",two_prop:"Cohen's h uses the arcsine transformation. h=0.2 small, h=0.5 medium, h=0.8 large.",one_sample_t:"Cohen's d compares sample mean to a reference μ₀, standardised by sample SD."};
  document.getElementById('interpretBox').textContent=interp[p.test]||'';
  document.getElementById('formulaBox').innerHTML=`<div class="formula">Effect size: ${p.formula}</div>`;
  document.getElementById('resultSection').scrollIntoView({behavior:'smooth',block:'start'});
}
function buildPowerChart(p){
  if(chartInstance){chartInstance.destroy();chartInstance=null;}
  const maxN=Math.max(p.requiredN*2,p.n*2,60);
  const step=Math.max(1,Math.floor(maxN/100));
  const nV=[],pwV=[];
  for(let n=2;n<=maxN;n+=step){nV.push(n);pwV.push(parseFloat((computePower(n,p.effect,p.alpha,p.tails,p.test)*100).toFixed(2)));}
  [p.n,p.requiredN].forEach(nv=>{if(!nV.includes(nv))nV.push(nv);});
  nV.sort((a,b)=>a-b);
  const pw2=nV.map(n=>parseFloat((computePower(n,p.effect,p.alpha,p.tails,p.test)*100).toFixed(2)));
  const curPt=nV.map(n=>n===p.n?computePower(p.n,p.effect,p.alpha,p.tails,p.test)*100:null);
  const reqPt=nV.map(n=>n===p.requiredN?p.power*100:null);
  document.getElementById('chartLegend').innerHTML=`<span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#64748b;"><span style="width:18px;height:2px;background:#6366f1;display:inline-block;border-radius:1px;"></span>Power curve</span><span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#64748b;"><span style="width:18px;height:1.5px;background:#22c55e;display:inline-block;border-top:2px dashed #22c55e;"></span>Target ${Math.round(p.power*100)}%</span><span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#64748b;"><span style="width:10px;height:10px;border-radius:50%;background:#ef4444;display:inline-block;"></span>Current N=${p.n}</span><span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#64748b;"><span style="width:10px;height:10px;border-radius:50%;background:#22c55e;display:inline-block;"></span>Required N=${p.requiredN}</span>`;
  chartInstance=new Chart(document.getElementById('powerChart'),{type:'line',data:{labels:nV,datasets:[{label:'Power',data:pw2,borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,.07)',borderWidth:2.5,pointRadius:0,fill:true,tension:.35},{label:'Target',data:nV.map(()=>p.power*100),borderColor:'rgba(34,197,94,.7)',borderDash:[6,4],borderWidth:1.5,pointRadius:0,fill:false},{label:'Current N',data:curPt,borderColor:'#ef4444',backgroundColor:'#ef4444',pointRadius:nV.map(n=>n===p.n?7:0),showLine:false,fill:false},{label:'Required N',data:reqPt,borderColor:'#22c55e',backgroundColor:'#22c55e',pointRadius:nV.map(n=>n===p.requiredN?7:0),showLine:false,fill:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',titleColor:'#f1f5f9',bodyColor:'#94a3b8',callbacks:{title:i=>`N = ${i[0].label}`,label:i=>i.raw!==null?`${i.dataset.label}: ${typeof i.raw==='number'?i.raw.toFixed(1):''}%`:''}}},scales:{x:{grid:{color:'rgba(0,0,0,.04)'},ticks:{color:'#94a3b8',font:{size:11}},title:{display:true,text:'Sample size (N)',color:'#94a3b8',font:{size:12}}},y:{grid:{color:'rgba(0,0,0,.04)'},ticks:{color:'#94a3b8',font:{size:11},callback:v=>v+'%'},title:{display:true,text:'Statistical power (%)',color:'#94a3b8',font:{size:12}},min:0,max:100}}}});
}
function buildDetailTable(p){
  const allD={'Test':testNames[p.test],'Alpha (α)':p.alpha,'Target power (1−β)':p.power,'Tails':p.tails,...p.detail,'Current N':p.n,'Required N':p.requiredN,'Deficit/Surplus':(p.n-p.requiredN),'Achieved power':(p.achievedPower*100).toFixed(2)+'%','Verdict':{green:'Sufficient ✓',orange:'Borderline ⚠',red:'Insufficient ✗'}[p.verdict]};
  const hl=['Required N','Achieved power','Verdict'];
  let rows=Object.entries(allD).map(([k,v])=>`<tr${hl.includes(k)?' class="highlight"':''}><td>${k}</td><td>${v}</td></tr>`).join('');
  document.getElementById('detailTableWrap').innerHTML=`<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function switchTab(name,btn){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  ['tabSummary','tabPower','tabDetail'].forEach(id=>document.getElementById(id).style.display='none');
  document.getElementById({summary:'tabSummary',power:'tabPower',detail:'tabDetail'}[name]).style.display='';
  if(name==='power'&&chartInstance)setTimeout(()=>chartInstance.resize(),50);
}
</script>
</body>
</html>"""

    components.html(CALC_HTML, height=1100, scrolling=True)
    st.stop()


# ═══════════════════════════════════════════════════════════
# ══════════  STATISTICAL ANALYSIS TOOL  ════════════════════
# ═══════════════════════════════════════════════════════════

# ── Back button in sidebar ───────────────────────────────
with st.sidebar:
    if st.button("← Tool selector", key="back_to_hub_sb"):
        st.session_state["active_tool"] = None
        st.rerun()

# ─────────────────────────────────────────────────────────
# GUIDE CONTENT
# ─────────────────────────────────────────────────────────
GUIDES = {
    "default": {"title": "👋 Getting Started", "body": "Upload your Excel file or try sample data.<br><br><b>Select any option</b> in the sidebar — this panel will show a contextual guide.<br><br><b>Two layouts:</b><br>• <code>Long</code> — one measurement col + one group col<br>• <code>Wide</code> — each group is its own column"},
    "nominal": {"title": "🟢 Nominal Data", "body": "<b>Nominal</b> — categories with <em>no order</em>.<br><br>Examples: Blood group, Gender code 0/1, Treatment arm.<br><br>✅ Tests: Chi-square, Mann-Whitney, ANOVA (as grouping factor)."},
    "ordinal": {"title": "🟡 Ordinal Data", "body": "<b>Ordinal</b> — ranked categories with unequal gaps.<br><br>Examples: Pain 0–3, Likert 1–5, Disease stage I–IV.<br><br>✅ Tests: Mann-Whitney, Kruskal-Wallis, Spearman, Wilcoxon.<br><br>⚠️ Use <b>median &amp; IQR</b>, not mean ± SD."},
    "wide": {"title": "📊 Wide Format", "body": "Each <b>group</b> is its own column.<br><br><code>Day | Cow_A | Cow_B | Cow_C</code><br><br>Select 2 cols → t-test + Mann-Whitney<br>Select ≥3 cols → ANOVA + Kruskal-Wallis + post-hoc"},
    "long": {"title": "📋 Long Format", "body": "One <b>measurement</b> col + one <b>group</b> col.<br><br><code>Milk_Yield | Cow</code><br><br>Pick numeric → <code>Milk_Yield</code>, categorical → <code>Cow</code>."},
    "outliers": {"title": "🧹 Outlier Removal", "body": "IQR method:<br><code>Lower = Q1 − 1.5×IQR</code><br><code>Upper = Q3 + 1.5×IQR</code><br><br>Applied per group. Removed rows shown in <b>Outlier Report</b>."},
    "paired": {"title": "🔗 Paired Observations", "body": "Tick when each row is a matched pair.<br><br>Examples: Before/After, Left/Right eye, Same cow same day.<br><br>Tests: Paired t-test (normal) or Wilcoxon (non-parametric)."},
    "chi2": {"title": "🔢 Chi-square Test", "body": "Tests association between two <b>categorical</b> variables.<br><br>Requirements: expected cell count ≥5.<br><br>Output: χ², p-value, df, contingency table + heatmap."},
    "friedman": {"title": "♻️ Friedman Test", "body": "Non-parametric repeated-measures ANOVA.<br><br>Use for ≥3 matched groups / time points.<br><br>Wide format with ≥3 columns; each row = one subject."},
    "twoway": {"title": "⚗️ Two-way ANOVA", "body": "Tests two factors + their interaction on a numeric outcome.<br><br>Output: main effect F1, main effect F2, interaction F1×F2."},
    "mlr": {"title": "📐 Multiple Linear Regression", "body": "Models how multiple predictors explain a numeric outcome.<br><br>Output: R², coefficients, p-values, residual plots."},
    "impute": {"title": "🔧 Missing Data", "body": "Options:<br>• <b>None</b> — drop rows with any NaN<br>• <b>Mean</b> — good for symmetric data<br>• <b>Median</b> — better for skewed data<br>• <b>Mode</b> — most frequent value"},
    "scatter": {"title": "🔵 Extra Plot Types", "body": "<b>Scatter:</b> raw points + Pearson r<br><b>Regression:</b> adds 95% CI band<br><b>Bland-Altman:</b> method agreement (needs 2 Y)<br><b>Violin:</b> distribution shape<br><b>Histogram:</b> frequency + KDE<br><b>Bar Chart:</b> mean ± SD<br><b>Paired Lines:</b> before/after per subject"},
}


def render_guide(key):
    g = GUIDES.get(key, GUIDES["default"])
    st.markdown(
        f'<div class="guide-box"><h4>{g["title"]}</h4>{g["body"]}</div>',
        unsafe_allow_html=True,
    )


def set_guide(key):
    st.session_state["guide_key"] = key


# ─────────────────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────────────────
def make_sample_data():
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        "Subject": list(range(1, n + 1)),
        "Group": ["Control"] * 20 + ["Treatment"] * 20,
        "Sex": np.random.choice(["M", "F"], n),
        "Timepoint": np.tile(["Pre", "Post"], n // 2),
        "Age": np.random.randint(25, 65, n),
        "Score_A": np.concatenate([np.random.normal(50, 10, 20), np.random.normal(62, 10, 20)]),
        "Score_B": np.concatenate([np.random.normal(48, 12, 20), np.random.normal(60, 11, 20)]),
        "Score_C": np.concatenate([np.random.normal(52, 9, 20), np.random.normal(58, 13, 20)]),
        "Biomarker": np.concatenate([np.random.normal(1.2, 0.3, 20), np.random.normal(0.9, 0.2, 20)]),
        "Pain_Code": np.random.choice([0, 1, 2, 3], n),
        "Gender_Code": np.random.choice([0, 1], n),
        "Pain_Label": np.random.choice(["None", "Mild", "Moderate", "Severe"], n),
    })


# ─────────────────────────────────────────────────────────
# STAT HELPERS
# ─────────────────────────────────────────────────────────
def cohens_d_ind(g1, g2):
    n1, n2 = len(g1), len(g2)
    p = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / p if p else np.nan

def cohens_d_paired(diff):
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd else np.nan

def mw_r(u, n1, n2): return 1 - (2*u) / (n1*n2)

def wil_r(stat, n):
    exp = n*(n+1)/4
    std_w = np.sqrt(n*(n+1)*(2*n+1)/24)
    return abs((stat-exp)/std_w) / np.sqrt(n) if std_w else np.nan

def eta_sq(f, dfb, dfw): return (f*dfb) / (f*dfb + dfw)
def eps_sq(h, n): return h / (n-1)

def star(p):
    if pd.isna(p): return "ns"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def add_row(rows, cmp, zone, gs, test, p, eff, etype, nt=""):
    try: pv = round(float(p), 4) if not np.isnan(float(p)) else np.nan
    except: pv = np.nan
    try: ev = round(float(eff), 4) if not np.isnan(float(eff)) else np.nan
    except: ev = np.nan
    rows.append({"Comparison": cmp, "Variable/Zone": zone,
        "Group Stats (mean±SD)": gs, "Test": test,
        "P-value": pv, "Significance": star(pv),
        "Effect Size": ev, "Effect Type": etype, "Normality": nt})

def grp_stat_str(sd):
    parts = []
    for nm, s in sd.items():
        s = s.dropna()
        parts.append(f"{nm} n={len(s)} | {s.mean():.2f}±{s.std():.2f}")
    return "  |  ".join(parts)

def norm_str(sd):
    parts = []
    for nm, s in sd.items():
        s = s.dropna()
        if len(s) >= 3:
            _, p = shapiro(s)
            parts.append(f"{nm}: {'Normal' if p >= 0.05 else 'Non-normal'}")
        else:
            parts.append(f"{nm}: n<3")
    return "  |  ".join(parts)

def all_normal(sl):
    for s in sl:
        s = s.dropna()
        if len(s) < 3 or shapiro(s)[1] < 0.05:
            return False
    return True

def iqr_mask(s):
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return (s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)

def save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf

def smart_grid(n):
    if n <= 3: return 1, n
    if n == 4: return 2, 2
    if n <= 6: return 2, 3
    if n <= 9: return 3, 3
    return (n+3)//4, 4

def dunn_bonf(gd):
    names = list(gd.keys())
    res = []
    for a, b in itertools.combinations(names, 2):
        g1, g2 = gd[a], gd[b]
        if len(g1) < 2 or len(g2) < 2: continue
        u = mannwhitneyu(g1, g2, alternative="two-sided")
        res.append({"Group1": a, "Group2": b, "P_raw": u.pvalue,
                    "Effect_Size": mw_r(u.statistic, len(g1), len(g2))})
    if res:
        nt = len(res)
        for r in res:
            r["P_Bonferroni"] = min(r["P_raw"] * nt, 1.0)
            r["Significance"] = star(r["P_Bonferroni"])
    return pd.DataFrame(res)


# ═══════════════════════════════════════════════════════════
# SIDEBAR — Stats Tool
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("① Data Source")
    data_src = st.radio("Source", ["📁 Upload file (Excel/CSV)", "🧪 Sample data"])

    uploaded_file = None
    if data_src == "📁 Upload file (Excel/CSV)":
        uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
        if uploaded_file:
            try:
                name = getattr(uploaded_file, "name", "").lower()
                if name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                if not isinstance(st.session_state["df_loaded"], pd.DataFrame) or \
                   not df_raw.equals(st.session_state["df_loaded"]):
                    st.session_state["df_loaded"] = df_raw
                    st.session_state["analysis_done"] = False
                st.success(f"✅ {df_raw.shape[0]}r × {df_raw.shape[1]}c")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        df_raw = make_sample_data()
        if st.session_state["df_loaded"] is None or \
           not isinstance(st.session_state["df_loaded"], pd.DataFrame):
            st.session_state["df_loaded"] = df_raw
        st.info(f"🧪 Sample: {df_raw.shape[0]}r × {df_raw.shape[1]}c")

    df_raw = st.session_state.get("df_loaded")

    if df_raw is not None:
        _num0 = df_raw.select_dtypes(include=np.number).columns.tolist()
        _cat0 = df_raw.select_dtypes(exclude=np.number).columns.tolist()

        st.header("② Column Type Override")
        override_nominal = st.multiselect(
            "🟢 Treat as Nominal", _num0,
            default=st.session_state["override_nominal"],
            on_change=set_guide, args=("nominal",))
        st.session_state["override_nominal"] = override_nominal

        override_ordinal = st.multiselect(
            "🟡 Treat as Ordinal",
            [c for c in _num0 if c not in override_nominal],
            default=[c for c in st.session_state["override_ordinal"] if c not in override_nominal],
            on_change=set_guide, args=("ordinal",))
        st.session_state["override_ordinal"] = override_ordinal

        _all_override = override_nominal + override_ordinal
        num_cols = [c for c in _num0 if c not in _all_override]
        cat_cols  = _cat0 + _all_override
        col_subtype = {c: "Nominal" for c in override_nominal}
        col_subtype.update({c: "Ordinal" for c in override_ordinal})

        st.header("③ Missing Data")
        impute_method = st.selectbox(
            "Impute", ["None (drop rows)", "Mean", "Median", "Mode (most frequent)"],
            on_change=set_guide, args=("impute",))

        st.header("④ Analysis Mode")
        mode = st.radio(
            "Data structure",
            ["📋 Long format  (measurement col + group col)",
             "📊 Wide format  (each group = its own column)"],
            on_change=set_guide, args=("long",))
        wide_mode = mode.startswith("📊")
        if wide_mode: set_guide("wide")

        st.header("⑤ Variables")
        if wide_mode:
            group_cols_wide = st.multiselect("Compare these columns (≥2)", num_cols)
            selected_numeric, selected_cat = [], []
        else:
            selected_numeric = st.multiselect("Numeric (dependent) variables", num_cols)
            selected_cat     = st.multiselect("Categorical grouping variables", cat_cols)
            group_cols_wide  = []

        st.header("⑥ Options")
        remove_outliers = st.checkbox("🧹 Remove outliers (IQR)", value=True,
                                      on_change=set_guide, args=("outliers",))
        do_paired = st.checkbox("🔗 Paired / matched", value=False,
                                on_change=set_guide, args=("paired",))
        subject_id = pairing_col = None
        if do_paired and not wide_mode:
            subject_id  = st.selectbox("Subject ID column", ["None"] + list(df_raw.columns))
            pairing_col = st.selectbox("Pairing column", ["None"] + cat_cols)
            subject_id  = None if subject_id == "None" else subject_id
            pairing_col = None if pairing_col == "None" else pairing_col

        do_corr   = st.checkbox("Correlation matrix & heatmap", value=True)
        use_facet = st.checkbox("Faceted boxplots", value=True)

        st.header("⑦ Advanced Tests")
        do_chi2     = st.checkbox("Chi-square", value=False, on_change=set_guide, args=("chi2",))
        do_friedman = st.checkbox("Friedman test", value=False, on_change=set_guide, args=("friedman",))
        do_twoway   = st.checkbox("Two-way ANOVA", value=False, on_change=set_guide, args=("twoway",))
        do_mlr      = st.checkbox("Multiple linear regression", value=False, on_change=set_guide, args=("mlr",))

        chi2_vars = twoway_cat1 = twoway_cat2 = twoway_vars = mlr_outcome = mlr_predictors = None
        if do_chi2 and len(cat_cols) >= 2:
            chi2_vars = st.multiselect("Chi-square: 2 categorical cols", cat_cols, max_selections=2)
        if do_twoway and not wide_mode:
            twoway_vars = st.multiselect("Two-way ANOVA: numeric outcome(s)", num_cols)
            twoway_cat1 = st.selectbox("Factor 1", ["None"] + cat_cols)
            twoway_cat2 = st.selectbox("Factor 2", ["None"] + cat_cols)
            twoway_cat1 = None if twoway_cat1 == "None" else twoway_cat1
            twoway_cat2 = None if twoway_cat2 == "None" else twoway_cat2
        if do_mlr:
            mlr_outcome    = st.selectbox("MLR outcome (Y)", ["None"] + num_cols)
            mlr_predictors = st.multiselect("MLR predictors (X)", num_cols)
            mlr_outcome    = None if mlr_outcome == "None" else mlr_outcome

        st.header("⑧ Extra Plots")
        PLOT_TYPES = ["Scatter Plot", "Regression Plot", "Bland-Altman",
                      "Mean Line Graph", "Violin Plot", "Histogram",
                      "Bar Chart (Mean±SD)", "Paired Lines Plot"]
        selected_plots = st.multiselect("Plot types", PLOT_TYPES, default=[],
                                        on_change=set_guide, args=("scatter",))
        xvars_extra = st.multiselect("X variable(s)", num_cols)
        yvars_extra = st.multiselect("Y variable(s)", num_cols)
        group_extra = st.selectbox("Grouping (optional)", ["None"] + cat_cols)
        group_extra = None if group_extra == "None" else group_extra

        c1b, c2b = st.columns(2)
        with c1b:
            if st.button("🔄 Reset", use_container_width=True):
                for k in list(_ss_defaults.keys()):
                    st.session_state[k] = _ss_defaults[k]
                st.rerun()
        with c2b:
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        st.markdown("""<div class="linkedin-bar">✉️ <a href="https://www.linkedin.com/in/syed-ishaq-893052285/" target="_blank">🔗 Syed Ishaq on LinkedIn</a></div>""", unsafe_allow_html=True)

    else:
        run_btn = False

# ─────────────────────────────────────────────────────────
# PHASE DETECTION
# ─────────────────────────────────────────────────────────
phase = 1
if df_raw is not None:
    phase = 3 if (st.session_state.get("analysis_done") or run_btn) else 2

# ─────────────────────────────────────────────────────────
# PHASE 1
# ─────────────────────────────────────────────────────────
if phase == 1:
    st.markdown("""
## Welcome to the Statistical Analysis Tool! 👋

### 📊 Wide Format
| Day | Cow_A | Cow_B | Cow_C |
|-----|-------|-------|-------|
| 1 | 25.3 | 22.8 | 21.5 |

→ 2 cols → t-test + Mann-Whitney · ≥3 cols → ANOVA + Kruskal-Wallis

### 📋 Long Format
| Milk_Yield | Cow |
|------------|-----|
| 25.3 | A |

### ⚙️ Features
| Feature | Details |
|---------|---------|
| Parametric | t-test, paired t-test, one-way & two-way ANOVA |
| Non-parametric | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman |
| Post-hoc | Tukey HSD, Dunn-Bonferroni |
| Categorical | Chi-square |
| Regression | MLR with diagnostics |
| Correlation | Pearson + Spearman heatmaps |
| Effect sizes | Cohen's d, eta-squared, rank-biserial r |
| 8 plot types | Scatter, Regression, Violin, Bland-Altman, and more |

👈 **Upload your file or choose Sample data in the sidebar to begin.**
""")
    st.stop()

# ─────────────────────────────────────────────────────────
# PHASE 2
# ─────────────────────────────────────────────────────────
if phase == 2:
    with st.expander("🔍 Data Preview (first 10 rows)", expanded=False):
        st.dataframe(df_raw.head(10), use_container_width=True)
    st.info("👈 Configure your analysis in the sidebar, then click **🚀 Run Analysis**.")
    st.markdown("#### 📖 Guide")
    render_guide(st.session_state["guide_key"])
    st.stop()

# ─────────────────────────────────────────────────────────
# PHASE 3 — RUN ANALYSIS
# ─────────────────────────────────────────────────────────
if run_btn:
    if wide_mode and len(group_cols_wide) < 2:
        st.error("⚠️ Select ≥2 columns in wide-format mode."); st.stop()
    if not wide_mode and not selected_numeric:
        st.error("⚠️ Select at least one numeric variable."); st.stop()
    if not wide_mode and not selected_cat and len(selected_numeric) < 2:
        st.error("⚠️ Select ≥2 numeric variables OR add a categorical grouping variable."); st.stop()

    progress = st.progress(0, "Preparing data…")

    df = df_raw.copy()
    for c in override_nominal + override_ordinal:
        if c in df.columns: df[c] = df[c].astype(str)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        df[c] = df[c].astype(str)

    if impute_method != "None (drop rows)":
        for c in num_cols:
            if df[c].isna().any():
                if   impute_method == "Mean":   df[c].fillna(df[c].mean(),    inplace=True)
                elif impute_method == "Median": df[c].fillna(df[c].median(),  inplace=True)
                else:                           df[c].fillna(df[c].mode()[0], inplace=True)

    outlier_df = pd.DataFrame(); outlier_rows = set()
    if remove_outliers:
        cols_out = group_cols_wide if wide_mode else selected_numeric
        for col in cols_out:
            if selected_cat and not wide_mode:
                for grp in selected_cat:
                    for g in df[grp].dropna().unique():
                        sub = df[df[grp] == g][col].dropna()
                        if len(sub) >= 4: outlier_rows.update(sub[iqr_mask(sub)].index.tolist())
            else:
                sub = df[col].dropna()
                if len(sub) >= 4: outlier_rows.update(sub[iqr_mask(sub)].index.tolist())
        if outlier_rows:
            keep = list(set(group_cols_wide + selected_numeric + selected_cat + ([subject_id] if subject_id else [])))
            keep = [c for c in keep if c in df.columns]
            outlier_df = df.loc[list(outlier_rows), keep].copy()
            df = df.drop(index=outlier_rows).reset_index(drop=True)

    progress.progress(15, "Running statistical tests…")
    unified_rows = []; posthoc_rows = []

    if wide_mode:
        n_g = len(group_cols_wide)
        gdata = {c: df[c].dropna().values for c in group_cols_wide}
        sdict = {c: df[c].dropna() for c in group_cols_wide}
        gs = grp_stat_str(sdict); nt = norm_str(sdict)
        label = " vs ".join(group_cols_wide)
        if n_g == 2:
            c1, c2 = group_cols_wide; g1, g2 = gdata[c1], gdata[c2]
            if do_paired:
                paired = df[[c1, c2]].dropna(); diff = paired[c1] - paired[c2]
                t, tp = ttest_rel(paired[c1], paired[c2])
                add_row(unified_rows, label, "Paired", gs, "Paired t-test", tp, cohens_d_paired(diff.values), "Cohen's d (paired)", nt)
                try:
                    w = wilcoxon(diff.values)
                    add_row(unified_rows, label, "Paired", gs, "Wilcoxon signed-rank", w.pvalue, wil_r(w.statistic, len(diff)), "Rank-biserial r", nt)
                except: pass
            else:
                t, tp = ttest_ind(g1, g2)
                add_row(unified_rows, label, "Column comparison", gs, "Independent t-test", tp, cohens_d_ind(g1, g2), "Cohen's d", nt)
                u = mannwhitneyu(g1, g2, alternative="two-sided")
                add_row(unified_rows, label, "Column comparison", gs, "Mann-Whitney U", u.pvalue, mw_r(u.statistic, len(g1), len(g2)), "Rank-biserial r", nt)
        else:
            arrs = list(gdata.values()); n_tot = sum(len(a) for a in arrs)
            f, fp = f_oneway(*arrs)
            add_row(unified_rows, label, "Column comparison", gs, "One-way ANOVA", fp, eta_sq(f, n_g-1, n_tot-n_g), "Eta-squared", nt)
            if fp < 0.05:
                sv = np.concatenate(arrs)
                sg = np.concatenate([[c]*len(gdata[c]) for c in group_cols_wide])
                tukey = pairwise_tukeyhsd(sv, sg, alpha=0.05)
                td = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                for _, row in td.iterrows():
                    posthoc_rows.append({"Grouping": "Wide", "Variable": "Columns",
                        "Comparison": f"{row['group1']} vs {row['group2']}",
                        "Test": "Tukey HSD", "P_adj": round(float(row["p-adj"]), 4),
                        "Significance": star(row["p-adj"]),
                        "Mean_Diff": round(float(row["meandiff"]), 4),
                        "Lower_CI": round(float(row["lower"]), 4),
                        "Upper_CI": round(float(row["upper"]), 4)})
            h, hp = kruskal(*arrs)
            add_row(unified_rows, label, "Column comparison", gs, "Kruskal-Wallis", hp, eps_sq(h, n_tot), "Epsilon-squared", nt)
            if hp < 0.05:
                dunn = dunn_bonf(gdata)
                if not dunn.empty:
                    for _, row in dunn.iterrows():
                        posthoc_rows.append({"Grouping": "Wide", "Variable": "Columns",
                            "Comparison": f"{row['Group1']} vs {row['Group2']}",
                            "Test": "Dunn (Bonferroni)",
                            "P_adj": round(float(row["P_Bonferroni"]), 4),
                            "Significance": star(row["P_Bonferroni"]),
                            "Effect_Size": round(float(row["Effect_Size"]), 4),
                            "Effect_Type": "Rank-biserial r"})
    else:
        if selected_numeric and not selected_cat:
            for v1, v2 in itertools.combinations(selected_numeric, 2):
                pdf = df[[v1, v2]].dropna(); g1, g2 = pdf[v1].values, pdf[v2].values
                gs = grp_stat_str({v1: pdf[v1], v2: pdf[v2]}); nt = norm_str({v1: pdf[v1], v2: pdf[v2]})
                lbl = f"{v1} vs {v2}"
                if do_paired:
                    diff = g1-g2; t, tp = ttest_rel(g1, g2)
                    add_row(unified_rows, lbl, "Numeric comparison", gs, "Paired t-test", tp, cohens_d_paired(diff), "Cohen's d (paired)", nt)
                    try:
                        w = wilcoxon(diff)
                        add_row(unified_rows, lbl, "Numeric comparison", gs, "Wilcoxon signed-rank", w.pvalue, wil_r(w.statistic, len(diff)), "Rank-biserial r", nt)
                    except: pass
                else:
                    t, tp = ttest_ind(g1, g2)
                    add_row(unified_rows, lbl, "Numeric comparison", gs, "Independent t-test", tp, cohens_d_ind(g1, g2), "Cohen's d", nt)
                    u = mannwhitneyu(g1, g2, alternative="two-sided")
                    add_row(unified_rows, lbl, "Numeric comparison", gs, "Mann-Whitney U", u.pvalue, mw_r(u.statistic, len(g1), len(g2)), "Rank-biserial r", nt)

        for grp in selected_cat:
            levels = df[grp].dropna().unique(); n_lev = len(levels)
            if n_lev < 2: continue
            is_paired = do_paired and pairing_col and grp == pairing_col and subject_id
            for var in selected_numeric:
                sd_g = {g: df[df[grp]==g][var].dropna() for g in levels}
                gs = grp_stat_str(sd_g); nt = norm_str(sd_g)
                if is_paired:
                    wp = df.pivot(index=subject_id, columns=grp, values=var).dropna()
                    if wp.shape[1] == 2:
                        c1, c2 = wp.columns[0], wp.columns[1]; diff = wp[c1] - wp[c2]
                        t, tp = ttest_rel(wp[c1], wp[c2])
                        add_row(unified_rows, f"{grp}(paired)", var, gs, "Paired t-test", tp, cohens_d_paired(diff.values), "Cohen's d (paired)", nt)
                        try:
                            w = wilcoxon(diff.values)
                            add_row(unified_rows, f"{grp}(paired)", var, gs, "Wilcoxon signed-rank", w.pvalue, wil_r(w.statistic, len(diff)), "Rank-biserial r", nt)
                        except: pass
                elif n_lev == 2:
                    g1 = df[df[grp]==levels[0]][var].dropna().values
                    g2 = df[df[grp]==levels[1]][var].dropna().values
                    if len(g1) >= 2 and len(g2) >= 2:
                        lbl = f"{grp} ({levels[0]} vs {levels[1]})"
                        t, tp = ttest_ind(g1, g2)
                        add_row(unified_rows, lbl, var, gs, "Independent t-test", tp, cohens_d_ind(g1, g2), "Cohen's d", nt)
                        u = mannwhitneyu(g1, g2, alternative="two-sided")
                        add_row(unified_rows, lbl, var, gs, "Mann-Whitney U", u.pvalue, mw_r(u.statistic, len(g1), len(g2)), "Rank-biserial r", nt)
                else:
                    sub = df.dropna(subset=[var, grp])
                    arrs = [sub[sub[grp]==g][var].dropna().values for g in levels]
                    if not all(len(a) >= 2 for a in arrs): continue
                    n_tot = len(sub); f, fp = f_oneway(*arrs)
                    add_row(unified_rows, grp, var, gs, "One-way ANOVA", fp, eta_sq(f, n_lev-1, n_tot-n_lev), "Eta-squared", nt)
                    if fp < 0.05:
                        tukey = pairwise_tukeyhsd(sub[var], sub[grp], alpha=0.05)
                        td = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                        for _, row in td.iterrows():
                            posthoc_rows.append({"Grouping": grp, "Variable": var,
                                "Comparison": f"{row['group1']} vs {row['group2']}",
                                "Test": "Tukey HSD", "P_adj": round(float(row["p-adj"]), 4),
                                "Significance": star(row["p-adj"]),
                                "Mean_Diff": round(float(row["meandiff"]), 4),
                                "Lower_CI": round(float(row["lower"]), 4),
                                "Upper_CI": round(float(row["upper"]), 4)})
                    h, hp = kruskal(*arrs)
                    add_row(unified_rows, grp, var, gs, "Kruskal-Wallis", hp, eps_sq(h, n_tot), "Epsilon-squared", nt)
                    if hp < 0.05:
                        dunn = dunn_bonf({g: sub[sub[grp]==g][var].dropna().values for g in levels})
                        if not dunn.empty:
                            for _, row in dunn.iterrows():
                                posthoc_rows.append({"Grouping": grp, "Variable": var,
                                    "Comparison": f"{row['Group1']} vs {row['Group2']}",
                                    "Test": "Dunn (Bonferroni)",
                                    "P_adj": round(float(row["P_Bonferroni"]), 4),
                                    "Significance": star(row["P_Bonferroni"]),
                                    "Effect_Size": round(float(row["Effect_Size"]), 4),
                                    "Effect_Type": "Rank-biserial r"})

    progress.progress(38, "Advanced tests…")

    chi2_results = []
    if do_chi2 and chi2_vars and len(chi2_vars) == 2:
        c1v, c2v = chi2_vars
        ct = pd.crosstab(df[c1v], df[c2v])
        chi2s, pc, dof, exp = chi2_contingency(ct)
        chi2_results.append({"Var1": c1v, "Var2": c2v,
            "Chi-sq": round(chi2s, 4), "df": dof, "P-value": round(pc, 4),
            "Significance": star(pc), "N": int(ct.values.sum()),
            "Note": "Fisher's exact recommended (expected cell <5)" if (exp<5).any() else ""})

    friedman_results = []
    if do_friedman and wide_mode and len(group_cols_wide) >= 3:
        pf = df[group_cols_wide].dropna()
        if len(pf) >= 3:
            F, fp = friedmanchisquare(*[pf[c].values for c in group_cols_wide])
            friedman_results.append({"Columns": " | ".join(group_cols_wide),
                "Friedman stat": round(F, 4), "P-value": round(fp, 4),
                "Significance": star(fp), "N": len(pf)})

    twoway_results = []
    if do_twoway and not wide_mode and twoway_vars and twoway_cat1 and twoway_cat2:
        for var in twoway_vars:
            sub = df[[var, twoway_cat1, twoway_cat2]].dropna().copy()
            sub.columns = ["Y", "F1", "F2"]
            try:
                model = smf.ols("Y ~ C(F1) + C(F2) + C(F1):C(F2)", data=sub).fit()
                at = sm.stats.anova_lm(model, typ=2)
                for idx, row in at.iterrows():
                    twoway_results.append({"Outcome": var, "Effect": str(idx),
                        "SS": round(float(row["sum_sq"]), 4), "df": round(float(row["df"]), 2),
                        "F": round(float(row["F"]), 4) if "F" in row else np.nan,
                        "P-value": round(float(row["PR(>F)"]), 4) if "PR(>F)" in row else np.nan,
                        "Significance": star(row["PR(>F)"]) if "PR(>F)" in row else ""})
            except Exception as ex:
                twoway_results.append({"Outcome": var, "Effect": f"Error: {ex}",
                    "SS": np.nan, "df": np.nan, "F": np.nan, "P-value": np.nan, "Significance": ""})

    mlr_results = []; mlr_fig_bytes = None; mlr_txt = ""
    if do_mlr and mlr_outcome and mlr_predictors:
        sub_mlr = df[[mlr_outcome] + mlr_predictors].dropna()
        if len(sub_mlr) > len(mlr_predictors) + 1:
            X = sm.add_constant(sub_mlr[mlr_predictors]); y = sub_mlr[mlr_outcome]
            mod = sm.OLS(y, X).fit(); mlr_txt = mod.summary().as_text()
            for nm, coef, se, t, p in zip(mod.params.index, mod.params.values,
                                           mod.bse.values, mod.tvalues.values, mod.pvalues.values):
                mlr_results.append({"Predictor": nm, "Coefficient": round(coef, 4),
                    "Std Error": round(se, 4), "t-stat": round(t, 4),
                    "P-value": round(p, 4), "Significance": star(p)})
            fig_r, axes_r = plt.subplots(1, 2, figsize=(12, 5))
            axes_r[0].scatter(mod.fittedvalues, mod.resid, alpha=0.6, s=40, color="steelblue")
            axes_r[0].axhline(0, color="red", ls="--", lw=1.5)
            axes_r[0].set_xlabel("Fitted"); axes_r[0].set_ylabel("Residuals")
            axes_r[0].set_title("Residuals vs Fitted", fontweight="bold")
            sm.qqplot(mod.resid, line="s", ax=axes_r[1], alpha=0.6)
            axes_r[1].set_title("Q-Q Plot", fontweight="bold")
            fig_r.suptitle(f"MLR Diagnostics — {mlr_outcome}", fontsize=12, fontweight="bold")
            fig_r.tight_layout()
            mlr_fig_bytes = save_fig(fig_r).read(); plt.close(fig_r)

    progress.progress(60, "Descriptive stats & correlation…")

    vars_desc = group_cols_wide if wide_mode else selected_numeric
    desc_rows = []
    for var in vars_desc:
        s = df[var].dropna()
        if len(s) == 0: continue
        _, sw_p = shapiro(s) if len(s) >= 3 else (np.nan, np.nan)
        desc_rows.append({"Variable": var, "N": len(s), "Missing": df[var].isna().sum(),
            "Mean": round(s.mean(), 3), "SD": round(s.std(), 3), "SE": round(s.sem(), 3),
            "Min": round(s.min(), 3), "Q1": round(s.quantile(0.25), 3),
            "Median": round(s.median(), 3), "Q3": round(s.quantile(0.75), 3),
            "Max": round(s.max(), 3), "Skewness": round(float(s.skew()), 3),
            "Kurtosis": round(float(s.kurtosis()), 3),
            "Shapiro-Wilk p": round(sw_p, 4) if sw_p else np.nan,
            "Normal?": "Yes" if sw_p and sw_p >= 0.05 else "No"})
    desc_df = pd.DataFrame(desc_rows)

    desc_fig_bytes = None
    if vars_desc:
        n_d = len(vars_desc); nr_d, nc_d = smart_grid(n_d)
        fig_d, axes_d = plt.subplots(nr_d, nc_d, figsize=(5*nc_d, 3.5*nr_d))
        axs_d = np.array(axes_d).flatten() if n_d > 1 else [axes_d]
        for i, var in enumerate(vars_desc):
            sns.histplot(df[var].dropna(), kde=True, ax=axs_d[i], color="steelblue", alpha=0.7)
            axs_d[i].set_title(var, fontsize=9, fontweight="bold"); axs_d[i].set_xlabel("")
        for i in range(n_d, len(axs_d)): axs_d[i].set_visible(False)
        fig_d.suptitle("Distributions", fontsize=12, fontweight="bold")
        fig_d.tight_layout()
        desc_fig_bytes = save_fig(fig_d).read(); plt.close(fig_d)

    corr_vars = group_cols_wide if wide_mode else selected_numeric
    pearson_corr = spearman_corr = pd.DataFrame(); corr_fig_bytes = None
    if do_corr and len(corr_vars) > 1:
        cd = df[corr_vars].dropna()
        pearson_corr = cd.corr(); spearman_corr = cd.corr(method="spearman")
        fs = max(6, len(corr_vars)*0.9)
        fc, ac = plt.subplots(figsize=(fs, fs*0.8))
        sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", center=0,
                    fmt=".2f", linewidths=0.5, ax=ac, square=True, cbar_kws={"shrink": 0.8})
        ac.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
        fc.tight_layout()
        corr_fig_bytes = save_fig(fc).read(); plt.close(fc)

    progress.progress(76, "Generating plots…")
    bp_figs = []
    if wide_mode and group_cols_wide:
        fig, ax = plt.subplots(figsize=(max(8, len(group_cols_wide)*1.8), 6))
        melt = df[group_cols_wide].melt(var_name="Column", value_name="Value").dropna()
        sns.boxplot(data=melt, x="Column", y="Value", ax=ax, palette="Set2")
        sns.stripplot(data=melt, x="Column", y="Value", ax=ax, color="black", alpha=0.35, size=3)
        ax.set_title("Column Comparison — Boxplot", fontweight="bold", fontsize=12)
        ax.tick_params(axis="x", rotation=30); fig.tight_layout()
        bp_figs.append(("Wide-format column comparison", save_fig(fig).read())); plt.close(fig)
    elif selected_cat and use_facet:
        for grp in selected_cat:
            pval_d = {}
            for var in selected_numeric:
                sub = df.dropna(subset=[var, grp]); lev = sub[grp].dropna().unique()
                if len(lev) < 2: continue
                arrs = [sub[sub[grp]==g][var].dropna().values for g in lev]
                nrm = all_normal([sub[sub[grp]==g][var].dropna() for g in lev])
                if len(lev) == 2:
                    g1, g2 = arrs[0], arrs[1]
                    pval_d[var] = ttest_ind(g1, g2)[1] if nrm else mannwhitneyu(g1, g2, alternative="two-sided")[1]
                else:
                    pval_d[var] = f_oneway(*arrs)[1] if nrm else kruskal(*arrs)[1]
            n_v = len(selected_numeric); nr, nc = smart_grid(n_v); n_last = n_v-(nr-1)*nc
            fig = plt.figure(figsize=(5*nc, 4.2*nr))
            gs2 = gridspec.GridSpec(nr, nc, figure=fig, hspace=0.5, wspace=0.35)
            axes = []
            for r in range(nr):
                panels = nc if r < nr-1 else n_last
                offset = int((nc-panels)/2) if panels < nc else 0
                for c in range(panels):
                    try: ax = fig.add_subplot(gs2[r, offset+c])
                    except: ax = fig.add_subplot(gs2[r, c])
                    axes.append(ax)
            for idx, var in enumerate(selected_numeric):
                ax = axes[idx]
                sns.boxplot(data=df, x=grp, y=var, ax=ax, palette="Set2")
                sns.stripplot(data=df, x=grp, y=var, ax=ax, color="black", alpha=0.4, size=2)
                ax.set_title(var, fontweight="bold", fontsize=9); ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=40, labelsize=8)
                pv = pval_d.get(var, np.nan)
                try: pv_f = float(pv)
                except: pv_f = np.nan
                txt = f"p={pv_f:.4f} {star(pv_f)}" if not np.isnan(pv_f) else "p=N/A"
                ax.text(0.5, 0.97, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#aaa", boxstyle="round"))
            fig.suptitle(f"Boxplots by {grp}", fontsize=13, fontweight="bold", y=1.01)
            bp_figs.append((f"Boxplot — {grp}", save_fig(fig).read())); plt.close(fig)

    extra_figs = []
    ep = sns.color_palette("husl", max(len(xvars_extra), len(yvars_extra), 4))
    all_ec = list(set(xvars_extra + yvars_extra + ([group_extra] if group_extra else [])))
    data_ex = df[[c for c in all_ec if c in df.columns]].dropna() if all_ec else pd.DataFrame()

    def scatter_base(xv, yv, ax):
        if group_extra and group_extra in data_ex.columns:
            for gi, (gv, gd) in enumerate(data_ex.groupby(group_extra)):
                ax.scatter(gd[xv], gd[yv], alpha=0.7, label=str(gv), s=45, color=ep[gi%len(ep)])
            ax.legend(title=group_extra, fontsize=8)
        else:
            ax.scatter(data_ex[xv], data_ex[yv], alpha=0.7, s=45, color="steelblue")
        try:
            m, b = np.polyfit(data_ex[xv], data_ex[yv], 1)
            xl = np.linspace(data_ex[xv].min(), data_ex[xv].max(), 100)
            ax.plot(xl, m*xl+b, "r--", lw=1.5)
            r, p = pearsonr(data_ex[xv], data_ex[yv])
            ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.4f} {star(p)}",
                    transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8))
        except: pass
        ax.set_xlabel(xv, fontsize=11); ax.set_ylabel(yv, fontsize=11)

    for pt in selected_plots:
        if pt == "Scatter Plot" and xvars_extra and yvars_extra:
            for xv in xvars_extra:
                for yv in yvars_extra:
                    if xv == yv: continue
                    fig, ax = plt.subplots(figsize=(7, 5)); scatter_base(xv, yv, ax)
                    ax.set_title(f"Scatter: {yv} vs {xv}", fontweight="bold")
                    fig.tight_layout(); extra_figs.append((f"Scatter {yv} vs {xv}", save_fig(fig).read())); plt.close(fig)
        elif pt == "Regression Plot" and xvars_extra and yvars_extra:
            for xv in xvars_extra:
                for yv in yvars_extra:
                    if xv == yv: continue
                    fig, ax = plt.subplots(figsize=(7, 5))
                    if group_extra and group_extra in data_ex.columns:
                        for gi, (gv, gd) in enumerate(data_ex.groupby(group_extra)):
                            sns.regplot(data=gd, x=xv, y=yv, ax=ax, label=str(gv),
                                        color=ep[gi%len(ep)], scatter_kws={"alpha": 0.6, "s": 40}, line_kws={"lw": 2})
                        ax.legend(title=group_extra, fontsize=8)
                    else:
                        sns.regplot(data=data_ex, x=xv, y=yv, ax=ax,
                                    scatter_kws={"alpha": 0.6, "s": 40, "color": "steelblue"}, line_kws={"lw": 2, "color": "red"})
                    try:
                        r, p = pearsonr(data_ex[xv], data_ex[yv])
                        ax.text(0.05, 0.95, f"r={r:.3f}, p={p:.4f} {star(p)}",
                                transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.8))
                    except: pass
                    ax.set_xlabel(xv, fontsize=11); ax.set_ylabel(yv, fontsize=11)
                    ax.set_title(f"Regression: {yv} ~ {xv}", fontweight="bold")
                    fig.tight_layout(); extra_figs.append((f"Regression {yv}~{xv}", save_fig(fig).read())); plt.close(fig)
        elif pt == "Bland-Altman":
            if len(yvars_extra) == 2:
                y1, y2 = yvars_extra; pd2 = data_ex[[y1, y2]].dropna()
                diff = pd2[y1]-pd2[y2]; mean_ba = (pd2[y1]+pd2[y2])/2
                md, sd_ba = diff.mean(), diff.std(ddof=1)
                fig, ax = plt.subplots(figsize=(9, 6))
                if group_extra and group_extra in data_ex.columns:
                    for gi, (gv, gd) in enumerate(data_ex.groupby(group_extra)):
                        ax.scatter((gd[y1]+gd[y2])/2, gd[y1]-gd[y2], alpha=0.7, label=str(gv), s=45, color=ep[gi%len(ep)])
                    ax.legend(title=group_extra)
                else:
                    ax.scatter(mean_ba, diff, alpha=0.7, s=45, color="steelblue")
                ax.axhline(md, color="red", lw=2, label=f"Mean diff={md:.2f}")
                ax.axhline(md+1.96*sd_ba, color="gray", ls="--", lw=1.5, label=f"+1.96SD={md+1.96*sd_ba:.2f}")
                ax.axhline(md-1.96*sd_ba, color="gray", ls="--", lw=1.5, label=f"-1.96SD={md-1.96*sd_ba:.2f}")
                ax.set_xlabel(f"Mean of {y1} & {y2}", fontsize=11); ax.set_ylabel(f"Difference ({y1}−{y2})", fontsize=11)
                ax.set_title(f"Bland-Altman: {y1} vs {y2}", fontweight="bold"); ax.legend(fontsize=8)
                fig.tight_layout(); extra_figs.append((f"Bland-Altman {y1} vs {y2}", save_fig(fig).read())); plt.close(fig)
        elif pt == "Violin Plot" and yvars_extra:
            if group_extra:
                for yv in yvars_extra:
                    pd3 = df[[yv, group_extra]].dropna()
                    fig, ax = plt.subplots(figsize=(9, 5))
                    sns.violinplot(data=pd3, x=group_extra, y=yv, ax=ax, palette="Set2", inner="box")
                    sns.stripplot(data=pd3, x=group_extra, y=yv, ax=ax, color="black", alpha=0.3, size=3)
                    ax.set_title(f"Violin: {yv} by {group_extra}", fontweight="bold")
                    ax.tick_params(axis="x", rotation=40); fig.tight_layout()
                    extra_figs.append((f"Violin {yv}", save_fig(fig).read())); plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(max(6, len(yvars_extra)*2), 5))
                pd3 = df[yvars_extra].melt(var_name="Variable", value_name="Value").dropna()
                sns.violinplot(data=pd3, x="Variable", y="Value", ax=ax, palette="Set2", inner="box")
                ax.set_title("Violin Plot", fontweight="bold"); fig.tight_layout()
                extra_figs.append(("Violin", save_fig(fig).read())); plt.close(fig)
        elif pt == "Histogram" and yvars_extra:
            n_h = len(yvars_extra); nr_h, nc_h = smart_grid(n_h)
            fig, axes_h = plt.subplots(nr_h, nc_h, figsize=(5*nc_h, 4*nr_h))
            axs = np.array(axes_h).flatten() if n_h > 1 else [axes_h]
            for i, yv in enumerate(yvars_extra):
                ax = axs[i]
                if group_extra and group_extra in df.columns:
                    for gi, (gv, gd) in enumerate(df.groupby(group_extra)):
                        sns.histplot(gd[yv].dropna(), ax=ax, label=str(gv), color=ep[gi%len(ep)], alpha=0.5, kde=True)
                    ax.legend(fontsize=7)
                else:
                    sns.histplot(df[yv].dropna(), ax=ax, kde=True, color="steelblue")
                ax.set_title(yv, fontweight="bold", fontsize=10); ax.set_xlabel("")
            for i in range(n_h, len(axs)): axs[i].set_visible(False)
            fig.suptitle("Histograms", fontsize=13, fontweight="bold"); fig.tight_layout()
            extra_figs.append(("Histograms", save_fig(fig).read())); plt.close(fig)
        elif pt == "Bar Chart (Mean±SD)" and yvars_extra:
            if group_extra:
                for yv in yvars_extra:
                    gd3 = df[[yv, group_extra]].dropna().groupby(group_extra)[yv].agg(["mean", "std"]).reset_index()
                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.bar(gd3[group_extra].astype(str), gd3["mean"], yerr=gd3["std"],
                           capsize=5, color=ep[:len(gd3)], alpha=0.8, edgecolor="black")
                    ax.set_xlabel(group_extra, fontsize=11); ax.set_ylabel(f"Mean ± SD ({yv})", fontsize=11)
                    ax.set_title(f"Bar Chart: {yv} by {group_extra}", fontweight="bold")
                    ax.tick_params(axis="x", rotation=40); fig.tight_layout()
                    extra_figs.append((f"Bar {yv}", save_fig(fig).read())); plt.close(fig)
            else:
                means = [df[yv].dropna().mean() for yv in yvars_extra]
                sds   = [df[yv].dropna().std()  for yv in yvars_extra]
                fig, ax = plt.subplots(figsize=(max(6, len(yvars_extra)*1.8), 5))
                ax.bar(yvars_extra, means, yerr=sds, capsize=5,
                       color=ep[:len(yvars_extra)], alpha=0.8, edgecolor="black")
                ax.set_ylabel("Mean ± SD", fontsize=11); ax.set_title("Bar Chart: Column Means", fontweight="bold")
                ax.tick_params(axis="x", rotation=30); fig.tight_layout()
                extra_figs.append(("Bar Means", save_fig(fig).read())); plt.close(fig)
        elif pt == "Paired Lines Plot":
            if len(yvars_extra) == 2:
                y1, y2 = yvars_extra; pd4 = df[[y1, y2]].dropna()
                fig, ax = plt.subplots(figsize=(7, 5))
                for _, row in pd4.iterrows():
                    ax.plot([y1, y2], [row[y1], row[y2]], "o-", color="steelblue", alpha=0.35, lw=1, ms=3)
                ax.plot([y1, y2], [pd4[y1].mean(), pd4[y2].mean()], "o-",
                        color="red", lw=3, ms=8, label="Mean", zorder=5)
                ax.set_ylabel("Value", fontsize=11); ax.set_title(f"Paired Lines: {y1} vs {y2}", fontweight="bold")
                ax.legend(); fig.tight_layout()
                extra_figs.append((f"Paired Lines {y1} vs {y2}", save_fig(fig).read())); plt.close(fig)

    progress.progress(92, "Saving results…")

    # Cache all results
    st.session_state.update({
        "res_unified_df":       pd.DataFrame(unified_rows),
        "res_posthoc_df":       pd.DataFrame(posthoc_rows),
        "res_desc_df":          desc_df,
        "res_outlier_df":       outlier_df,
        "res_pearson_corr":     pearson_corr,
        "res_spearman_corr":    spearman_corr,
        "res_corr_fig_bytes":   corr_fig_bytes,
        "res_bp_figs":          bp_figs,
        "res_extra_figs":       extra_figs,
        "res_chi2_results":     chi2_results,
        "res_friedman_results": friedman_results,
        "res_twoway_results":   twoway_results,
        "res_mlr_results":      mlr_results,
        "res_mlr_fig_bytes":    mlr_fig_bytes,
        "res_mlr_txt":          mlr_txt,
        "res_desc_fig_bytes":   desc_fig_bytes,
        "res_col_subtype":      col_subtype,
        "res_num_cols":         num_cols,
        "res_vars_desc":        vars_desc,
        "res_corr_vars":        corr_vars,
        "res_do_corr":          do_corr,
        "res_do_chi2":          do_chi2,
        "res_do_friedman":      do_friedman,
        "res_do_twoway":        do_twoway,
        "res_do_mlr":           do_mlr,
        "res_chi2_vars":        chi2_vars,
        "res_mlr_outcome":      mlr_outcome,
        "res_mlr_predictors":   mlr_predictors if mlr_predictors else [],
        "res_wide_mode":        wide_mode,
        "res_group_cols_wide":  group_cols_wide,
        "res_df_clean":         df,
        "analysis_done":        True,
    })
    progress.progress(100, "✅ Done!")

# Load cached results
unified_df       = st.session_state.get("res_unified_df",       pd.DataFrame())
posthoc_df       = st.session_state.get("res_posthoc_df",       pd.DataFrame())
desc_df          = st.session_state.get("res_desc_df",          pd.DataFrame())
outlier_df       = st.session_state.get("res_outlier_df",       pd.DataFrame())
pearson_corr     = st.session_state.get("res_pearson_corr",     pd.DataFrame())
spearman_corr    = st.session_state.get("res_spearman_corr",    pd.DataFrame())
corr_fig_bytes   = st.session_state.get("res_corr_fig_bytes",   None)
bp_figs          = st.session_state.get("res_bp_figs",          [])
extra_figs       = st.session_state.get("res_extra_figs",       [])
chi2_results     = st.session_state.get("res_chi2_results",     [])
friedman_results = st.session_state.get("res_friedman_results", [])
twoway_results   = st.session_state.get("res_twoway_results",   [])
mlr_results      = st.session_state.get("res_mlr_results",      [])
mlr_fig_bytes    = st.session_state.get("res_mlr_fig_bytes",    None)
mlr_txt          = st.session_state.get("res_mlr_txt",          "")
desc_fig_bytes   = st.session_state.get("res_desc_fig_bytes",   None)
col_subtype      = st.session_state.get("res_col_subtype",      {})
vars_desc        = st.session_state.get("res_vars_desc",        [])
corr_vars        = st.session_state.get("res_corr_vars",        [])
_do_corr         = st.session_state.get("res_do_corr",          True)
_do_chi2         = st.session_state.get("res_do_chi2",          False)
_do_friedman     = st.session_state.get("res_do_friedman",      False)
_do_twoway       = st.session_state.get("res_do_twoway",        False)
_do_mlr          = st.session_state.get("res_do_mlr",           False)
chi2_vars_disp   = st.session_state.get("res_chi2_vars",        None)
mlr_outcome_disp = st.session_state.get("res_mlr_outcome",      None)
mlr_pred_disp    = st.session_state.get("res_mlr_predictors",   [])
df_clean         = st.session_state.get("res_df_clean",         df_raw)

for _df in [unified_df, posthoc_df, desc_df, outlier_df]:
    if _df is None: _df = pd.DataFrame()

# ─────────────────────────────────────────────────────────
# RENDER RESULTS TABS
# ─────────────────────────────────────────────────────────
with st.expander("🔍 Data Preview (first 20 rows)", expanded=False):
    preview_df = df_clean if df_clean is not None else df_raw
    st.dataframe(preview_df.head(20), use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows (clean)", len(preview_df))
    m2.metric("Columns", len(preview_df.columns))
    m3.metric("Numeric cols", len(st.session_state.get("res_num_cols", [])))
    m4.metric("Outliers removed", len(outlier_df) if outlier_df is not None else 0)
    if col_subtype:
        st.markdown("**Column type overrides:**")
        for c, t in col_subtype.items():
            cls = "badge-nominal" if t == "Nominal" else "badge-ordinal"
            st.markdown(f'<span class="{cls}">{t}</span> `{c}`', unsafe_allow_html=True)

tabs = st.tabs([
    "📋 Summary", "📊 Descriptive", "📈 Correlation",
    "📦 Boxplots", "🔬 Post-hoc",
    "🔢 Chi-square", "♻️ Friedman", "⚗️ Two-way ANOVA",
    "📐 MLR", "🎨 Extra Plots", "❓ Help"
])

with tabs[0]:
    st.subheader("Statistical Test Results")
    if unified_df is not None and not unified_df.empty:
        sig = int((unified_df["Significance"] != "ns").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Tests run", len(unified_df))
        c2.metric("Significant", sig)
        c3.metric("Non-significant", len(unified_df)-sig)
        st.dataframe(unified_df, use_container_width=True, height=420)
        st.download_button("📥 Download CSV",
            unified_df.to_csv(index=False).encode(), "summary.csv", "text/csv")
    else:
        st.info("No tests performed — check variable selections.")

with tabs[1]:
    st.subheader("Descriptive Statistics + Normality")
    if desc_df is not None and not desc_df.empty:
        st.dataframe(desc_df, use_container_width=True)
        st.download_button("📥 Download CSV",
            desc_df.to_csv(index=False).encode(), "descriptive.csv", "text/csv")
        if desc_fig_bytes:
            st.subheader("Distribution Overview")
            st.image(desc_fig_bytes)
            st.download_button("📥 Download PNG", desc_fig_bytes, "distributions.png", "image/png")
    else:
        st.info("No numeric variables selected.")

with tabs[2]:
    if _do_corr and len(corr_vars) > 1:
        c1c, c2c = st.columns(2)
        with c1c:
            st.subheader("Pearson Correlation")
            if pearson_corr is not None and not pearson_corr.empty:
                st.dataframe(pearson_corr.style.format("{:.3f}").background_gradient(cmap="coolwarm", axis=None))
        with c2c:
            st.subheader("Spearman Correlation")
            if spearman_corr is not None and not spearman_corr.empty:
                st.dataframe(spearman_corr.style.format("{:.3f}").background_gradient(cmap="coolwarm", axis=None))
        if corr_fig_bytes:
            st.image(corr_fig_bytes)
            st.download_button("📥 Heatmap PNG", corr_fig_bytes, "heatmap.png", "image/png")
    else:
        st.info("Select ≥2 numeric/wide columns and enable correlation.")

with tabs[3]:
    if bp_figs:
        for name, fig_bytes in bp_figs:
            st.subheader(name); st.image(fig_bytes)
            sn = name.replace(" ", "_").replace("—", "_")
            st.download_button("📥 Download", fig_bytes, f"bp_{sn}.png", "image/png", key=f"bp_{sn}")
    else:
        st.info("Select categorical grouping variables or use wide-format mode.")

with tabs[4]:
    if posthoc_df is not None and not posthoc_df.empty:
        st.subheader("Post-hoc Comparisons")
        st.dataframe(posthoc_df, use_container_width=True)
        st.download_button("📥 Download CSV",
            posthoc_df.to_csv(index=False).encode(), "posthoc.csv", "text/csv")
    else:
        st.info("No post-hoc (no significant omnibus test or only 2 groups).")

with tabs[5]:
    st.subheader("Chi-square Test of Independence")
    if chi2_results:
        st.dataframe(pd.DataFrame(chi2_results), use_container_width=True)
        if chi2_vars_disp and len(chi2_vars_disp) == 2 and df_clean is not None:
            c1v, c2v = chi2_vars_disp
            if c1v in df_clean.columns and c2v in df_clean.columns:
                ct = pd.crosstab(df_clean[c1v], df_clean[c2v])
                st.subheader("Contingency Table"); st.dataframe(ct, use_container_width=True)
                fig_ct, ax_ct = plt.subplots(figsize=(max(6, len(ct.columns)), max(5, len(ct)*0.6)))
                sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax_ct, linewidths=0.5)
                ax_ct.set_title(f"Contingency: {c1v} × {c2v}", fontweight="bold")
                fig_ct.tight_layout()
                ct_bytes = save_fig(fig_ct).read(); plt.close(fig_ct)
                st.image(ct_bytes)
    elif _do_chi2:
        st.info("Select exactly 2 categorical columns in the sidebar.")
    else:
        st.info("Enable Chi-square in ⑦ Advanced Tests.")

with tabs[6]:
    st.subheader("Friedman Test")
    if friedman_results:
        st.dataframe(pd.DataFrame(friedman_results), use_container_width=True)
        st.info("Significant → use Dunn's post-hoc for pairwise comparisons.")
    elif _do_friedman:
        st.info("Use wide format with ≥3 columns selected.")
    else:
        st.info("Enable Friedman in ⑦ Advanced Tests.")

with tabs[7]:
    st.subheader("Two-way ANOVA")
    if twoway_results:
        st.dataframe(pd.DataFrame(twoway_results), use_container_width=True)
        st.caption("C(F1): Factor 1 | C(F2): Factor 2 | C(F1):C(F2): Interaction")
    elif _do_twoway:
        st.info("Select outcome variable(s) and 2 categorical factors.")
    else:
        st.info("Enable Two-way ANOVA in ⑦ Advanced Tests.")

with tabs[8]:
    st.subheader("Multiple Linear Regression")
    if mlr_results:
        if mlr_outcome_disp and mlr_pred_disp and df_clean is not None:
            try:
                sub_r2 = df_clean[[mlr_outcome_disp] + mlr_pred_disp].dropna()
                r2v = round(float(sm.OLS(sub_r2[mlr_outcome_disp],
                    sm.add_constant(sub_r2[mlr_pred_disp])).fit().rsquared), 4)
                st.metric("R² (Model fit)", r2v)
            except: pass
        st.dataframe(pd.DataFrame(mlr_results), use_container_width=True)
        st.download_button("📥 Download CSV",
            pd.DataFrame(mlr_results).to_csv(index=False).encode(), "mlr.csv", "text/csv")
        if mlr_fig_bytes:
            st.subheader("Diagnostic Plots"); st.image(mlr_fig_bytes)
            st.download_button("📥 Download PNG", mlr_fig_bytes, "mlr_diag.png", "image/png")
        if mlr_txt:
            with st.expander("Full OLS summary"): st.text(mlr_txt)
    elif _do_mlr:
        st.info("Select outcome + predictors in the sidebar.")
    else:
        st.info("Enable MLR in ⑦ Advanced Tests.")

with tabs[9]:
    if extra_figs:
        for name, fig_bytes in extra_figs:
            st.subheader(name); st.image(fig_bytes)
            sn = name.replace(" ", "_").replace("~", "_").replace("—", "_")
            st.download_button("📥 Download", fig_bytes, f"{sn}.png", "image/png", key=f"ex_{sn}")
    else:
        st.info("Select plot types and X/Y variables in ⑧ Extra Plots.")

with tabs[10]:
    st.markdown("""
## 📖 Help & User Guide

| Format | Triggers |
|--------|---------|
| Wide, 2 cols | t-test + Mann-Whitney U |
| Wide, ≥3 cols | ANOVA + Kruskal-Wallis + post-hoc |
| Long, 2 groups | t-test + Mann-Whitney U |
| Long, ≥3 groups | ANOVA + Kruskal-Wallis + post-hoc |

### Significance stars
| Symbol | p-value |
|--------|---------|
| ns | ≥ 0.05 |
| * | < 0.05 |
| ** | < 0.01 |
| *** | < 0.001 |

### Effect size benchmarks
| Metric | Small | Medium | Large |
|--------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| Eta-squared η² | 0.01 | 0.06 | 0.14 |
| Rank-biserial r | 0.1 | 0.3 | 0.5 |

---
✉️ **Contact:** [Syed Ishaq on LinkedIn](https://www.linkedin.com/in/syed-ishaq-893052285/)
""")

if outlier_df is not None and not outlier_df.empty:
    with st.expander(f"📋 Outlier Report — {len(outlier_df)} rows removed", expanded=False):
        st.dataframe(outlier_df, use_container_width=True)
        st.download_button("📥 Download",
            outlier_df.to_csv(index=False).encode(), "outliers.csv", "text/csv")

st.success("✅ Analysis complete! Adjust settings and click **🚀 Run Analysis** to refresh.")
