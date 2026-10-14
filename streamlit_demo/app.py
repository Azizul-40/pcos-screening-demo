import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PCOS Risk Screening — Prototype",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load the locked, frozen Table-9 (Merged, 20% split) models.
# Nothing here is re-trained — the app only loads the frozen pipeline and runs
# it on one new input (same as Step 44D's real-row latency measurement).
# The 13 features / order are read off the frozen pipeline's input schema.
# Paths are resolved relative to THIS FILE so Streamlit Cloud works even when
# app.py lives in a subfolder.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "SVM": os.path.join(BASE_DIR, "models", "svm_merged20.joblib"),
    "KNN": os.path.join(BASE_DIR, "models", "knn_merged20.joblib"),
}

FEATURE_ORDER = [
    "Regular_menstruation", "Weight", "Height", "Age", "Regular_physical_activity",
    "Menstrual_period_duration", "Frequent_fast_food", "Weight_gain", "ABO_blood_group",
    "Skin_hyperpigmentation", "Hair_loss", "Acne", "Hirsutism",
]

BLOOD_GROUPS = {
    "A+": 11, "A-": 12, "B+": 13, "B-": 14,
    "O+": 15, "O-": 16, "AB+": 17, "AB-": 18,
}

# Yes/No question keys, grouped by section (used for per-section pill colours)
LIFE_KEYS = ["reg_cycle", "reg_exercise", "fast_food", "weight_gain"]
SYM_KEYS = ["skin", "hair_loss", "acne", "hirsutism"]


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Bilingual text (বাংলা / English)
# ---------------------------------------------------------------------------
TXT = {
    "bn": {
        "hero_title": "🩺 পিসিওএস (PCOS) ঝুঁকি স্ক্রিনিং — রিসার্চ প্রোটোটাইপ",
        "hero_sub": "একটা academic research prototype — ক্লিনিক্যাল ডায়াগনসিস টুল না। "
                    "locked Table-9 (Merged, 20% split) pipeline-এর উপর বানানো।",
        "sidebar_model_header": "⚙️ মডেল সেটিংস",
        "sidebar_model_label": "Classifier বেছে নিন",
        "sidebar_source": "📦 উৎস",
        "sidebar_source_text": "<code>Pcos Rerun 01</code> Drive প্রজেক্ট — frozen Table 9 (Merged, split=20%) "
                                "models, কোনো নতুন train করা হয়নি।",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।",
        "two_model_title": "ℹ️ দুটো মডেল (SVM, KNN) কেন?",
        "two_model_note": "দুই ভিন্ন algorithm একই ডেটায় কেমন আচরণ করে তুলনা করার জন্য। probability "
                           "আলাদা হতে পারে (SVM smooth, KNN ধাপে ধাপে) — আসল বিষয় হলো "
                           "<b>predicted ঝুঁকি</b> মেলে কিনা।",
        "sec_basic": "🧍 মৌলিক তথ্য",
        "age": "বয়স (বছর)",
        "weight": "ওজন (kg)",
        "height": "উচ্চতা (cm)",
        "cycle_len": "মাসিক (দিন)",
        "blood_group": "রক্তের গ্রুপ",
        "sec_lifestyle": "🏃 জীবনযাত্রা",
        "reg_cycle": "মাসিক নিয়মিত?",
        "reg_exercise": "নিয়মিত ব্যায়াম?",
        "fast_food": "ঘন ঘন ফাস্ট ফুড?",
        "weight_gain": "ওজন বৃদ্ধি?",
        "sec_symptoms": "✨ ত্বক / চুল লক্ষণ",
        "skin": "ত্বক কালচে",
        "hair_loss": "চুল পড়া",
        "acne": "ব্রণ",
        "hirsutism": "অতিরিক্ত লোম",
        "yes": "হ্যাঁ",
        "no": "না",
        "predict_btn": "🔍 পূর্বাভাস দিন (Predict)",
        "placeholder": "তথ্য দিয়ে <b>Predict</b> চাপুন — ফলাফল এখানে দেখাবে।",
        "result_high": "PCOS-এর ঝুঁকি বেশি বলে অনুমান",
        "result_low": "PCOS-এর ঝুঁকি কম বলে অনুমান",
        "pred_class": "Predicted class",
        "class_high": "PCOS (1)",
        "class_low": "No PCOS (0)",
        "prob_title": "PCOS সম্ভাবনা",
        "metric_latency": "Prediction সময়",
        "metric_model": "মডেল",
        "metric_feats": "ব্যবহৃত feature",
        "footer_warning": "⚠️ গবেষণা প্রোটোটাইপ — ডাক্তারি নির্ণয় না। PCOS সন্দেহ হলে gynecologist-এর "
                           "সাথে পরামর্শ করুন।",
    },
    "en": {
        "hero_title": "🩺 PCOS Risk Screening — Research Prototype",
        "hero_sub": "An academic research prototype — not a clinical diagnosis tool. Built on our "
                    "locked Table-9 (Merged, 20% split) pipeline.",
        "sidebar_model_header": "⚙️ Model Settings",
        "sidebar_model_label": "Choose classifier",
        "sidebar_source": "📦 Source",
        "sidebar_source_text": "<code>Pcos Rerun 01</code> Drive project — frozen Table 9 (Merged, split=20%) "
                                "models. No new training was done.",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "Built for research purposes only — not medical advice or diagnosis.",
        "two_model_title": "ℹ️ Why two models (SVM, KNN)?",
        "two_model_note": "To compare how two algorithms behave on the same data. Raw probability can "
                           "differ (SVM smooth, KNN stepped) — what matters is whether the "
                           "<b>predicted risk class</b> agrees.",
        "sec_basic": "🧍 Basic Information",
        "age": "Age (years)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "cycle_len": "Cycle length (days)",
        "blood_group": "Blood group",
        "sec_lifestyle": "🏃 Lifestyle",
        "reg_cycle": "Regular periods?",
        "reg_exercise": "Regular exercise?",
        "fast_food": "Frequent fast food?",
        "weight_gain": "Recent weight gain?",
        "sec_symptoms": "✨ Skin / Hair Symptoms",
        "skin": "Skin darkening",
        "hair_loss": "Hair loss",
        "acne": "Acne",
        "hirsutism": "Hirsutism",
        "yes": "Yes",
        "no": "No",
        "predict_btn": "🔍 Predict",
        "placeholder": "Fill in the details and press <b>Predict</b> — the result appears here.",
        "result_high": "Elevated PCOS risk predicted",
        "result_low": "Low PCOS risk predicted",
        "pred_class": "Predicted class",
        "class_high": "PCOS (1)",
        "class_low": "No PCOS (0)",
        "prob_title": "PCOS PROBABILITY",
        "metric_latency": "Latency",
        "metric_model": "Model",
        "metric_feats": "Features used",
        "footer_warning": "⚠️ Research prototype — not a medical diagnosis. Please consult a gynecologist "
                           "if you suspect PCOS.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "bn"

# ---------------------------------------------------------------------------
# Styling
#  - Streamlit's own top-right toolbar (Share / star / edit / GitHub / ⋮) is KEPT,
#    but scaled up so it is easy to see and click, with hover highlight
#  - compact, roomy-but-fits-one-screen layout (no page scroll)
#  - Yes/No radios restyled as pill buttons, coloured per section, with hover
#  - Predict / pills / inputs all have hover effects
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ---------- header / toolbar (kept, made bigger) ---------- */
header[data-testid="stHeader"] { background: transparent; height: 3.6rem; }
[data-testid="stToolbar"] {
    transform: scale(1.3); transform-origin: top right; margin-right: 1.2rem; margin-top: 0.15rem;
}
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a {
    border-radius: 10px !important; transition: background .15s ease, transform .15s ease;
}
[data-testid="stToolbar"] button:hover, [data-testid="stToolbar"] a:hover {
    background: rgba(20,184,166,.25) !important; transform: translateY(-1px);
}
[data-testid="stToolbar"] svg { width: 1.15rem; height: 1.15rem; }
footer { visibility: hidden; }

.main .block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 3rem !important; padding-bottom: 0.4rem !important; max-width: 1450px;
}
html, body, [class*="css"] { font-size: 16px; }
[data-testid="stVerticalBlock"] { gap: 0.75rem; }

.ctl-label { font-weight: 700; font-size: 1rem; margin-bottom: 0.25rem; }

/* ---------- hero ---------- */
.hero {
    background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%);
    padding: 0.75rem 1.4rem; border-radius: 12px; color: #fff;
    box-shadow: 0 4px 14px rgba(15,118,110,0.35);
}
.hero h1 { margin: 0; font-size: 1.45rem; line-height: 1.25; color: #fff; padding: 0; }
.hero p  { margin: 0.2rem 0 0 0; font-size: 0.92rem; line-height: 1.35; color: #e6fffb; }

/* ---------- section cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px; padding: 0.55rem 1.05rem 0.8rem 1.05rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: 0.7rem; }
.sec-head { display: inline-block; font-weight: 700; font-size: 1.02rem; padding: 0.15rem 1rem;
    border-radius: 999px; color: #fff; }
.sec-basic { background: linear-gradient(90deg, #0d9488, #14b8a6); }
.sec-life  { background: linear-gradient(90deg, #d97706, #f59e0b); }
.sec-sym   { background: linear-gradient(90deg, #be185d, #ec4899); }

/* ---------- field labels: bigger, bolder ---------- */
[data-testid="stMain"] label[data-testid="stWidgetLabel"] p,
.main label[data-testid="stWidgetLabel"] p {
    font-size: 1.05rem !important; font-weight: 700 !important; margin-bottom: 0.1rem; opacity: 1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div { min-height: 2.5rem; font-size: 1.05rem; }
div[data-testid="stNumberInput"] > div > div, div[data-baseweb="select"] > div {
    transition: border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stNumberInput"] > div > div:hover, div[data-baseweb="select"] > div:hover {
    border-color: #14b8a6 !important; box-shadow: 0 0 0 1px #14b8a6;
}

/* ---------- keep every field row on one level (inputs bottom-aligned) ---------- */
:is(.st-key-card_basic, .st-key-card_life, .st-key-card_sym) [data-testid="stHorizontalBlock"] { align-items: flex-end; }
:is(.st-key-card_basic, .st-key-card_life, .st-key-card_sym) label[data-testid="stWidgetLabel"] { min-height: 1.7rem; }

/* ---------- segmented controls (Yes/No, language, model): native widget, restyled ---------- */
[data-testid="stButtonGroup"] { width: 100%; }
[data-testid="stButtonGroup"] > div { display: flex; width: 100%; gap: 0.5rem; flex-wrap: nowrap; }
[data-testid="stButtonGroup"] button {
    flex: 1; min-height: 2.6rem; border-radius: 9px !important;
    font-weight: 700 !important;
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease, border-color .15s ease;
}
[data-testid="stButtonGroup"] button p, [data-testid="stButtonGroup"] button span { font-size: 1.05rem !important; font-weight: 700 !important; }
[data-testid="stButtonGroup"] button:hover {
    transform: translateY(-2px); border-color: #14b8a6 !important; filter: brightness(1.18);
    box-shadow: 0 5px 14px rgba(0,0,0,.28);
}
/* selected: teal by default, amber in Lifestyle, pink in Symptoms */
[data-testid="stBaseButton-segmented_controlActive"], button[kind="segmented_controlActive"] {
    background: linear-gradient(90deg, #0d9488, #14b8a6) !important; color: #fff !important; border-color: #5eead4 !important;
}
.st-key-card_life [data-testid="stBaseButton-segmented_controlActive"], .st-key-card_life button[kind="segmented_controlActive"] {
    background: linear-gradient(90deg, #d97706, #f59e0b) !important; border-color: #fbbf24 !important; }
.st-key-card_sym [data-testid="stBaseButton-segmented_controlActive"], .st-key-card_sym button[kind="segmented_controlActive"] {
    background: linear-gradient(90deg, #be185d, #ec4899) !important; border-color: #f472b6 !important; }
/* selected "No" (second button) in the two symptom/lifestyle cards: slate */
:is(.st-key-card_life, .st-key-card_sym) [data-testid="stBaseButton-segmented_controlActive"]:nth-of-type(2),
:is(.st-key-card_life, .st-key-card_sym) button[kind="segmented_controlActive"]:nth-of-type(2) {
    background: #3b4560 !important; border-color: #6b7aa0 !important; }
[data-testid="stBaseButton-segmented_controlActive"] *, button[kind="segmented_controlActive"] * { color: #fff !important; }

:is(.st-key-lang, .st-key-model_label) button p, :is(.st-key-lang, .st-key-model_label) button span { font-size: 1.2rem !important; }

/* ---------- Predict button ---------- */
.stButton > button {
    border-radius: 12px; padding: 0.7rem 1rem; font-weight: 700; font-size: 1.15rem;
    background: linear-gradient(90deg, #0f766e, #0891b2); color: #fff; border: none; width: 100%;
    box-shadow: 0 4px 14px rgba(8,145,178,.40);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton > button:hover {
    color: #fff; transform: translateY(-3px); filter: brightness(1.15);
    box-shadow: 0 10px 24px rgba(8,145,178,.60);
}
.stButton > button:active { transform: translateY(0); filter: brightness(.95); }

/* ---------- result band ---------- */
.res-card { border-radius: 12px; padding: 0.6rem 1.1rem; height: 132px; box-sizing: border-box; overflow: hidden; }
.verdict { display: flex; gap: 0.9rem; align-items: center; color: #fff; }
.verdict .big { font-size: 2.5rem; line-height: 1; }
.verdict h2 { margin: 0; padding: 0; font-size: 1.3rem; line-height: 1.25; color: #fff; }
.verdict small { color: #f1f5f9; font-size: 0.9rem; }
.v-high { background: linear-gradient(120deg, #7f1d1d 0%, #dc2626 55%, #f87171 100%);
          border: 1px solid #fca5a5; box-shadow: 0 6px 18px rgba(220,38,38,.35); }
.v-low  { background: linear-gradient(120deg, #064e3b 0%, #059669 55%, #34d399 100%);
          border: 1px solid #6ee7b7; box-shadow: 0 6px 18px rgba(16,185,129,.35); }
.gauge { background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.35); }
.gauge .t { font-size: 0.88rem; font-weight: 700; opacity: .85; }
.gauge .v { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.vr { color: #f87171; } .vg { color: #34d399; }
.bar { position: relative; height: 14px; border-radius: 7px; margin-top: 0.7rem;
       background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); }
.mk { position: absolute; top: -5px; width: 4px; height: 24px; background: #fff; border-radius: 2px;
      box-shadow: 0 0 6px #000; }
.th { position: absolute; top: -2px; height: 18px; border-left: 2px dashed rgba(255,255,255,.65); }
.scale { display: flex; justify-content: space-between; font-size: 0.78rem; opacity: .7; margin-top: 0.45rem; }
.mets { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; height: 132px; }
.m { background: rgba(20,184,166,.10); border: 1px solid rgba(20,184,166,.45); border-radius: 10px;
     padding: 0.55rem 0.8rem; display: flex; flex-direction: column; justify-content: center; transition: transform .15s ease, box-shadow .15s ease; }
.m:hover { transform: translateY(-2px); box-shadow: 0 6px 14px rgba(20,184,166,.25); }
.m .a { font-size: 0.88rem; font-weight: 600; opacity: .85; }
.m .b { font-size: 1.4rem; font-weight: 700; color: #2dd4bf; margin-top: 0.15rem; }
.placeholder { border: 2px dashed rgba(20,184,166,.55); background: rgba(20,184,166,.07);
    border-radius: 12px; text-align: center; padding: 3rem 1rem; font-size: 1.05rem; }
.foot { font-size: 0.85rem; opacity: .75; text-align: center; margin-top: 0.3rem; padding-bottom: 0.4rem; }

/* ---------- sidebar open / close buttons: always visible ---------- */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
    visibility: visible !important; opacity: 1 !important; z-index: 999991;
}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button {
    background: linear-gradient(90deg, #0f766e, #0891b2) !important; color: #fff !important;
    border-radius: 10px !important; box-shadow: 0 4px 12px rgba(8,145,178,.5);
    transition: transform .15s ease, filter .15s ease;
}
[data-testid="stExpandSidebarButton"]:hover, [data-testid="stSidebarCollapsedControl"] button:hover, [data-testid="collapsedControl"] button:hover {
    transform: scale(1.12); filter: brightness(1.2);
}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg { color: #fff !important; fill: #fff !important; }
[data-testid="stSidebarCollapseButton"] button { background: rgba(255,255,255,.15); border-radius: 10px; }
[data-testid="stSidebarCollapseButton"] button:hover { background: rgba(20,184,166,.5); }

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3b3a 0%, #0c4a5a 55%, #14213d 100%);
    border-right: 2px solid #14b8a6;
}
section[data-testid="stSidebar"][aria-expanded="true"] { min-width: 330px; }
section[data-testid="stSidebar"] * { color: #e6fffb; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
section[data-testid="stSidebar"] h3 { font-size: 1.45rem; margin: 0.3rem 0 0.4rem 0; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.7rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.40);
    border-radius: 999px; padding: 0.35rem 1.15rem; transition: transform .15s ease, background .15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(20,184,166,.35); transform: translateY(-2px); }
section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 1.3rem !important; font-weight: 700; }
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p { font-size: 1.2rem !important; font-weight: 700; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    min-height: 3.2rem; background: rgba(255,255,255,0.13); border: 1px solid #14b8a6;
    transition: box-shadow .15s ease; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover { box-shadow: 0 0 0 2px #14b8a6; }
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #ffffff !important; font-size: 1.35rem !important; font-weight: 700 !important; }
.side-card { border-radius: 12px; padding: 0.65rem 0.85rem; margin-top: 0.7rem; font-size: 0.98rem; line-height: 1.45; }
.side-card b { font-size: 1.05rem; }
.side-src  { background: rgba(20,184,166,0.20); border-left: 5px solid #14b8a6; }
.side-warn { background: rgba(245,158,11,0.20); border-left: 5px solid #f59e0b; }
.side-note { background: rgba(236,72,153,0.17); border-left: 5px solid #ec4899; }
</style>
"""


def _sel(keys):
    # :is() applies the rest of the selector to EVERY key (a bare comma list would only prefix the last one)
    return ":is(" + ", ".join(f".st-key-{k}" for k in keys) + ")"


st.markdown(
    _CSS.replace("__PILL_SEL__", _sel(LIFE_KEYS + SYM_KEYS))
        .replace("__LIFE_SEL__", _sel(LIFE_KEYS))
        .replace("__SYM_SEL__", _sel(SYM_KEYS)),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Language / model state. The controls live in the TOP BAR of the page (always visible),
# not in the sidebar, so they never disappear if the sidebar gets collapsed.
# ---------------------------------------------------------------------------
st.session_state.setdefault("lang", "bn")
st.session_state.setdefault("model_label", "SVM")
T = TXT[st.session_state.lang]


def _restore(key):
    """segmented_control returns None when the active option is clicked again; put the last value back."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get("_last_" + key)


def seg(container, label, options, key, default, fmt=None, hide_label=False):
    """Single-select segmented control that can never end up empty."""
    st.session_state.setdefault(key, default)
    st.session_state.setdefault("_last_" + key, st.session_state[key] if st.session_state[key] is not None else default)
    val = container.segmented_control(
        label, options, selection_mode="single", key=key, format_func=fmt or (lambda v: v),
        on_change=_restore, args=(key,), label_visibility="collapsed" if hide_label else "visible",
    )
    if val is None:
        val = st.session_state["_last_" + key]
    st.session_state["_last_" + key] = val
    return val


# Sidebar: information only (source / disclaimer / why two models)
with st.sidebar:
    st.markdown(f"### {T['sidebar_model_header']}")
    st.markdown(
        f'<div class="side-card side-note"><b>{T["two_model_title"]}</b><br>{T["two_model_note"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="side-card side-src"><b>{T["sidebar_source"]}</b><br>{T["sidebar_source_text"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="side-card side-warn"><b>{T["sidebar_disclaimer"]}</b><br>{T["sidebar_disclaimer_text"]}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
c_hero, c_lang, c_model = st.columns([2.7, 1.25, 1.05], gap="medium", vertical_alignment="center")
c_hero.markdown(
    f"""<div class="hero"><h1>{T['hero_title']}</h1><p>{T['hero_sub']}</p></div>""",
    unsafe_allow_html=True,
)
with c_lang:
    st.markdown('<div class="ctl-label">🌐 Language / ভাষা</div>', unsafe_allow_html=True)
    seg(st, "Language", ["bn", "en"], "lang", "bn", fmt=lambda v: "বাংলা" if v == "bn" else "English", hide_label=True)
with c_model:
    st.markdown(f'<div class="ctl-label">{T["sidebar_model_label"]}</div>', unsafe_allow_html=True)
    model_label = seg(st, "Classifier", list(MODEL_FILES.keys()), "model_label", "SVM", hide_label=True)
model = load_model(MODEL_FILES[model_label])


def yes_no(col, name, default_yes):
    """Yes/No segmented control. Returns 1 for Yes, 0 for No (stable option values, localised labels)."""
    return seg(col, T[name], [1, 0], name, 1 if default_yes else 0,
               fmt=lambda v: T["yes"] if v == 1 else T["no"])


# Row 1 — basic information (5 fields)
with st.container(border=True, key="card_basic"):
    st.markdown(f'<span class="sec-head sec-basic">{T["sec_basic"]}</span>', unsafe_allow_html=True)
    b1, b2, b3, b4, b5 = st.columns(5, gap="large")
    age = b1.number_input(T["age"], min_value=10, max_value=55, value=25, step=1)
    weight = b2.number_input(T["weight"], min_value=30.0, max_value=130.0, value=60.0, step=0.5)
    height = b3.number_input(T["height"], min_value=130.0, max_value=190.0, value=157.0, step=0.5)
    cycle_len = b4.number_input(T["cycle_len"], min_value=1, max_value=15, value=5, step=1)
    blood_group = b5.selectbox(T["blood_group"], list(BLOOD_GROUPS.keys()), index=4)

# Row 2 — lifestyle | symptoms, side by side, 2x2 pills each
col_life, col_sym = st.columns(2, gap="medium")

with col_life:
    with st.container(border=True, key="card_life"):
        st.markdown(f'<span class="sec-head sec-life">{T["sec_lifestyle"]}</span>', unsafe_allow_html=True)
        a1, a2 = st.columns(2, gap="large")
        reg_cycle = yes_no(a1, "reg_cycle", False)
        reg_exercise = yes_no(a2, "reg_exercise", True)
        a3, a4 = st.columns(2, gap="large")
        fast_food = yes_no(a3, "fast_food", False)
        weight_gain = yes_no(a4, "weight_gain", False)

with col_sym:
    with st.container(border=True, key="card_sym"):
        st.markdown(f'<span class="sec-head sec-sym">{T["sec_symptoms"]}</span>', unsafe_allow_html=True)
        s1, s2 = st.columns(2, gap="large")
        skin = yes_no(s1, "skin", False)
        hair_loss = yes_no(s2, "hair_loss", False)
        s3, s4 = st.columns(2, gap="large")
        acne = yes_no(s3, "acne", False)
        hirsutism = yes_no(s4, "hirsutism", False)

# Predict button — centred under the features
_, mid, _ = st.columns([1, 1.3, 1])
predict_clicked = mid.button(T["predict_btn"], type="primary", use_container_width=True)

# Result band — directly under the button
r1, r2, r3 = st.columns([1.15, 1.5, 1.35], gap="medium")

if not predict_clicked:
    st.markdown(f'<div class="placeholder">{T["placeholder"]}</div>', unsafe_allow_html=True)
else:
    row = pd.DataFrame([{
        "Regular_menstruation": reg_cycle,
        "Weight": weight,
        "Height": height,
        "Age": age,
        "Regular_physical_activity": reg_exercise,
        "Menstrual_period_duration": cycle_len,
        "Frequent_fast_food": fast_food,
        "Weight_gain": weight_gain,
        "ABO_blood_group": BLOOD_GROUPS[blood_group],
        "Skin_hyperpigmentation": skin,
        "Hair_loss": hair_loss,
        "Acne": acne,
        "Hirsutism": hirsutism,
    }])[FEATURE_ORDER]

    t0 = time.perf_counter()
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    pcos_prob = float(proba[1] if len(proba) > 1 else proba[0])
    pct = pcos_prob * 100
    marker = min(max(pct, 1.0), 99.0)

    high = pred == 1
    v_cls = "v-high" if high else "v-low"
    v_icon = "⚠️" if high else "✅"
    v_text = T["result_high"] if high else T["result_low"]
    v_class = T["class_high"] if high else T["class_low"]
    num_cls = "vr" if high else "vg"

    r1.markdown(
        f"""<div class="res-card verdict {v_cls}"><div class="big">{v_icon}</div>
        <div><h2>{v_text}</h2><small>{T['pred_class']}: {v_class} · {model_label}</small></div></div>""",
        unsafe_allow_html=True,
    )
    r2.markdown(
        f"""<div class="res-card gauge"><div class="t">{T['prob_title']}</div>
        <div class="v {num_cls}">{pct:.1f}%</div>
        <div class="bar"><div class="th" style="left:50%"></div><div class="mk" style="left:calc({marker:.1f}% - 2px)"></div></div>
        <div class="scale"><span>0%</span><span>50%</span><span>100%</span></div></div>""",
        unsafe_allow_html=True,
    )
    r3.markdown(
        f"""<div class="mets">
        <div class="m"><div class="a">{T['metric_model']}</div><div class="b">{model_label}</div></div>
        <div class="m"><div class="a">{T['metric_latency']}</div><div class="b">{latency_ms:.1f} ms</div></div>
        <div class="m"><div class="a">{T['metric_feats']}</div><div class="b">{len(FEATURE_ORDER)}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown(f'<div class="foot">{T["footer_warning"]}</div>', unsafe_allow_html=True)
