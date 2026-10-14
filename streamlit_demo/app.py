import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PCOS Risk Screening — Prototype",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",   # we draw our own side panel (see below) — it can always be closed AND re-opened
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

# (key, default answer: 1 = Yes, 0 = No)
# Test-set numbers from the Table-9 report (Merged, 20% split, new code "Pcos Rerun 01")
PERF = {"SVM": {"acc": 86.57, "rec": 60.98}, "KNN": {"acc": 84.08, "rec": 53.66}}
BORDER_LO, BORDER_HI = 30.0, 70.0   # display band only — the model's own 0/1 decision is unchanged

LIFESTYLE_Q = [("reg_cycle", 0), ("reg_exercise", 1), ("fast_food", 0), ("weight_gain", 0)]
SYMPTOM_Q = [("skin", 0), ("hair_loss", 0), ("acne", 0), ("hirsutism", 0)]


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Bilingual text (বাংলা / English)
# ---------------------------------------------------------------------------
TXT = {
    "bn": {
        "hero_title": "🩺 PCOS Risk Screening — Research Prototype",
        "hero_sub": "একটা academic research prototype — ক্লিনিক্যাল ডায়াগনসিস টুল না।",
        "panel_title": "⚙️ সেটিংস",
        "lang_label": "ভাষা / Language",
        "sidebar_model_label": "Classifier (মডেল)",
        "sidebar_source": "📦 উৎস",
        "sidebar_source_text": "<code>Pcos Rerun 01</code> Drive প্রজেক্টের frozen মডেল — কোনো নতুন train করা হয়নি।",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।",
        "two_model_title": "ℹ️ দুটো মডেল (SVM, KNN) কেন?",
        "two_model_note": "দুই ভিন্ন algorithm একই ডেটায় কেমন আচরণ করে তুলনা করার জন্য। probability "
                           "আলাদা হতে পারে (SVM smooth, KNN ধাপে ধাপে) — আসল বিষয় হলো "
                           "<b class=\"inl\">predicted ঝুঁকি</b> মেলে কিনা।",
        "close_tip": "মেনু বন্ধ করুন",
        "open_panel": "☰ মেনু",
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
        "result_mid": "মাঝামাঝি / borderline — নিশ্চিত নয়",
        "mid_hint": "মডেলের সিদ্ধান্ত 50%-এর কাছাকাছি; চিকিৎসকের পরামর্শ নিন",
        "perf_title": "📊 মডেলের নির্ভুলতা",
        "perf_text": "Test accuracy <b class=\"inl\">{acc:.1f}%</b>, PCOS recall <b class=\"inl\">{rec:.0f}%</b> — প্রতি ১০ জন PCOS রোগীর মধ্যে প্রায় {miss} জন ধরা পড়ে না। তাই ফলাফল ভুলও হতে পারে।",
        "pred_class": "Predicted class",
        "class_high": "PCOS (1)",
        "class_low": "No PCOS (0)",
        "prob_title": "মডেলের আত্মবিশ্বাস (PCOS)",
        "metric_latency": "Prediction সময়",
        "metric_model": "মডেল",
        "metric_feats": "ব্যবহৃত feature",
        "footer_warning": "⚠️ গবেষণা প্রোটোটাইপ — ডাক্তারি নির্ণয় না। PCOS সন্দেহ হলে gynecologist-এর "
                           "সাথে পরামর্শ করুন।",
    },
    "en": {
        "hero_title": "🩺 PCOS Risk Screening — Research Prototype",
        "hero_sub": "An academic research prototype — not a clinical diagnosis tool.",
        "panel_title": "⚙️ Settings",
        "lang_label": "Language / ভাষা",
        "sidebar_model_label": "Classifier (model)",
        "sidebar_source": "📦 Source",
        "sidebar_source_text": "<code>Pcos Rerun 01</code> Drive project — frozen models. No new training was done.",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "Built for research purposes only — not medical advice or diagnosis.",
        "two_model_title": "ℹ️ Why two models (SVM, KNN)?",
        "two_model_note": "To compare how two algorithms behave on the same data. Raw probability can "
                           "differ (SVM smooth, KNN stepped) — what matters is whether the "
                           "<b class=\"inl\">predicted risk class</b> agrees.",
        "close_tip": "Close menu",
        "open_panel": "☰ Menu",
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
        "result_mid": "Borderline — not clear-cut",
        "mid_hint": "Model output is close to 50% — please consult a doctor",
        "perf_title": "📊 Model accuracy",
        "perf_text": "Test accuracy <b class=\"inl\">{acc:.1f}%</b>, PCOS recall <b class=\"inl\">{rec:.0f}%</b> — about {miss} in 10 PCOS cases are missed, so the result can be wrong.",
        "pred_class": "Predicted class",
        "class_high": "PCOS (1)",
        "class_low": "No PCOS (0)",
        "prob_title": "MODEL CONFIDENCE (PCOS)",
        "metric_latency": "Latency",
        "metric_model": "Model",
        "metric_feats": "Features used",
        "footer_warning": "⚠️ Research prototype — not a medical diagnosis. Please consult a gynecologist "
                           "if you suspect PCOS.",
    },
}

# ---------------------------------------------------------------------------
# Session state (plain keys — never lost when the side panel is hidden)
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("lang", "bn")
ss.setdefault("model_val", "SVM")
for _k, _d in LIFESTYLE_Q + SYMPTOM_Q:
    ss.setdefault(_k, _d)

T = TXT[ss.lang]


def _set_lang(v):
    ss.lang = v


def _set_model(v):
    ss.model_val = v


def _set_ans(name, v):
    ss[name] = v


# ---------------------------------------------------------------------------
# Styling — works in BOTH light and dark Streamlit themes.
#  * Colour variables switch with the theme (a tiny script at the bottom tells the CSS which
#    theme Streamlit is using; if it cannot, the browser's light/dark setting is used).
#  * Yes/No pills are real st.button()s; the current answer is encoded in each button's key
#    (…_yes_on / …_yes_off / …_no_on / …_no_off) so CSS can colour the selected one.
#  * The "sidebar" is our own fixed panel (key = side_panel): ◀ closes it, ☰ Menu re-opens it.
#  * Responsive: on phones the panel stacks on top and the Yes/No rows stay on one line.
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ===== theme variables ===== */
:root {
    --text: #f8fafc; --text-dim: #cbd5e1; --card-border: #2a3142;
    --pill-bg: #1f2636; --pill-border: #3a4258; --pill-text: #cbd5e1;
    --red: #f87171; --green: #34d399; --teal: #2dd4bf;
}
@media (prefers-color-scheme: light) {
    :root { --text: #0f172a; --text-dim: #475569; --card-border: #d3dae6;
            --pill-bg: #f1f5f9; --pill-border: #c3ccda; --pill-text: #334155;
            --red: #dc2626; --green: #059669; --teal: #0f766e; }
}
html[data-pcos-theme="dark"] {
    --text: #f8fafc; --text-dim: #cbd5e1; --card-border: #2a3142;
    --pill-bg: #1f2636; --pill-border: #3a4258; --pill-text: #cbd5e1;
    --red: #f87171; --green: #34d399; --teal: #2dd4bf;
}
html[data-pcos-theme="light"] {
    --text: #0f172a; --text-dim: #475569; --card-border: #d3dae6;
    --pill-bg: #f1f5f9; --pill-border: #c3ccda; --pill-text: #334155;
    --red: #dc2626; --green: #059669; --teal: #0f766e;
}

/* ===== header / toolbar (kept, made bigger) ===== */
header[data-testid="stHeader"] { background: transparent; height: 3.2rem; pointer-events: none; }
[data-testid="stToolbar"], [data-testid="stToolbar"] * { pointer-events: auto; }
[data-testid="stToolbar"] { transform: scale(1.25); transform-origin: right center; margin-right: 1.4rem; top: 0 !important; height: 3.2rem; align-items: center; }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a { border-radius: 10px !important; transition: background .15s ease, transform .15s ease; }
[data-testid="stToolbar"] button:hover, [data-testid="stToolbar"] a:hover { background: rgba(20,184,166,.25) !important; transform: translateY(-1px); }
[data-testid="stToolbar"] svg { width: 1.15rem; height: 1.15rem; }
footer { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] { display: none !important; }

.main .block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 3.4rem !important; padding-bottom: 0.2rem !important; padding-right: 2rem !important; padding-left: 350px !important; max-width: none !important;
}
html, body, [class*="css"] { font-size: 16px; line-height: 1.6; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }

/* ===== hero ===== */
.hero { background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%); padding: 0.75rem 1.4rem; border-radius: 12px; color: #fff;
        box-shadow: 0 4px 14px rgba(15,118,110,0.35); }
.hero h1 { margin: 0; font-size: 1.45rem; line-height: 1.35; color: #fff; padding: 0; }
.hero p  { margin: 0.25rem 0 0 0; font-size: 0.95rem; line-height: 1.6; color: #e6fffb; }

/* ===== section cards ===== */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; padding: 0.55rem 1.05rem 0.8rem 1.05rem; border-color: var(--card-border) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: 0.4rem; }
.sec-head { display: inline-block; font-weight: 700; font-size: 1.02rem; line-height: 1.6; padding: 0.15rem 1rem; margin-bottom: 0.2rem; border-radius: 999px; color: #fff; }
.sec-basic { background: linear-gradient(90deg, #0d9488, #14b8a6); }
.sec-life  { background: linear-gradient(90deg, #d97706, #f59e0b); }
.sec-sym   { background: linear-gradient(90deg, #be185d, #ec4899); }

/* ===== field labels: big, bold, high-contrast in both themes ===== */
.q-label { font-size: 1.08rem; font-weight: 800; color: var(--text); line-height: 1.7; margin: 0.25rem 0 0.2rem 0; padding: 0.05rem 0; overflow: visible; }
label[data-testid="stWidgetLabel"] p {
    line-height: 1.7 !important; font-size: 1.1rem !important; font-weight: 800 !important; color: var(--text) !important; margin-bottom: 0.1rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.st-key-card_basic [data-testid="stHorizontalBlock"] { align-items: flex-end; }
.st-key-card_basic label[data-testid="stWidgetLabel"] { min-height: 1.8rem; }
div[data-testid="stNumberInput"] input { font-size: 1.08rem; }
div[data-testid="stNumberInput"] > div > div, div[data-baseweb="select"] > div { min-height: 2.7rem; transition: border-color .15s ease, box-shadow .15s ease; }
div[data-testid="stNumberInput"] > div > div:hover, div[data-baseweb="select"] > div:hover { border-color: #14b8a6 !important; box-shadow: 0 0 0 1px #14b8a6; }

/* ===== Yes / No — compact joined pair, clear selected state ===== */
:is(.st-key-card_life, .st-key-card_sym) [data-testid="stHorizontalBlock"] { gap: 0 !important; }
[class*="st-key-q_"] button {
    width: 100%; min-height: 2rem; padding: 0.05rem 0.4rem; background: var(--pill-bg); border: 1px solid var(--pill-border);
    color: var(--pill-text); box-shadow: none;
    transition: transform .12s ease, filter .12s ease, border-color .12s ease, background .12s ease;
}
[class*="st-key-q_"][class*="_yes_"] button { border-radius: 8px 0 0 8px; }
[class*="st-key-q_"][class*="_no_"] button  { border-radius: 0 8px 8px 0; margin-left: -1px; }
[class*="st-key-q_"] button p { font-size: 0.98rem; font-weight: 700; color: inherit; margin: 0; }
[class*="st-key-q_"] button:hover { border-color: #14b8a6; color: var(--text); filter: brightness(1.06); position: relative; z-index: 1; }
/* selected (and instant feedback while pressing) */
[class*="st-key-q_"][class*="_yes_on"] button, [class*="st-key-q_"][class*="_yes_off"] button:active {
    color: #fff; border-color: transparent; background: linear-gradient(90deg, #0d9488, #14b8a6); }
.st-key-card_life [class*="_yes_on"] button, .st-key-card_life [class*="_yes_off"] button:active { background: linear-gradient(90deg, #d97706, #f59e0b); }
.st-key-card_sym  [class*="_yes_on"] button, .st-key-card_sym  [class*="_yes_off"] button:active { background: linear-gradient(90deg, #be185d, #ec4899); }
[class*="st-key-q_"][class*="_no_on"] button, [class*="st-key-q_"][class*="_no_off"] button:active {
    color: #fff; border-color: transparent; background: linear-gradient(90deg, #2563eb, #3b82f6); }
[class*="st-key-q_"] button:focus { outline: none; box-shadow: none; }
/* instant (optimistic) state set by the script at the bottom the moment a button is clicked */
[class*="st-key-q_"][data-pend="off"] button { color: var(--pill-text); border-color: var(--pill-border); background: var(--pill-bg); }
[class*="st-key-q_"][data-pend="on"][class*="_yes_"] button { color: #fff; border-color: transparent; background: linear-gradient(90deg, #0d9488, #14b8a6); }
.st-key-card_life [class*="st-key-q_"][data-pend="on"][class*="_yes_"] button { background: linear-gradient(90deg, #d97706, #f59e0b); }
.st-key-card_sym  [class*="st-key-q_"][data-pend="on"][class*="_yes_"] button { background: linear-gradient(90deg, #be185d, #ec4899); }
[class*="st-key-q_"][data-pend="on"][class*="_no_"] button { color: #fff; border-color: transparent; background: linear-gradient(90deg, #2563eb, #3b82f6); }
[class*="st-key-lang_"][data-pend="off"] button, [class*="st-key-model_"][data-pend="off"] button { background: rgba(255,255,255,.10); border-color: rgba(255,255,255,.45); box-shadow: none; }
[class*="st-key-lang_"][data-pend="on"] button, [class*="st-key-model_"][data-pend="on"] button { background: linear-gradient(90deg, #0d9488, #14b8a6); border-color: #99f6e4; }

/* ===== Predict button ===== */
.st-key-predict button {
    border-radius: 12px; padding: 0.65rem 1rem; font-weight: 700; background: linear-gradient(90deg, #0f766e, #0891b2); color: #fff; border: none; width: 100%;
    box-shadow: 0 4px 14px rgba(8,145,178,.40); transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.st-key-predict button p { font-size: 1.15rem; font-weight: 700; color: #fff; }
.st-key-predict button:hover { transform: translateY(-3px); filter: brightness(1.15); box-shadow: 0 10px 24px rgba(8,145,178,.6); color: #fff; }
.st-key-predict button:active { transform: translateY(0); filter: brightness(.95); }

/* ===== result band ===== */
.res-card { border-radius: 12px; padding: 0.5rem 1.1rem; height: 116px; box-sizing: border-box; overflow: hidden; }
.verdict { display: flex; gap: 0.9rem; align-items: center; color: #fff; }
.verdict .big { font-size: 2.5rem; line-height: 1; }
.verdict h2 { margin: 0; padding: 0; font-size: 1.3rem; line-height: 1.25; color: #fff; }
.verdict small { color: #f1f5f9; font-size: 0.9rem; }
.v-high { background: linear-gradient(120deg, #7f1d1d 0%, #dc2626 55%, #f87171 100%); border: 1px solid #fca5a5; box-shadow: 0 6px 18px rgba(220,38,38,.35); }
.v-mid  { background: linear-gradient(120deg, #78350f 0%, #d97706 55%, #fbbf24 100%); border: 1px solid #fcd34d; box-shadow: 0 6px 18px rgba(245,158,11,.35); }
.v-low  { background: linear-gradient(120deg, #064e3b 0%, #059669 55%, #34d399 100%); border: 1px solid #6ee7b7; box-shadow: 0 6px 18px rgba(16,185,129,.35); }
.gauge { background: rgba(128,128,128,.08); border: 1px solid var(--card-border); color: var(--text); }
.gauge .t { font-size: 0.88rem; font-weight: 700; color: var(--text-dim); }
.gauge .v { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.vr { color: var(--red); } .vg { color: var(--green); } .vy { color: #d97706; }
.bar { position: relative; height: 14px; border-radius: 7px; margin-top: 0.7rem; background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%); }
.mk { position: absolute; top: -5px; width: 4px; height: 24px; background: var(--text); border-radius: 2px; box-shadow: 0 0 0 2px rgba(128,128,128,.55); }
.th { position: absolute; top: -2px; height: 18px; border-left: 2px dashed rgba(255,255,255,.8); }
.scale { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-dim); margin-top: 0.45rem; }
.mets { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; height: 116px; }
.m { background: rgba(20,184,166,.10); border: 1px solid rgba(20,184,166,.45); border-radius: 10px; padding: 0.55rem 0.8rem;
     display: flex; flex-direction: column; justify-content: center; transition: transform .15s ease, box-shadow .15s ease; }
.m:hover { transform: translateY(-2px); box-shadow: 0 6px 14px rgba(20,184,166,.25); }
.m .a { font-size: 0.9rem; font-weight: 700; color: var(--text-dim); }
.m .b { font-size: 1.4rem; font-weight: 700; color: var(--teal); margin-top: 0.15rem; }
.placeholder { border: 2px dashed rgba(20,184,166,.6); background: rgba(20,184,166,.07); border-radius: 12px; text-align: center; height: 116px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; padding: 0 1rem; font-size: 1.05rem; color: var(--text); }
.placeholder > span { display: block; line-height: 1.7; white-space: normal; }
.foot { font-size: 0.85rem; color: var(--text-dim); text-align: center; margin-top: 0; padding: 1.1rem 0 0 0; line-height: 1.5; position: relative; clear: both; }
@media (min-width: 901px) and (max-height: 820px) {
    .main .block-container, [data-testid="stMainBlockContainer"] { padding-top: 3.2rem !important; }
    [data-testid="stVerticalBlock"] { gap: 0.4rem; }
    .hero { padding: 0.5rem 1.2rem; } .hero h1 { font-size: 1.25rem; } .hero p { margin-top: 0.1rem; font-size: 0.88rem; }
    .res-card, .mets, .placeholder { height: 100px; }
    div[data-testid="stNumberInput"] > div > div, div[data-baseweb="select"] > div { min-height: 2.4rem; }
}

/* ===== OUR side panel (always dark, so text is always readable) ===== */
.st-key-side_panel {
    position: fixed; top: 0; left: 0; bottom: 0; width: 320px; z-index: 1000001; overflow-y: auto;
    background: linear-gradient(180deg, #0b3b3a 0%, #0c4a5a 55%, #14213d 100%);
    border-right: 2px solid #14b8a6; padding: 1.2rem 1.2rem 1.2rem 1.2rem; gap: 1rem;
}
.st-key-side_panel, .st-key-side_panel * { color: #e6fffb; }
.st-key-side_panel [data-testid="stVerticalBlock"] { gap: 0.9rem; }
.side-title { font-size: 1.45rem; line-height: 1.6; font-weight: 800; color: #fff; padding-right: 3.2rem; }
html[data-pcos-panel="closed"] .st-key-side_panel { display: none; }
html[data-pcos-panel="closed"] .main .block-container, html[data-pcos-panel="closed"] [data-testid="stMainBlockContainer"] { padding-left: 4.6rem !important; }
#pcos-toggle { position: fixed; z-index: 1000002; top: 10px; left: 12px; width: 42px; height: 42px; border-radius: 12px; border: 1px solid rgba(255,255,255,.5); cursor: pointer;
    background: linear-gradient(90deg, #0f766e, #0891b2); color: #fff; font-size: 1.2rem; font-weight: 800; line-height: 1; padding: 0; box-shadow: 0 4px 12px rgba(8,145,178,.5); transition: transform .12s ease, filter .12s ease; }
html[data-pcos-panel="open"] #pcos-toggle { left: 262px; background: rgba(255,255,255,.16); box-shadow: none; }
#pcos-toggle:hover { filter: brightness(1.2); transform: translateY(-1px); } #pcos-toggle:active { transform: scale(.94); }
.side-label { font-size: 1.15rem; font-weight: 800; color: #fff; line-height: 1.7; margin: 0.5rem 0 0.1rem 0; }
/* language + model pills */
[class*="st-key-lang_"] button, [class*="st-key-model_"] button {
    width: 100%; min-height: 2.8rem; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.45); box-shadow: none;
    transition: transform .15s ease, background .15s ease, border-color .15s ease;
}
[class*="st-key-lang_"] button { border-radius: 999px; }
[class*="st-key-model_"] button { border-radius: 10px; }
[class*="st-key-lang_"] button p, [class*="st-key-model_"] button p { font-size: 1.2rem; font-weight: 800; color: #fff; margin: 0; }
[class*="st-key-lang_"] button:hover, [class*="st-key-model_"] button:hover { transform: translateY(-2px); background: rgba(20,184,166,.40); border-color: #5eead4; }
[class*="st-key-lang_"][class*="_on"] button, [class*="st-key-model_"][class*="_on"] button {
    background: linear-gradient(90deg, #0d9488, #14b8a6); border-color: #99f6e4; box-shadow: 0 4px 12px rgba(20,184,166,.45); }
/* close / open buttons */
.st-key-side_close button, .st-key-side_open button {
    border-radius: 10px; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.45); color: #fff; min-height: 2.5rem; box-shadow: none;
    transition: transform .15s ease, background .15s ease;
}
.st-key-side_open button { background: linear-gradient(90deg, #0f766e, #0891b2); border: none; box-shadow: 0 4px 12px rgba(8,145,178,.5); min-height: 2.8rem; }
.st-key-side_close button p, .st-key-side_open button p { font-size: 1.1rem; font-weight: 800; color: #fff; }
.st-key-side_close button:hover, .st-key-side_open button:hover { transform: translateY(-2px); background: #14b8a6; color: #fff; }
.side-card { border-radius: 12px; padding: 0.8rem 0.95rem; font-size: 0.98rem; line-height: 1.65; color: #e6fffb; margin-top: 0.1rem; }
.side-card b { display: block; font-size: 1.05rem; line-height: 1.6; color: #fff; margin-bottom: 0.3rem; }
.side-card b.inl { display: inline; font-size: inherit; margin: 0; }
.side-divider { border: 0; border-top: 1px solid rgba(255,255,255,.28); margin: 0.5rem 0 0.2rem 0; }
.side-card code { background: rgba(255,255,255,.18); color: #fff; padding: 0.05rem 0.4rem; border-radius: 6px; font-size: 0.92rem; }
.side-src  { background: rgba(20,184,166,0.22); border-left: 5px solid #14b8a6; }
.side-warn { background: rgba(245,158,11,0.22); border-left: 5px solid #f59e0b; }
.side-perf { background: rgba(99,102,241,0.22); border-left: 5px solid #818cf8; }
.side-note { background: rgba(236,72,153,0.20); border-left: 5px solid #ec4899; }

/* ===== phones & tablets ===== */
@media (max-width: 900px) {
    .main .block-container, [data-testid="stMainBlockContainer"], html[data-pcos-panel="closed"] .main .block-container, html[data-pcos-panel="closed"] [data-testid="stMainBlockContainer"] { padding-left: 0.9rem !important; }
    html[data-pcos-panel="open"] #pcos-toggle { left: 12px; }
    .side-title { padding-left: 3.2rem; padding-right: 0; }
    .st-key-side_panel { position: static; width: auto; border: 1px solid #14b8a6; border-radius: 14px; margin-top: 0.2rem; }
    .side-card { display: none; }                      /* keep the stacked panel short on phones */
    [data-testid="stToolbar"] { transform: none; margin-right: 0.3rem; }
}
@media (max-width: 640px) {
    .main .block-container, [data-testid="stMainBlockContainer"] { padding-left: 0.9rem !important; padding-right: 0.9rem !important; padding-top: 3.2rem !important; }
    .hero h1 { font-size: 1.15rem; } .hero p { font-size: 0.82rem; }
    .q-label { font-size: 0.95rem; }
    /* keep every Yes/No row on ONE line (Streamlit would otherwise stack the columns) */
    :is(.st-key-card_life, .st-key-card_sym) [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    :is(.st-key-card_life, .st-key-card_sym) :is([data-testid="stColumn"], [data-testid="column"]) { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
    :is(.st-key-card_life, .st-key-card_sym) :is([data-testid="stColumn"], [data-testid="column"]):first-child:nth-last-child(3),
    :is(.st-key-card_life, .st-key-card_sym) :is([data-testid="stColumn"], [data-testid="column"]):last-child:nth-child(3) { flex: 2 1 0 !important; }
    :is(.st-key-card_life, .st-key-card_sym) :is([data-testid="stColumn"], [data-testid="column"]):nth-child(2):nth-last-child(2),
    :is(.st-key-card_life, .st-key-card_sym) :is([data-testid="stColumn"], [data-testid="column"]):nth-child(3):nth-last-child(3):not(:first-child):not(:last-child) { flex: 0.3 1 0 !important; }
    /* basic info: two fields per row */
    .st-key-card_basic [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; row-gap: 0.6rem; }
    .st-key-card_basic :is([data-testid="stColumn"], [data-testid="column"]) { flex: 1 1 calc(50% - 1rem) !important; min-width: calc(50% - 1rem) !important; }
    .res-card, .mets { height: auto; min-height: 120px; }
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Side panel — language, classifier, info — with ◀ close button
# ---------------------------------------------------------------------------
if True:   # panel is always rendered; the ◀/☰ button (added by the script at the bottom) shows/hides it instantly in the browser
    with st.container(key="side_panel"):
        st.markdown(f'<div class="side-title">{T["panel_title"]}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="side-label">{T["lang_label"]}</div>', unsafe_allow_html=True)
        c_bn, c_en = st.columns(2, gap="small")
        c_bn.button(("✓ " if ss.lang == "bn" else "") + "বাংলা", key=f"lang_bn_{'on' if ss.lang == 'bn' else 'off'}",
                    on_click=_set_lang, args=("bn",), use_container_width=True)
        c_en.button(("✓ " if ss.lang == "en" else "") + "English", key=f"lang_en_{'on' if ss.lang == 'en' else 'off'}",
                    on_click=_set_lang, args=("en",), use_container_width=True)

        st.markdown(f'<div class="side-label">{T["sidebar_model_label"]}</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2, gap="small")
        for col, name in ((m1, "SVM"), (m2, "KNN")):
            on = ss.model_val == name
            col.button(("✓ " if on else "") + name, key=f"model_{name.lower()}_{'on' if on else 'off'}",
                       on_click=_set_model, args=(name,), use_container_width=True)

        st.markdown(
            f'<hr class="side-divider"><div class="side-card side-note"><b>{T["two_model_title"]}</b>{T["two_model_note"]}</div>',
            unsafe_allow_html=True,
        )
        _p = PERF[ss.model_val]
        st.markdown(
            f'<div class="side-card side-perf"><b>{T["perf_title"]}</b>'
            + T["perf_text"].format(acc=_p["acc"], rec=_p["rec"], miss=round((100 - _p["rec"]) / 10))
            + f' ({ss.model_val})</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="side-card side-src"><b>{T["sidebar_source"]}</b>{T["sidebar_source_text"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="side-card side-warn"><b>{T["sidebar_disclaimer"]}</b>{T["sidebar_disclaimer_text"]}</div>',
            unsafe_allow_html=True,
        )

model_label = ss.model_val
model = load_model(MODEL_FILES[model_label])

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
_hero_html = f"""<div class="hero"><h1>{T['hero_title']}</h1><p>{T['hero_sub']}</p></div>"""
st.markdown(_hero_html, unsafe_allow_html=True)


@st.fragment
def question_card(card_key, head_cls, head_text, questions):
    """One card of Yes/No questions. A fragment, so a click only re-runs THIS card (fast), not the whole page."""
    with st.container(border=True, key=card_key):
        st.markdown(f'<span class="sec-head {head_cls}">{head_text}</span>', unsafe_allow_html=True)
        for i in range(0, len(questions), 2):
            pair = questions[i:i + 2]
            lab = st.columns([2, 0.3, 2], gap="small")
            btn = st.columns([1, 1, 0.3, 1, 1], gap="small")
            for j, (name, _default) in enumerate(pair):
                lab_col = lab[0] if j == 0 else lab[2]
                lab_col.markdown(f'<div class="q-label">{T[name]}</div>', unsafe_allow_html=True)
                yes_col = btn[0] if j == 0 else btn[3]
                no_col = btn[1] if j == 0 else btn[4]
                cur = ss[name]
                yes_col.button(("✓ " if cur == 1 else "") + T["yes"], key=f"q_{name}_yes_{'on' if cur == 1 else 'off'}",
                               on_click=_set_ans, args=(name, 1), use_container_width=True)
                no_col.button(("✓ " if cur == 0 else "") + T["no"], key=f"q_{name}_no_{'on' if cur == 0 else 'off'}",
                              on_click=_set_ans, args=(name, 0), use_container_width=True)


# Row 1 — basic information (5 fields on one level)
with st.container(border=True, key="card_basic"):
    st.markdown(f'<span class="sec-head sec-basic">{T["sec_basic"]}</span>', unsafe_allow_html=True)
    b1, b2, b3, b4, b5 = st.columns(5, gap="large")
    age = b1.number_input(T["age"], min_value=10, max_value=55, value=25, step=1)
    weight = b2.number_input(T["weight"], min_value=30.0, max_value=130.0, value=60.0, step=0.5)
    height = b3.number_input(T["height"], min_value=130.0, max_value=190.0, value=157.0, step=0.5)
    cycle_len = b4.number_input(T["cycle_len"], min_value=1, max_value=15, value=5, step=1)
    blood_group = b5.selectbox(T["blood_group"], list(BLOOD_GROUPS.keys()), index=4)

# Row 2 — lifestyle | symptoms
col_life, col_sym = st.columns(2, gap="medium")
with col_life:
    question_card("card_life", "sec-life", T["sec_lifestyle"], LIFESTYLE_Q)
with col_sym:
    question_card("card_sym", "sec-sym", T["sec_symptoms"], SYMPTOM_Q)

# Predict button — centred under the features
_, mid, _ = st.columns([1, 1.3, 1])
predict_clicked = mid.button(T["predict_btn"], key="predict", type="primary", use_container_width=True)

# Result band — directly under the button
r1, r2, r3 = st.columns([1.15, 1.5, 1.35], gap="medium")

if not predict_clicked:
    st.markdown(f'<div class="placeholder"><span>{T["placeholder"]}</span></div>', unsafe_allow_html=True)
else:
    row = pd.DataFrame([{
        "Regular_menstruation": ss["reg_cycle"],
        "Weight": weight,
        "Height": height,
        "Age": age,
        "Regular_physical_activity": ss["reg_exercise"],
        "Menstrual_period_duration": cycle_len,
        "Frequent_fast_food": ss["fast_food"],
        "Weight_gain": ss["weight_gain"],
        "ABO_blood_group": BLOOD_GROUPS[blood_group],
        "Skin_hyperpigmentation": ss["skin"],
        "Hair_loss": ss["hair_loss"],
        "Acne": ss["acne"],
        "Hirsutism": ss["hirsutism"],
    }])[FEATURE_ORDER]

    t0 = time.perf_counter()
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    pcos_prob = float(proba[1] if len(proba) > 1 else proba[0])
    pct = pcos_prob * 100
    marker = min(max(pct, 1.0), 99.0)

    high = pred == 1
    v_class = T["class_high"] if high else T["class_low"]
    if BORDER_LO <= pct <= BORDER_HI:
        v_cls, v_icon, v_text, num_cls = "v-mid", "⚖️", T["result_mid"], "vy"
    elif high:
        v_cls, v_icon, v_text, num_cls = "v-high", "⚠️", T["result_high"], "vr"
    else:
        v_cls, v_icon, v_text, num_cls = "v-low", "✅", T["result_low"], "vg"

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

# One small script, injected into the page itself (so it keeps working across reruns):
#  * detects light/dark theme  * ◀/☰ button that opens/closes the side panel instantly (no server round-trip)
#  * paints Yes/No, language and model buttons the moment they are pressed (before the server answers)
_JS = r"""
(function () {
  if (window.__pcosInit) return; window.__pcosInit = 1;
  var D = document, H = D.documentElement;
  function lum(c) { var m = c.match(/[\d.]+/g); if (!m) return null; if (m.length > 3 && parseFloat(m[3]) === 0) return null;
    return (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) / 255; }
  function theme() {
    var els = [D.querySelector('.stApp'), D.querySelector('[data-testid="stAppViewContainer"]'), D.body], l = null;
    for (var i = 0; i < els.length && l === null; i++) { if (els[i]) l = lum(getComputedStyle(els[i]).backgroundColor); }
    if (l === null) return; var t = l < 0.5 ? 'dark' : 'light';
    if (H.getAttribute('data-pcos-theme') !== t) H.setAttribute('data-pcos-theme', t);
  }
  theme(); setInterval(theme, 700);

  var open = window.innerWidth > 900;
  try { var v = localStorage.getItem('pcosPanel'); if (v) open = (v === 'open'); } catch (e) {}
  var btn = D.createElement('button'); btn.id = 'pcos-toggle'; btn.type = 'button';
  function setPanel(o) {
    open = o; H.setAttribute('data-pcos-panel', o ? 'open' : 'closed');
    btn.textContent = o ? '◀' : '☰'; btn.title = o ? 'Close menu' : 'Open menu';
    try { localStorage.setItem('pcosPanel', o ? 'open' : 'closed'); } catch (e) {}
  }
  btn.addEventListener('click', function () { setPanel(!open); });
  D.body.appendChild(btn); setPanel(open);

  D.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('button'); if (!b) return;
    var box = b.closest('[class*="st-key-"]'); if (!box) return;
    var m = box.className.match(/st-key-((?:q|lang|model)_[A-Za-z0-9_]+?)_(?:on|off)(?![A-Za-z0-9_])/);
    if (!m) return;
    var group = m[1].replace(/_[A-Za-z0-9]+$/, '');
    D.querySelectorAll('[class*="st-key-' + group + '_"]').forEach(function (el) { el.setAttribute('data-pend', el === box ? 'on' : 'off'); });
  }, true);
})();
"""
components.html(
    "<script>(function(){try{var P=window.parent;var s=P.document.createElement('script');s.textContent="
    + json.dumps(_JS) + ";P.document.head.appendChild(s);}catch(e){}})();</script>",
    height=0,
)
