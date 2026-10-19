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
BORDER_LO, BORDER_HI = 40.0, 60.0
# Value ranges the models were trained on — anything outside is rejected with a message (never silently predicted)
VALID = {"age": (13, 58), "weight": (30.0, 110.0), "height": (135.0, 190.0), "cycle_len": (0, 15)}   # = ranges in the merged training data (D1 + D2), rounded slightly   # display band only — the model's own 0/1 decision is unchanged

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
        "two_model_note": "দুই algorithm একই ডেটায় তুলনা করার জন্য। probability আলাদা হতে পারে; "
                           "আসল বিষয় <b class=\"inl\">predicted class</b> মেলে কিনা।",
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
        "band_title": "🎯 ফলাফলের ব্যান্ড",
        "band_low": "কম ঝুঁকি",
        "band_mid": "মাঝামাঝি",
        "band_high": "বেশি ঝুঁকি",
        "missing_title": "ℹ️ সব ঘর পূরণ করুন — পূর্বাভাস দেওয়া হয়নি",
        "missing_hint": "নিচের ঘরগুলো এখনও ফাঁকা:",
        "bg_placeholder": "নির্বাচন করুন",
        "warn_min": "⚠ সর্বনিম্ন {v} দিতে হবে",
        "warn_max": "⚠ সর্বোচ্চ {v} পর্যন্ত",
        "invalid_title": "⛔ ইনপুট ঠিক নেই — পূর্বাভাস দেওয়া হয়নি",
        "invalid_hint": "মডেল যে পরিসরের ডেটায় শেখানো হয়েছে, তার বাইরের মান দিলে ফল ভরসাযোগ্য হয় না। নিচের ঘরগুলো ঠিক করে আবার Predict চাপুন।",
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
        "tab_predict": "🔍 পূর্বাভাস",
        "tab_patient": "🧾 রোগীর তথ্য",
        "tab_report": "📄 রিপোর্ট ডাউনলোড",
        "patient_section_title": "🧾 রোগীর তথ্য",
        "patient_hint": "এই তথ্য শুধু ডাউনলোড করা রিপোর্টে বসবে — কোনো সার্ভারে সেভ হয় না।",
        "patient_name": "নাম",
        "patient_age": "বয়স (বছর)",
        "patient_mobile": "মোবাইল নম্বর",
        "patient_dr": "ডাক্তারের নাম",
        "patient_passport": "পাসপোর্ট / NID নম্বর (ঐচ্ছিক)",
        "adv_do": "✅ যা করা যেতে পারে",
        "adv_avoid": "🚫 যা এড়ানো উচিত",
        "adv_sources": "তথ্যসূত্র",
        "report_need_predict": "আগে \"পূর্বাভাস\" ট্যাবে গিয়ে সব তথ্য দিয়ে Predict চাপুন — তারপর এখানে রিপোর্ট ডাউনলোড করতে পারবেন।",
        "report_btn": "⬇️ রিপোর্ট ডাউনলোড করুন",
        "report_title": "PCOS স্ক্রিনিং রিপোর্ট",
        "report_patient_hdr": "রোগীর তথ্য",
        "report_result_hdr": "স্ক্রিনিং ফলাফল",
        "report_generated": "তৈরির তারিখ",
        "report_not_given": "দেওয়া হয়নি",
        "report_disclaimer": "এটি একটি গবেষণা প্রোটোটাইপের ফলাফল, চূড়ান্ত চিকিৎসা নির্ণয় না। চূড়ান্ত সিদ্ধান্তের জন্য একজন gynecologist/endocrinologist-এর পরামর্শ নিন।",
        "threshold_caption": "মাঝের দাগ হলো সিদ্ধান্তের সীমা (৫০%)।",
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
        "two_model_note": "Two algorithms compared on the same data. Probability can differ; "
                           "what matters is whether the <b class=\"inl\">predicted class</b> agrees.",
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
        "band_title": "🎯 Result bands",
        "band_low": "Low risk",
        "band_mid": "Borderline",
        "band_high": "High risk",
        "missing_title": "ℹ️ Please fill every field — no prediction made",
        "missing_hint": "These fields are still empty:",
        "bg_placeholder": "Select",
        "warn_min": "⚠ Minimum is {v}",
        "warn_max": "⚠ Maximum is {v}",
        "invalid_title": "⛔ Invalid input — no prediction made",
        "invalid_hint": "Values outside the range the models were trained on give unreliable results. Fix the fields below and press Predict again.",
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
        "tab_predict": "🔍 Prediction",
        "tab_patient": "🧾 Patient Information",
        "tab_report": "📄 Download Report",
        "patient_section_title": "🧾 Patient Information",
        "patient_hint": "This information is used only in the downloaded report — nothing is saved on any server.",
        "patient_name": "Name",
        "patient_age": "Age (years)",
        "patient_mobile": "Mobile number",
        "patient_dr": "Doctor's name",
        "patient_passport": "Passport / NID number (optional)",
        "adv_do": "✅ What you can do",
        "adv_avoid": "🚫 What to avoid",
        "adv_sources": "Sources",
        "report_need_predict": "Go to the \"Prediction\" tab, fill everything in and press Predict first — then come back here to download the report.",
        "report_btn": "⬇️ Download report",
        "report_title": "PCOS Screening Report",
        "report_patient_hdr": "Patient Information",
        "report_result_hdr": "Screening Result",
        "report_generated": "Generated on",
        "report_not_given": "Not provided",
        "report_disclaimer": "This is the result of a research prototype, not a final medical diagnosis. Please consult a gynecologist/endocrinologist for a final decision.",
        "threshold_caption": "The middle mark is the decision threshold (50%).",
    },
}

# ---------------------------------------------------------------------------
# Lifestyle / symptom advice shown with the result, and reused in the
# downloadable report. Generic, non-prescriptive wellness guidance only —
# no dosing, no diagnosis. Kept short on purpose (3–4 points each).
# ---------------------------------------------------------------------------
ADVICE = {
    "low": {
        "bn": {
            "do": [
                "সুষম খাবার খান এবং সপ্তাহে অন্তত ১৫০ মিনিট হালকা–মাঝারি ব্যায়াম করুন।",
                "মাসিক চক্র, ওজন বা ত্বক/চুলে নতুন কোনো পরিবর্তন খেয়াল রাখুন।",
                "বছরে অন্তত একবার রুটিন গাইনোকোলজিক্যাল চেকআপ করান।",
            ],
            "avoid": [
                "কম ঝুঁকি মানেই PCOS নেই তা নিশ্চিত নয় — উপসর্গ দেখা দিলে অগ্রাহ্য করবেন না।",
                "ডাক্তারের পরামর্শ ছাড়া কোনো medication নিজে শুরু বা বন্ধ করবেন না।",
            ],
        },
        "en": {
            "do": [
                "Eat a balanced diet and aim for at least 150 minutes of moderate activity a week.",
                "Keep an eye on any new change in periods, weight, skin or hair.",
                "Get a routine gynecological check-up at least once a year.",
            ],
            "avoid": [
                "A low result does not rule out PCOS — don't ignore symptoms if they appear.",
                "Don't start or stop any medication on your own without medical advice.",
            ],
        },
    },
    "mid": {
        "bn": {
            "do": [
                "দ্রুত একজন gynecologist/endocrinologist দেখান — নিশ্চিত নির্ণয়ের জন্য hormone test ও USG লাগতে পারে।",
                "ওজন নিয়ন্ত্রণ, নিয়মিত ব্যায়াম ও সুষম খাদ্যাভ্যাস আজ থেকেই শুরু করুন।",
                "মাসিক চক্র ও উপসর্গের একটা ডায়েরি/নোট রাখুন, ডাক্তারকে দেখাতে কাজে লাগবে।",
            ],
            "avoid": [
                "ফলাফল borderline — এর ভিত্তিতে নিজে থেকে কোনো সিদ্ধান্তে আসবেন না।",
                "যাচাই করাতে দেরি করবেন না; borderline case দ্রুত আরও পরীক্ষার দাবি রাখে।",
            ],
        },
        "en": {
            "do": [
                "See a gynecologist/endocrinologist soon — hormone tests and a USG may be needed to confirm.",
                "Start managing weight, regular exercise and a balanced diet from today.",
                "Keep a short diary of your cycle and symptoms — useful for the doctor's visit.",
            ],
            "avoid": [
                "The result is borderline — don't draw your own conclusion from it alone.",
                "Don't delay follow-up testing; a borderline result needs a closer look soon.",
            ],
        },
    },
    "high": {
        "bn": {
            "do": [
                "দ্রুত gynecologist/endocrinologist-এর সাথে যোগাযোগ করে প্রয়োজনীয় টেস্ট (hormonal panel, USG) করান।",
                "জীবনযাত্রায় পরিবর্তন আনুন: সুষম খাদ্য, নিয়মিত ব্যায়াম, পর্যাপ্ত ঘুম।",
                "পরিবারে PCOS বা ডায়াবেটিসের ইতিহাস থাকলে তা ডাক্তারকে জানান।",
            ],
            "avoid": [
                "শুধু এই প্রোটোটাইপের ফলাফলের ভিত্তিতে নিজে থেকে treatment শুরু করবেন না।",
                "উপসর্গ অবহেলা করবেন না — দেরি করলে জটিলতা বাড়তে পারে।",
            ],
        },
        "en": {
            "do": [
                "Contact a gynecologist/endocrinologist soon and get the needed tests (hormonal panel, USG).",
                "Make lifestyle changes: balanced diet, regular exercise, enough sleep.",
                "Tell your doctor if PCOS or diabetes runs in your family.",
            ],
            "avoid": [
                "Don't start any treatment on your own based only on this prototype's result.",
                "Don't ignore symptoms — delaying care can let things get more complicated.",
            ],
        },
    },
}
SOURCES = {
    "bn": [
        ("WHO — PCOS", "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome"),
        ("WHO — শারীরিক সক্রিয়তা", "https://www.who.int/news-room/fact-sheets/detail/physical-activity"),
    ],
    "en": [
        ("WHO — PCOS", "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome"),
        ("WHO — Physical activity", "https://www.who.int/news-room/fact-sheets/detail/physical-activity"),
    ],
}

# ---------------------------------------------------------------------------
# Session state (plain keys — never lost when the side panel is hidden)
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("lang", "bn")
ss.setdefault("model_val", "SVM")
for _k, _d in LIFESTYLE_Q + SYMPTOM_Q:
    ss.setdefault(_k, _d)
for _k in ("pt_name", "pt_mobile", "pt_dr", "pt_passport"):
    ss.setdefault(_k, "")
ss.setdefault("pt_age", None)
ss.setdefault("last_result", None)

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
.sec-head { display: inline-block; font-weight: 700; font-size: 1.02rem; line-height: 1.9; padding: 0.2rem 1rem 0.25rem 1rem; margin-bottom: 0.3rem; border-radius: 999px; color: #fff; }
.sec-basic { background: linear-gradient(90deg, #0d9488, #14b8a6); }
.sec-life  { background: linear-gradient(90deg, #d97706, #f59e0b); }
.sec-sym   { background: linear-gradient(90deg, #be185d, #ec4899); }

/* ===== field labels: big, bold, high-contrast in both themes ===== */
.q-label { font-size: 1.08rem; font-weight: 800; color: var(--text); line-height: 1.9; margin: 0.3rem 0 0.5rem 0; padding: 0.1rem 0 0.15rem 0; overflow: visible; }
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
.range-warn { color: var(--red); font-weight: 700; font-size: 0.9rem; line-height: 1.4; margin-top: 0.15rem; }
.bad-box { border: 2px solid #ef4444; background: rgba(239,68,68,.12); color: var(--text); border-radius: 12px; padding: 0.7rem 1.1rem; line-height: 1.6; }
.bad-box .bad-t { font-weight: 800; font-size: 1.1rem; color: var(--red); }
.bad-box ul { margin: 0.3rem 0 0 1.1rem; padding: 0; }
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
    border-right: 2px solid #14b8a6; padding: 0.9rem 1.1rem 0.9rem 1.1rem; gap: 0.6rem; scrollbar-width: thin; scrollbar-color: #5eead4 rgba(255,255,255,.12);
}
.st-key-side_panel, .st-key-side_panel * { color: #e6fffb; }
.st-key-side_panel::-webkit-scrollbar { width: 8px; }
.st-key-side_panel::-webkit-scrollbar-track { background: rgba(255,255,255,.10); border-radius: 8px; }
.st-key-side_panel::-webkit-scrollbar-thumb { background: #5eead4; border-radius: 8px; }
.st-key-side_panel [data-testid="stVerticalBlock"] { gap: 0.55rem; }
.side-title { font-size: 1.45rem; line-height: 1.8; padding-top: 0.2rem; font-weight: 800; color: #fff; padding-right: 3.2rem; }
html[data-pcos-panel="closed"] .st-key-side_panel { display: none; }
html[data-pcos-panel="closed"] .main .block-container, html[data-pcos-panel="closed"] [data-testid="stMainBlockContainer"] { padding-left: 4.6rem !important; }
#pcos-toggle { position: fixed; z-index: 1000002; top: 10px; left: 12px; width: 42px; height: 42px; border-radius: 12px; border: 1px solid rgba(255,255,255,.5); cursor: pointer;
    background: linear-gradient(90deg, #0f766e, #0891b2); color: #fff; font-size: 1.2rem; font-weight: 800; line-height: 1; padding: 0; box-shadow: 0 4px 12px rgba(8,145,178,.5); transition: transform .12s ease, filter .12s ease; }
html[data-pcos-panel="open"] #pcos-toggle { left: 262px; background: rgba(255,255,255,.16); box-shadow: none; }
#pcos-toggle:hover { filter: brightness(1.2); transform: translateY(-1px); } #pcos-toggle:active { transform: scale(.94); }
.side-label { font-size: 1.05rem; font-weight: 800; color: #fff; line-height: 1.9; margin: 0.5rem 0 0.45rem 0; padding: 0.1rem 0 0.15rem 0; }
/* language + model pills */
[class*="st-key-lang_"] button, [class*="st-key-model_"] button {
    width: 100%; min-height: 2.4rem; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.45); box-shadow: none;
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
.side-card { border-radius: 12px; padding: 0.6rem 0.85rem; font-size: 0.88rem; line-height: 1.55; color: #e6fffb; margin: 0; }
.side-card b { display: block; font-size: 0.98rem; line-height: 1.5; color: #fff; margin-bottom: 0.2rem; }
.side-card b.inl { display: inline; font-size: inherit; margin: 0; }
.side-stack { display: flex; flex-direction: column; gap: 1rem; margin-top: 0.6rem; }
.side-card code { background: rgba(255,255,255,.18); color: #fff; padding: 0.05rem 0.4rem; border-radius: 6px; font-size: 0.92rem; }
.side-src  { background: rgba(20,184,166,0.22); border-left: 5px solid #14b8a6; }
.side-warn { background: rgba(245,158,11,0.22); border-left: 5px solid #f59e0b; }
.band-row { display: flex; align-items: center; gap: 0.55rem; margin-top: 0.35rem; font-size: 0.9rem; }
.band-chip { min-width: 4.6rem; text-align: center; font-weight: 800; border-radius: 8px; padding: 0.1rem 0.4rem; color: #fff; }
.bc-low { background: linear-gradient(90deg, #059669, #34d399); } .bc-mid { background: linear-gradient(90deg, #d97706, #fbbf24); } .bc-high { background: linear-gradient(90deg, #b91c1c, #f87171); }
.side-band { background: rgba(255,255,255,0.10); border-left: 5px solid #5eead4; }
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

/* ===== top tabs (Prediction / Patient Information / Download Report) ===== */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--card-border); }
.stTabs [data-baseweb="tab"] { height: 2.6rem; padding: 0 1rem; border-radius: 10px 10px 0 0; font-weight: 700; color: var(--text-dim); }
.stTabs [aria-selected="true"] { color: var(--teal) !important; border-bottom: 3px solid #14b8a6 !important; }

/* ===== advice cards (what to do / what to avoid) ===== */
.adv-wrap { display: flex; gap: 1rem; margin-top: 0.9rem; flex-wrap: wrap; }
.adv-card { flex: 1 1 260px; min-width: 240px; border-radius: 12px; padding: 0.75rem 1rem; color: var(--text); }
.adv-card h4 { margin: 0 0 0.5rem 0; font-size: 1.02rem; }
.adv-card ol { margin: 0; padding-left: 1.2rem; line-height: 1.65; }
.adv-do { background: rgba(16,185,129,.10); border: 1px solid rgba(16,185,129,.40); }
.adv-avoid { background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.40); }
.adv-src { margin-top: 0.6rem; font-size: 0.85rem; color: var(--text-dim); }
.adv-src a { color: var(--teal); text-decoration: underline; }

/* ===== patient-info form ===== */
.patient-hint { font-size: 0.88rem; color: var(--text-dim); margin-bottom: 0.6rem; }

/* ===== centre the whole main page (hero + tabs + everything in them) in the space right of the side panel ===== */
.st-key-main_wrap { max-width: 1000px; margin: 0 auto; }
@media (max-width: 900px) {
    .st-key-main_wrap { max-width: 100%; }
}

/* ===== one merged result card: icon+title, predicted class, big %, bar, caption ===== */
.res-full { border-radius: 16px; padding: 1.1rem 1.4rem 1.3rem; text-align: center; margin-top: 1.1rem; }
.res-top { display: flex; align-items: center; justify-content: center; gap: 0.6rem; }
.res-top .chk { font-size: 1.6rem; line-height: 1; }
.res-top h2 { margin: 0; font-size: 1.35rem; color: #fff; }
.res-meta { color: #f1f5f9; font-size: 0.9rem; margin-top: 0.3rem; }
.res-sub { color: #f1f5f9; font-size: 0.95rem; margin-top: 0.9rem; }
.res-pct { font-size: 2.6rem; font-weight: 800; color: #fff; line-height: 1.15; margin-top: 0.1rem; }
.res-full .bar { margin: 0.9rem auto 0; max-width: 560px; }
.res-caption { font-size: 0.82rem; color: #f1f5f9; margin-top: 0.5rem; }
@media (max-width: 640px) {
    .res-top h2 { font-size: 1.1rem; } .res-pct { font-size: 2.1rem; }
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

        # all info cards in ONE block: a flex column, so the gap between cards is reliable and there is no stray <hr> margin
        st.markdown(
            '<div class="side-stack">'
            f'<div class="side-card side-band"><b>{T["band_title"]}</b>'
            f'<div class="band-row"><span class="band-chip bc-low">&lt; {BORDER_LO:g}%</span>{T["band_low"]}</div>'
            f'<div class="band-row"><span class="band-chip bc-mid">{BORDER_LO:g}–{BORDER_HI:g}%</span>{T["band_mid"]}</div>'
            f'<div class="band-row"><span class="band-chip bc-high">&gt; {BORDER_HI:g}%</span>{T["band_high"]}</div></div>'
            f'<div class="side-card side-note"><b>{T["two_model_title"]}</b>{T["two_model_note"]}</div>'
            f'<div class="side-card side-src"><b>{T["sidebar_source"]}</b>{T["sidebar_source_text"]}</div>'
            f'<div class="side-card side-warn"><b>{T["sidebar_disclaimer"]}</b>{T["sidebar_disclaimer_text"]}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

model_label = ss.model_val
model = load_model(MODEL_FILES[model_label])

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
with st.container(key="main_wrap"):
    _hero_html = f"""<div class="hero"><h1>{T['hero_title']}</h1><p>{T['hero_sub']}</p></div>"""
    st.markdown(_hero_html, unsafe_allow_html=True)

    tab_predict, tab_patient, tab_report = st.tabs([T["tab_predict"], T["tab_patient"], T["tab_report"]])


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


with tab_predict:
    # Row 1 — basic information (5 fields on one level)
    with st.container(border=True, key="card_basic"):
        st.markdown(f'<span class="sec-head sec-basic">{T["sec_basic"]}</span>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5, gap="large")
        def _ph(k):
            lo, hi = VALID[k]
            return f"{lo:g} – {hi:g}"

        # no default values: every field starts empty and shows its allowed range as the placeholder
        age = b1.number_input(T["age"], min_value=0, max_value=120, value=None, step=1, placeholder=_ph("age"))
        weight = b2.number_input(T["weight"], min_value=0.0, max_value=400.0, value=None, step=0.5, placeholder=_ph("weight"))
        height = b3.number_input(T["height"], min_value=0.0, max_value=260.0, value=None, step=0.5, placeholder=_ph("height"))
        cycle_len = b4.number_input(T["cycle_len"], min_value=0, max_value=90, value=None, step=1, placeholder=_ph("cycle_len"))
        blood_group = b5.selectbox(T["blood_group"], list(BLOOD_GROUPS.keys()), index=None, placeholder=T["bg_placeholder"])

    # instant range warning right under the field that is out of range
    def _range_note(col, key, val):
        lo, hi = VALID[key]
        if val is None:
            return
        if val < lo:
            msg = T["warn_min"].format(v=f"{lo:g}")
        elif val > hi:
            msg = T["warn_max"].format(v=f"{hi:g}")
        else:
            return
        col.markdown(f'<div class="range-warn">{msg}</div>', unsafe_allow_html=True)


    _range_note(b1, "age", age)
    _range_note(b2, "weight", weight)
    _range_note(b3, "height", height)
    _range_note(b4, "cycle_len", cycle_len)

    # Row 2 — lifestyle | symptoms
    col_life, col_sym = st.columns(2, gap="medium")
    with col_life:
        question_card("card_life", "sec-life", T["sec_lifestyle"], LIFESTYLE_Q)
    with col_sym:
        question_card("card_sym", "sec-sym", T["sec_symptoms"], SYMPTOM_Q)

    # Predict button — centred under the features
    _, mid, _ = st.columns([1, 1.3, 1])
    predict_clicked = mid.button(T["predict_btn"], key="predict", type="primary", use_container_width=True)

    # Result band — directly under the button, one merged card (icon+title, then the big % and the bar)
    _vals = {"age": age, "weight": weight, "height": height, "cycle_len": cycle_len}
    _missing = [k for k in ("age", "weight", "height", "cycle_len") if _vals[k] is None]
    if blood_group is None:
        _missing.append("blood_group")
    _bad = [k for k, (lo, hi) in VALID.items() if _vals[k] is not None and not (lo <= _vals[k] <= hi)]

    if not predict_clicked:
        st.markdown(f'<div class="placeholder"><span>{T["placeholder"]}</span></div>', unsafe_allow_html=True)
    elif _missing:
        _lines = "".join(f"<li><b class='inl'>{T[k]}</b></li>" for k in _missing)
        st.markdown(
            f'<div class="bad-box"><div class="bad-t">{T["missing_title"]}</div><div>{T["missing_hint"]}</div><ul>{_lines}</ul></div>',
            unsafe_allow_html=True,
        )
    elif _bad:
        _lines = "".join(
            f"<li><b class='inl'>{T[k]}</b>: {_vals[k]:g} → {VALID[k][0]:g} – {VALID[k][1]:g}</li>" for k in _bad
        )
        st.markdown(
            f'<div class="bad-box"><div class="bad-t">{T["invalid_title"]}</div><div>{T["invalid_hint"]}</div><ul>{_lines}</ul></div>',
            unsafe_allow_html=True,
        )
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
            v_cls, v_icon, v_text, num_cls, band_key = "v-mid", "⚖️", T["result_mid"], "vy", "mid"
        elif high:
            v_cls, v_icon, v_text, num_cls, band_key = "v-high", "⚠️", T["result_high"], "vr", "high"
        else:
            v_cls, v_icon, v_text, num_cls, band_key = "v-low", "✅", T["result_low"], "vg", "low"

        st.markdown(
            f"""<div class="res-full {v_cls}">
            <div class="res-top"><span class="chk">{v_icon}</span><h2>{v_text}</h2></div>
            <div class="res-meta">{T['pred_class']}: {v_class} · {model_label}</div>
            <div class="res-sub">{T['prob_title']}</div>
            <div class="res-pct">{pct:.1f}%</div>
            <div class="bar"><div class="th" style="left:50%"></div><div class="mk" style="left:calc({marker:.1f}% - 2px)"></div></div>
            <div class="res-caption">{T['threshold_caption']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        # what-to-do / what-to-avoid cards, straight from the band — no feature-impact bars
        _adv = ADVICE[band_key][ss.lang]
        _src = SOURCES[ss.lang]
        _do_html = "".join(f"<li>{x}</li>" for x in _adv["do"])
        _avoid_html = "".join(f"<li>{x}</li>" for x in _adv["avoid"])
        _src_html = " · ".join(f'<a href="{u}" target="_blank">{l}</a>' for l, u in _src)
        st.markdown(
            f"""<div class="adv-wrap">
            <div class="adv-card adv-do"><h4>{T['adv_do']}</h4><ol>{_do_html}</ol></div>
            <div class="adv-card adv-avoid"><h4>{T['adv_avoid']}</h4><ol>{_avoid_html}</ol></div>
            </div>
            <div class="adv-src">{T['adv_sources']}: {_src_html}</div>""",
            unsafe_allow_html=True,
        )

        # kept for the Download Report tab — same numbers, nothing recomputed there
        ss["last_result"] = {
            "band": band_key, "pct": pct, "v_text": v_text, "v_class": v_class,
            "model": model_label, "lang": ss.lang,
        }

    st.markdown(f'<div class="foot">{T["footer_warning"]}</div>', unsafe_allow_html=True)

with tab_patient:
    st.markdown(f'<span class="sec-head sec-basic">{T["patient_section_title"]}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="patient-hint">{T["patient_hint"]}</div>', unsafe_allow_html=True)
    with st.container(border=True, key="card_patient"):
        p1, p2 = st.columns(2, gap="large")
        p1.text_input(T["patient_name"], key="pt_name")
        p2.number_input(T["patient_age"], min_value=0, max_value=120, step=1, key="pt_age")
        p3, p4 = st.columns(2, gap="large")
        p3.text_input(T["patient_mobile"], key="pt_mobile")
        p4.text_input(T["patient_dr"], key="pt_dr")
        st.text_input(T["patient_passport"], key="pt_passport")

with tab_report:
    _res = ss.get("last_result")
    if not _res:
        st.info(T["report_need_predict"])
    else:
        _band_label = {"low": T["band_low"], "mid": T["band_mid"], "high": T["band_high"]}[_res["band"]]
        _adv = ADVICE[_res["band"]][_res["lang"]]
        _src = SOURCES[_res["lang"]]
        _not_given = T["report_not_given"]
        _today = time.strftime("%Y-%m-%d %H:%M")
        _do_li = "".join(f"<li>{x}</li>" for x in _adv["do"])
        _avoid_li = "".join(f"<li>{x}</li>" for x in _adv["avoid"])
        _src_li = "".join(f'<li><a href="{u}">{l}</a></li>' for l, u in _src)
        _report_html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{T['report_title']}</title>
<style>
body {{ font-family: 'Noto Sans Bengali', 'Nirmala UI', Arial, sans-serif; max-width: 720px; margin: 2rem auto; color: #0f172a; line-height: 1.6; }}
h1 {{ color: #0f766e; font-size: 1.4rem; }} h2 {{ color: #0d9488; font-size: 1.1rem; margin-top: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: .5rem; }} td {{ padding: .35rem .5rem; border-bottom: 1px solid #e2e8f0; }}
td.k {{ color: #475569; width: 40%; }} .band {{ display:inline-block; padding:.2rem .8rem; border-radius:999px; font-weight:700; color:#fff; }}
.band-low {{ background:#059669; }} .band-mid {{ background:#d97706; }} .band-high {{ background:#dc2626; }}
.foot {{ margin-top: 2rem; font-size: .85rem; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: .8rem; }}
ol {{ padding-left: 1.2rem; }}
</style></head><body>
<h1>🩺 {T['report_title']}</h1>
<div>{T['report_generated']}: {_today}</div>

<h2>{T['report_patient_hdr']}</h2>
<table>
<tr><td class="k">{T['patient_name']}</td><td>{ss.pt_name or _not_given}</td></tr>
<tr><td class="k">{T['patient_age']}</td><td>{ss.pt_age if ss.pt_age is not None else _not_given}</td></tr>
<tr><td class="k">{T['patient_mobile']}</td><td>{ss.pt_mobile or _not_given}</td></tr>
<tr><td class="k">{T['patient_dr']}</td><td>{ss.pt_dr or _not_given}</td></tr>
<tr><td class="k">{T['patient_passport']}</td><td>{ss.pt_passport or _not_given}</td></tr>
</table>

<h2>{T['report_result_hdr']}</h2>
<table>
<tr><td class="k">{T['band_title']}</td><td><span class="band band-{_res['band']}">{_band_label}</span></td></tr>
<tr><td class="k">{T['prob_title']}</td><td>{_res['pct']:.1f}%</td></tr>
<tr><td class="k">{T['pred_class']}</td><td>{_res['v_class']} · {_res['model']}</td></tr>
</table>

<h2>{T['adv_do']}</h2><ol>{_do_li}</ol>
<h2>{T['adv_avoid']}</h2><ol>{_avoid_li}</ol>
<h2>{T['adv_sources']}</h2><ol>{_src_li}</ol>

<div class="foot">⚠️ {T['report_disclaimer']}</div>
</body></html>"""
        st.markdown(
            f"""<div class="adv-wrap">
            <div class="adv-card adv-do"><h4>{T['adv_do']}</h4><ol>{_do_li}</ol></div>
            <div class="adv-card adv-avoid"><h4>{T['adv_avoid']}</h4><ol>{_avoid_li}</ol></div>
            </div>""",
            unsafe_allow_html=True,
        )
        _fname = (ss.pt_name or "patient").strip().replace(" ", "_")
        st.download_button(
            T["report_btn"], data=_report_html.encode("utf-8"),
            file_name=f"PCOS_report_{_fname}.html", mime="text/html",
            use_container_width=True,
        )

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
