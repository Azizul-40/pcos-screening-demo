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
        "sidebar_source_text": "`Pcos Rerun 01` Drive প্রজেক্ট — frozen Table 9 (Merged, split=20%) "
                                "models, কোনো নতুন train করা হয়নি।",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।",
        "two_model_note": "ℹ️ **দুটো মডেল (SVM, KNN) কেন?** দুই ভিন্ন algorithm একই ডেটায় কেমন আচরণ করে "
                           "তুলনা করার জন্য। probability আলাদা হতে পারে (SVM smooth, KNN ধাপে ধাপে) — "
                           "আসল বিষয় হলো **predicted ঝুঁকি** মেলে কিনা।",
        "sec_basic": "🧍 মৌলিক তথ্য",
        "age": "বয়স (বছর)",
        "weight": "ওজন (kg)",
        "height": "উচ্চতা (cm)",
        "cycle_len": "মাসিকের সময়কাল (দিন)",
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
        "predict_btn": "🔍 পূর্বাভাস দিন (Predict)",
        "result_title": "📊 ফলাফল",
        "placeholder": "তথ্য দিয়ে **Predict** চাপুন — ফলাফল এখানে দেখাবে।",
        "result_high": "⚠️ PCOS-এর ঝুঁকি আছে বলে মডেল অনুমান করছে",
        "result_low": "✅ PCOS-এর ঝুঁকি কম বলে মডেল অনুমান করছে",
        "prob_label": "সম্ভাবনা",
        "metric_prob": "PCOS সম্ভাবনা",
        "metric_latency": "Prediction সময়",
        "metric_model": "মডেল",
        "latency_caption": "এই prediction-এর real latency (Step 44D-র real-row latency-র সাথে তুলনীয়)।",
        "expander_title": "🔎 ইনপুট যাচাই করুন",
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
        "sidebar_source_text": "`Pcos Rerun 01` Drive project — frozen Table 9 (Merged, split=20%) "
                                "models. No new training was done.",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "Built for research purposes only — not medical advice or diagnosis.",
        "two_model_note": "ℹ️ **Why two models (SVM, KNN)?** To compare how two algorithms behave on the "
                           "same data. Raw probability can differ (SVM smooth, KNN stepped) — what matters "
                           "is whether the **predicted risk class** agrees.",
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
        "predict_btn": "🔍 Predict",
        "result_title": "📊 Result",
        "placeholder": "Fill in the details and press **Predict** — the result appears here.",
        "result_high": "⚠️ The model predicts an elevated PCOS risk",
        "result_low": "✅ The model predicts a low PCOS risk",
        "prob_label": "Probability",
        "metric_prob": "PCOS Probability",
        "metric_latency": "Latency",
        "metric_model": "Model",
        "latency_caption": "Real latency of this prediction (comparable to Step 44D's real-row latency).",
        "expander_title": "🔎 Inspect input",
        "footer_warning": "⚠️ Research prototype — not a medical diagnosis. Please consult a gynecologist "
                           "if you suspect PCOS.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "bn"

# ---------------------------------------------------------------------------
# Styling — compact (fits one screen, no page scroll), colored sections,
# colored sidebar with larger language / model controls, Streamlit's tiny
# top-right toolbar (Share / star / edit / GitHub / menu) hidden.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- remove top whitespace + toolbar + footer => no page scroll ---- */
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; visibility: hidden; }
    .main .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 0.6rem !important; padding-bottom: 0.2rem !important; max-width: 1400px;
    }
    html, body, [class*="css"] { font-size: 16px; }
    [data-testid="stVerticalBlock"] { gap: 0.45rem; }

    /* ---- hero ---- */
    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%);
        padding: 0.7rem 1.3rem; border-radius: 12px; color: #fff; margin-bottom: 0.2rem;
        box-shadow: 0 4px 14px rgba(15,118,110,0.35);
    }
    .hero h1 { margin: 0; font-size: 1.4rem; line-height: 1.25; color: #fff; padding: 0; }
    .hero p  { margin: 0.2rem 0 0 0; font-size: 0.9rem; line-height: 1.35; color: #e6fffb; opacity: .95; }

    /* ---- section cards (colored) ---- */
    .sec-head { display: inline-block; font-weight: 700; font-size: 1rem; padding: 0.15rem 0.8rem;
        border-radius: 999px; margin-bottom: 0.15rem; color: #fff; }
    .sec-basic { background: linear-gradient(90deg, #0d9488, #14b8a6); }
    .sec-life  { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .sec-sym   { background: linear-gradient(90deg, #be185d, #ec4899); }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px; padding: 0.15rem 0.2rem;
    }
    .card-basic div[data-testid="stVerticalBlockBorderWrapper"] { }
    label[data-testid="stWidgetLabel"] p { font-size: 0.9rem !important; font-weight: 600; margin-bottom: 0; }
    div[data-testid="stNumberInput"] input, div[data-baseweb="select"] > div { min-height: 2.1rem; }
    div[data-testid="stToggle"] label p, div[data-testid="stCheckbox"] label p { font-size: 0.95rem !important; }

    /* ---- result panel ---- */
    .result-card { border-radius: 12px; padding: 0.7rem 1rem; margin: 0.2rem 0 0.4rem 0; }
    .result-high { background: linear-gradient(120deg, #ff5f6d33, #ffc37133); border: 1px solid #ff5f6d88; }
    .result-low  { background: linear-gradient(120deg, #11998e33, #38ef7d33); border: 1px solid #11998e88; }
    .result-card h2 { margin: 0 0 0.15rem 0; font-size: 1.1rem; line-height: 1.3; padding: 0; }
    .result-card p  { margin: 0; font-size: 0.98rem; }
    .placeholder { border: 2px dashed rgba(20,184,166,.55); border-radius: 12px; padding: 1rem;
        text-align: center; font-size: 0.95rem; background: rgba(20,184,166,.07); }
    div[data-testid="stMetric"] { background: rgba(20,184,166,.10); border: 1px solid rgba(20,184,166,.35);
        border-radius: 10px; padding: 0.3rem 0.6rem; }
    div[data-testid="stMetricValue"] { font-size: 1.25rem; color: #14b8a6; }
    div[data-testid="stMetricLabel"] p { font-size: 0.8rem !important; }

    .stButton>button {
        border-radius: 10px; padding: 0.55rem 1rem; font-weight: 700; font-size: 1.05rem;
        background: linear-gradient(90deg, #0f766e, #0891b2); color: #fff; border: none; width: 100%;
    }
    .stButton>button:hover { filter: brightness(1.12); color: #fff; }

    /* ---- SIDEBAR: colored + bigger language / model controls ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3b3a 0%, #0c4a5a 55%, #14213d 100%);
        border-right: 2px solid #14b8a6; min-width: 330px !important;
    }
    section[data-testid="stSidebar"] * { color: #e6fffb; }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
    section[data-testid="stSidebar"] h3 { font-size: 1.35rem; margin: 0.3rem 0 0.2rem 0; }
    /* language radio: large pill buttons */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.6rem; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px; padding: 0.35rem 1.1rem; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 1.25rem !important; font-weight: 700; }
    /* model selectbox: large */
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p { font-size: 1.15rem !important; font-weight: 700; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        min-height: 3rem; font-size: 1.25rem; font-weight: 700; background: rgba(255,255,255,0.12);
        border: 1px solid #14b8a6; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #ffffff !important; font-size: 1.25rem; }
    /* sidebar info cards */
    .side-card { border-radius: 12px; padding: 0.6rem 0.8rem; margin-top: 0.6rem; font-size: 0.95rem; line-height: 1.45; }
    .side-src  { background: rgba(20,184,166,0.18); border-left: 5px solid #14b8a6; }
    .side-warn { background: rgba(245,158,11,0.18); border-left: 5px solid #f59e0b; }
    .side-note { background: rgba(236,72,153,0.16); border-left: 5px solid #ec4899; }
    .side-card b { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    lang_choice = st.radio(
        "🌐", ["বাংলা", "English"],
        index=0 if st.session_state.lang == "bn" else 1,
        horizontal=True, label_visibility="collapsed",
    )
    st.session_state.lang = "bn" if lang_choice == "বাংলা" else "en"

T = TXT[st.session_state.lang]

with st.sidebar:
    st.markdown(f"### {T['sidebar_model_header']}")
    model_label = st.selectbox(T["sidebar_model_label"], list(MODEL_FILES.keys()))
    st.markdown(f'<div class="side-card side-note">{T["two_model_note"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-card side-src"><b>{T["sidebar_source"]}</b><br>{T["sidebar_source_text"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="side-card side-warn"><b>{T["sidebar_disclaimer"]}</b><br>{T["sidebar_disclaimer_text"]}</div>',
        unsafe_allow_html=True,
    )

model = load_model(MODEL_FILES[model_label])

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><h1>{T['hero_title']}</h1><p>{T['hero_sub']}</p></div>""",
    unsafe_allow_html=True,
)

left, right = st.columns([2.3, 1], gap="medium")

with left:
    # Row 1 — basic information (5 fields in one row)
    with st.container(border=True):
        st.markdown(f'<span class="sec-head sec-basic">{T["sec_basic"]}</span>', unsafe_allow_html=True)
        b1, b2, b3, b4, b5 = st.columns(5)
        age = b1.number_input(T["age"], min_value=10, max_value=55, value=25, step=1)
        weight = b2.number_input(T["weight"], min_value=30.0, max_value=130.0, value=60.0, step=0.5)
        height = b3.number_input(T["height"], min_value=130.0, max_value=190.0, value=157.0, step=0.5)
        cycle_len = b4.number_input(T["cycle_len"], min_value=1, max_value=15, value=5, step=1)
        blood_group = b5.selectbox(T["blood_group"], list(BLOOD_GROUPS.keys()), index=4)

    # Row 2 — lifestyle (4 toggles in one row)
    with st.container(border=True):
        st.markdown(f'<span class="sec-head sec-life">{T["sec_lifestyle"]}</span>', unsafe_allow_html=True)
        l1, l2, l3, l4 = st.columns(4)
        reg_cycle = l1.toggle(T["reg_cycle"], value=False, key="reg_cycle")
        reg_exercise = l2.toggle(T["reg_exercise"], value=True, key="reg_exercise")
        fast_food = l3.toggle(T["fast_food"], value=False, key="fast_food")
        weight_gain = l4.toggle(T["weight_gain"], value=False, key="weight_gain")

    # Row 3 — skin / hair symptoms (4 toggles in one row)
    with st.container(border=True):
        st.markdown(f'<span class="sec-head sec-sym">{T["sec_symptoms"]}</span>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        skin = s1.toggle(T["skin"], value=False, key="skin")
        hair_loss = s2.toggle(T["hair_loss"], value=False, key="hairloss")
        acne = s3.toggle(T["acne"], value=False, key="acne")
        hirsutism = s4.toggle(T["hirsutism"], value=False, key="hirsutism")

with right:
    predict_clicked = st.button(T["predict_btn"], type="primary", use_container_width=True)

    if not predict_clicked:
        st.markdown(f'<div class="placeholder">{T["placeholder"]}</div>', unsafe_allow_html=True)
    else:
        row = pd.DataFrame([{
            "Regular_menstruation": int(reg_cycle),
            "Weight": weight,
            "Height": height,
            "Age": age,
            "Regular_physical_activity": int(reg_exercise),
            "Menstrual_period_duration": cycle_len,
            "Frequent_fast_food": int(fast_food),
            "Weight_gain": int(weight_gain),
            "ABO_blood_group": BLOOD_GROUPS[blood_group],
            "Skin_hyperpigmentation": int(skin),
            "Hair_loss": int(hair_loss),
            "Acne": int(acne),
            "Hirsutism": int(hirsutism),
        }])[FEATURE_ORDER]

        t0 = time.perf_counter()
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        pcos_prob = proba[1] if len(proba) > 1 else proba[0]

        result_class = "result-high" if pred == 1 else "result-low"
        result_text = T["result_high"] if pred == 1 else T["result_low"]
        st.markdown(
            f"""<div class="result-card {result_class}"><h2>{result_text}</h2>
            <p>{T['prob_label']}: <b>{pcos_prob*100:.1f}%</b></p></div>""",
            unsafe_allow_html=True,
        )
        st.progress(min(max(float(pcos_prob), 0.0), 1.0))

        m1, m2, m3 = st.columns(3)
        m1.metric(T["metric_prob"], f"{pcos_prob*100:.1f}%")
        m2.metric(T["metric_latency"], f"{latency_ms:.1f} ms")
        m3.metric(T["metric_model"], model_label)
        st.caption(T["latency_caption"])

        with st.expander(T["expander_title"]):
            st.dataframe(row.T.rename(columns={0: "value"}), use_container_width=True, height=250)

    st.caption(T["footer_warning"])
