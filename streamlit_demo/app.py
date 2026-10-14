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
# These are the exact models whose size/latency were measured in Step 44D
# (Deployment-Feasibility Assessment). Nothing here is re-trained — this app
# only loads the already-frozen pipeline and runs it on a single new input,
# which is what Step 44D's "real-row prediction latency" measured.
#
# The 13 input features below are NOT chosen by this app — they are read
# directly off the frozen pipeline's expected input schema
# (scaler.feature_names_in_), i.e. exactly the low-cost feature set that the
# project's earlier hybrid feature-selection / data-fusion step already
# settled on for the Merged dataset. This file only has to match that order.
#
# NOTE: paths are resolved relative to THIS FILE's own folder, not the
# process's working directory — Streamlit Cloud runs the app with the repo
# root as the working directory, so a bare "models/..." path breaks whenever
# app.py itself lives in a subfolder (e.g. streamlit_demo/app.py).
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
        "hero_sub": "একটা academic research prototype — ক্লিনিক্যাল ডায়াগনসিস টুল না। reproduced ও "
                    "locked Table-9 (Merged, 20% split) pipeline-এর উপর বানানো।",
        "sidebar_model_header": "⚙️ মডেল সেটিংস",
        "sidebar_model_label": "Classifier বেছে নিন",
        "sidebar_source": "📦 উৎস",
        "sidebar_source_text": "`Pcos Rerun 01` Drive প্রজেক্ট — frozen Table 9 (Merged, split=20%) "
                                "models, কোনো নতুন train করা হয়নি।",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।",
        "two_model_note": "ℹ️ **দুটো আলাদা মডেল (SVM, KNN) কেন?** দুই ভিন্ন algorithm একই ডেটায় কেমন আচরণ করে "
                           "তুলনা করে দেখানোর জন্য। probability সংখ্যা আলাদা হতে পারে (SVM smooth, KNN ধাপে "
                           "ধাপে) — এটা স্বাভাবিক। আসল যেটা দেখার, সেটা হলো দুই মডেলের **predicted ঝুঁকি** মেলে কিনা।",
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
        "yes": "হ্যাঁ",
        "no": "না",
        "predict_btn": "🔍 পূর্বাভাস দিন (Predict)",
        "result_high": "⚠️ PCOS-এর ঝুঁকি আছে বলে মডেল অনুমান করছে",
        "result_low": "✅ PCOS-এর ঝুঁকি কম বলে মডেল অনুমান করছে",
        "prob_label": "সম্ভাবনা",
        "metric_prob": "PCOS সম্ভাবনা",
        "metric_latency": "Prediction সময়",
        "metric_model": "ব্যবহৃত মডেল",
        "latency_caption": "এই prediction-এর real latency (Step 44D-তে measured real-row latency-র সাথে "
                            "তুলনীয়, একই frozen pipeline)।",
        "expander_title": "🔎 ইনপুট যাচাই করুন",
        "footer_warning": "⚠️ এটা গবেষণা প্রোটোটাইপ, পূর্বে reproduce ও audit করা ML pipeline দিয়ে বানানো — "
                           "ডাক্তারি নির্ণয় না, চিকিৎসা সিদ্ধান্তে এটার উপর ভরসা করা উচিত না। PCOS সন্দেহ হলে "
                           "একজন gynecologist-এর সাথে পরামর্শ করুন।",
    },
    "en": {
        "hero_title": "🩺 PCOS Risk Screening — Research Prototype",
        "hero_sub": "An academic research prototype — not a clinical diagnosis tool. Built on our "
                    "reproduced and locked Table-9 (Merged, 20% split) pipeline.",
        "sidebar_model_header": "⚙️ Model Settings",
        "sidebar_model_label": "Choose classifier",
        "sidebar_source": "📦 Source",
        "sidebar_source_text": "`Pcos Rerun 01` Drive project — frozen Table 9 (Merged, split=20%) "
                                "models. No new training was done.",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "Built for research purposes only — not medical advice or diagnosis.",
        "two_model_note": "ℹ️ **Why two models (SVM, KNN)?** To compare how two algorithms behave on the "
                           "same data. Raw probability can differ (SVM is smooth, KNN is stepped) — that's "
                           "expected. What matters is whether the **predicted risk class** agrees.",
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
        "result_high": "⚠️ The model predicts an elevated PCOS risk",
        "result_low": "✅ The model predicts a low PCOS risk",
        "prob_label": "Probability",
        "metric_prob": "PCOS Probability",
        "metric_latency": "Prediction Latency",
        "metric_model": "Model Used",
        "latency_caption": "Real latency of this prediction (comparable to Step 44D's measured "
                            "real-row latency, same frozen pipeline).",
        "expander_title": "🔎 Inspect input",
        "footer_warning": "⚠️ This is a research prototype built on a previously reproduced and audited "
                           "ML pipeline — not a medical diagnosis. Please consult a gynecologist if you "
                           "suspect PCOS.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "bn"

# ---------------------------------------------------------------------------
# Styling — theme-aware (works in both light & dark), medical teal palette,
# compact layout to minimize scrolling.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 16.5px; }
    .main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1250px; }

    .hero {
        background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%);
        padding: 1.1rem 1.6rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 0.8rem;
        box-shadow: 0 6px 18px rgba(15, 118, 110, 0.3);
    }
    .hero h1 { margin: 0; font-size: 1.55rem; line-height: 1.3; color: #ffffff; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.95; font-size: 0.98rem; line-height: 1.45; color: #e6fffb; }

    .section-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 12px;
        padding: 0.8rem 1rem 0.3rem 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        height: 100%;
    }
    .section-card h3 { margin-top: 0; margin-bottom: 0.4rem; font-size: 1.08rem; color: #14b8a6; }

    .info-note {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid #14b8a6;
        border-left: 5px solid #14b8a6;
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #0f766e; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; }

    .result-card { border-radius: 14px; padding: 1.1rem 1.3rem; margin-top: 0.3rem; margin-bottom: 0.7rem; }
    .result-high { background: linear-gradient(120deg, #ff5f6d22, #ffc37122); border: 1px solid #ff5f6d55; }
    .result-low  { background: linear-gradient(120deg, #11998e22, #38ef7d22); border: 1px solid #11998e55; }
    .result-card h2 { margin: 0 0 0.25rem 0; font-size: 1.3rem; color: var(--text-color); }
    .result-card p { margin: 0.1rem 0; font-size: 1rem; color: var(--text-color); }

    section[data-testid="stSidebar"] { font-size: 0.98rem; }
    section[data-testid="stSidebar"] h3 { font-size: 1.1rem; }

    .stButton>button {
        border-radius: 10px; padding: 0.6rem 1rem; font-weight: 700; font-size: 1rem;
        background-color: #0f766e; color: white; border: none;
    }
    .stButton>button:hover { background-color: #0d9488; color: white; }

    label[data-testid="stWidgetLabel"] p { font-size: 0.94rem !important; font-weight: 600; margin-bottom: 0.1rem; }
    div[role="radiogroup"] { gap: 0.05rem; }
    div.row-widget.stRadio > div { flex-wrap: nowrap; }

    /* tighten vertical gaps between stacked widgets inside the cards */
    .section-card div[data-testid="stVerticalBlock"] { gap: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.markdown("---")
    st.markdown(f"**{T['sidebar_source']}**")
    st.caption(T["sidebar_source_text"])
    st.markdown(f"**{T['sidebar_disclaimer']}**")
    st.caption(T["sidebar_disclaimer_text"])

model = load_model(MODEL_FILES[model_label])

st.markdown(
    f"""<div class="hero"><h1>{T['hero_title']}</h1><p>{T['hero_sub']}</p></div>""",
    unsafe_allow_html=True,
)
st.markdown(f"""<div class="info-note">{T['two_model_note']}</div>""", unsafe_allow_html=True)

YES_NO = {T["no"]: 0, T["yes"]: 1}

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown(f'<div class="section-card"><h3>{T["sec_basic"]}</h3>', unsafe_allow_html=True)
    r1a, r1b = st.columns(2)
    age = r1a.number_input(T["age"], min_value=10, max_value=55, value=25, step=1)
    weight = r1b.number_input(T["weight"], min_value=30.0, max_value=130.0, value=60.0, step=0.5)
    r2a, r2b = st.columns(2)
    height = r2a.number_input(T["height"], min_value=130.0, max_value=190.0, value=157.0, step=0.5)
    cycle_len = r2b.number_input(T["cycle_len"], min_value=1, max_value=15, value=5, step=1)
    blood_group = st.selectbox(T["blood_group"], list(BLOOD_GROUPS.keys()), index=4)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="section-card"><h3>{T["sec_lifestyle"]}</h3>', unsafe_allow_html=True)
    reg_cycle = st.radio(T["reg_cycle"], list(YES_NO.keys()), horizontal=True, key="reg_cycle")
    reg_exercise = st.radio(T["reg_exercise"], list(YES_NO.keys()), horizontal=True, index=1, key="reg_exercise")
    fast_food = st.radio(T["fast_food"], list(YES_NO.keys()), horizontal=True, key="fast_food")
    weight_gain = st.radio(T["weight_gain"], list(YES_NO.keys()), horizontal=True, key="weight_gain")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="section-card"><h3>{T["sec_symptoms"]}</h3>', unsafe_allow_html=True)
    skin = st.radio(T["skin"], list(YES_NO.keys()), horizontal=True, key="skin")
    hair_loss = st.radio(T["hair_loss"], list(YES_NO.keys()), horizontal=True, key="hairloss")
    acne = st.radio(T["acne"], list(YES_NO.keys()), horizontal=True, key="acne")
    hirsutism = st.radio(T["hirsutism"], list(YES_NO.keys()), horizontal=True, key="hirsutism")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict_clicked = st.button(T["predict_btn"], type="primary", use_container_width=True)

if predict_clicked:
    row = pd.DataFrame([{
        "Regular_menstruation": YES_NO[reg_cycle],
        "Weight": weight,
        "Height": height,
        "Age": age,
        "Regular_physical_activity": YES_NO[reg_exercise],
        "Menstrual_period_duration": cycle_len,
        "Frequent_fast_food": YES_NO[fast_food],
        "Weight_gain": YES_NO[weight_gain],
        "ABO_blood_group": BLOOD_GROUPS[blood_group],
        "Skin_hyperpigmentation": YES_NO[skin],
        "Hair_loss": YES_NO[hair_loss],
        "Acne": YES_NO[acne],
        "Hirsutism": YES_NO[hirsutism],
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

    m1, m2, m3 = st.columns(3)
    m1.metric(T["metric_prob"], f"{pcos_prob*100:.1f}%")
    m2.metric(T["metric_latency"], f"{latency_ms:.2f} ms")
    m3.metric(T["metric_model"], model_label)

    st.progress(min(max(pcos_prob, 0.0), 1.0))
    st.caption(T["latency_caption"])

    with st.expander(T["expander_title"]):
        st.dataframe(row.T.rename(columns={0: "value"}), use_container_width=True)

    st.warning(T["footer_warning"])
