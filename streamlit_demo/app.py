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
# Bilingual text (বাংলা / English) — everything the user sees comes from here
# ---------------------------------------------------------------------------
TXT = {
    "bn": {
        "hero_title": "🩺 পিসিওএস (PCOS) ঝুঁকি স্ক্রিনিং — রিসার্চ প্রোটোটাইপ",
        "hero_sub": "এটা একটা academic research prototype — ক্লিনিক্যাল ডায়াগনসিস টুল না। reproduced ও "
                    "locked Table-9 (Merged, 20% split) pipeline-এর উপর বানানো, যেটা Step 44D-তে "
                    "deployment-feasibility হিসেবে latency/size পরীক্ষা করা হয়েছিল।",
        "sidebar_lang": "🌐 ভাষা",
        "sidebar_model_header": "⚙️ মডেল সেটিংস",
        "sidebar_model_label": "Classifier বেছে নিন",
        "sidebar_source": "📦 উৎস",
        "sidebar_source_text": "`Pcos Rerun 01` Drive প্রজেক্ট — frozen Table 9 (Merged, split=20%) "
                                "models, কোনো নতুন train করা হয়নি।",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।",
        "two_model_note": "ℹ️ **দুটো আলাদা মডেল (SVM, KNN) কেন?** দুইটা ভিন্ন algorithm একই ডেটায় কেমন আচরণ "
                           "করে তা তুলনা করে দেখানোর জন্য। probability-র সংখ্যা দুই মডেলে আলাদা হতে পারে "
                           "(SVM smooth মান দেয়, KNN ধাপে ধাপে মান দেয়) — এটা স্বাভাবিক। মূল যেটা দেখার, সেটা হলো "
                           "দুই মডেলের **predicted ঝুঁকি (risk আছে/নাই)** মেলে কিনা।",
        "sec_basic": "🧍 মৌলিক তথ্য",
        "age": "বয়স (বছর)",
        "weight": "ওজন (kg)",
        "height": "উচ্চতা (cm)",
        "cycle_len": "মাসিকের সময়কাল (দিন)",
        "blood_group": "রক্তের গ্রুপ",
        "sec_lifestyle": "🏃 জীবনযাত্রা ও স্বাস্থ্য অভ্যাস",
        "reg_cycle": "মাসিক নিয়মিত?",
        "reg_exercise": "নিয়মিত ব্যায়াম?",
        "fast_food": "ঘন ঘন ফাস্ট ফুড?",
        "weight_gain": "সম্প্রতি ওজন বৃদ্ধি?",
        "sec_symptoms": "✨ ত্বক / চুল / লোম লক্ষণ",
        "skin": "ত্বক কালচে হওয়া",
        "hair_loss": "চুল পড়া",
        "acne": "ব্রণ (Acne)",
        "hirsutism": "অতিরিক্ত লোম",
        "yes": "হ্যাঁ",
        "no": "না",
        "predict_btn": "🔍 পূর্বাভাস দিন (Predict)",
        "result_high": "⚠️ PCOS-এর ঝুঁকি আছে বলে মডেল অনুমান করছে",
        "result_low": "✅ PCOS-এর ঝুঁকি কম বলে মডেল অনুমান করছে",
        "prob_label": "সম্ভাবনা (Probability)",
        "metric_prob": "PCOS সম্ভাবনা",
        "metric_latency": "Prediction সময়",
        "metric_model": "ব্যবহৃত মডেল",
        "latency_caption": "এই single prediction-এর real latency উপরে দেখানো হয়েছে (Step 44D-তে measured "
                            "real-row latency-র সাথে তুলনীয়, একই frozen pipeline)।",
        "expander_title": "🔎 ইনপুট যাচাই করুন (what was sent to the model)",
        "footer_warning": "⚠️ এটা একটা গবেষণা প্রোটোটাইপ, পূর্বে reproduce ও audit করা একটা ML pipeline দিয়ে "
                           "বানানো — এটা ডাক্তারি নির্ণয় (medical diagnosis) না এবং কোনো চিকিৎসা সিদ্ধান্তে এটার "
                           "উপর ভরসা করা উচিত না। PCOS সন্দেহ হলে দয়া করে একজন gynecologist-এর সাথে পরামর্শ করুন।",
    },
    "en": {
        "hero_title": "🩺 PCOS Risk Screening — Research Prototype",
        "hero_sub": "An academic research prototype — not a clinical diagnosis tool. Built on our "
                    "reproduced and locked Table-9 (Merged, 20% split) pipeline, whose latency/size "
                    "was measured as a deployment-feasibility check in Step 44D.",
        "sidebar_lang": "🌐 Language",
        "sidebar_model_header": "⚙️ Model Settings",
        "sidebar_model_label": "Choose classifier",
        "sidebar_source": "📦 Source",
        "sidebar_source_text": "`Pcos Rerun 01` Drive project — frozen Table 9 (Merged, split=20%) "
                                "models. No new training was done.",
        "sidebar_disclaimer": "⚠️ Disclaimer",
        "sidebar_disclaimer_text": "Built for research purposes only — not medical advice or diagnosis.",
        "two_model_note": "ℹ️ **Why two models (SVM, KNN)?** To compare how two different algorithms "
                           "behave on the same data. The raw probability can differ between them "
                           "(SVM gives a smooth value, KNN gives a stepped value) — that's expected. "
                           "What matters most is whether the **predicted risk class** (high/low) agrees "
                           "between the two.",
        "sec_basic": "🧍 Basic Information",
        "age": "Age (years)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "cycle_len": "Menstrual cycle length (days)",
        "blood_group": "Blood group",
        "sec_lifestyle": "🏃 Lifestyle & Health Habits",
        "reg_cycle": "Regular periods?",
        "reg_exercise": "Regular exercise?",
        "fast_food": "Frequent fast food?",
        "weight_gain": "Recent weight gain?",
        "sec_symptoms": "✨ Skin / Hair Symptoms",
        "skin": "Skin darkening",
        "hair_loss": "Hair loss",
        "acne": "Acne",
        "hirsutism": "Hirsutism (excess hair)",
        "yes": "Yes",
        "no": "No",
        "predict_btn": "🔍 Predict",
        "result_high": "⚠️ The model predicts an elevated PCOS risk",
        "result_low": "✅ The model predicts a low PCOS risk",
        "prob_label": "Probability",
        "metric_prob": "PCOS Probability",
        "metric_latency": "Prediction Latency",
        "metric_model": "Model Used",
        "latency_caption": "The real latency of this single prediction is shown above (comparable to "
                            "the real-row latency measured in Step 44D, same frozen pipeline).",
        "expander_title": "🔎 Inspect input (what was sent to the model)",
        "footer_warning": "⚠️ This is a research prototype built on a previously reproduced and audited "
                           "ML pipeline — it is not a medical diagnosis and should not be relied on for "
                           "any treatment decision. If you suspect PCOS, please consult a gynecologist.",
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "bn"

# ---------------------------------------------------------------------------
# Styling — bigger fonts, compact spacing, no stray scrollbars
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 17px; }
    .main .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(120deg, #7b2ff7 0%, #f107a3 100%);
        padding: 1.3rem 1.8rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(123, 47, 247, 0.25);
    }
    .hero h1 { margin: 0; font-size: 1.7rem; line-height: 1.3; }
    .hero p { margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1rem; line-height: 1.5; }

    .section-card {
        background: var(--secondary-background-color, #f6f3fb);
        border-radius: 14px;
        padding: 1rem 1.1rem 0.5rem 1.1rem;
        border: 1px solid rgba(123, 47, 247, 0.12);
        height: 100%;
    }
    .section-card h3 { margin-top: 0; font-size: 1.15rem; color: #7b2ff7; }

    .info-note {
        background: #eef4ff;
        border: 1px solid #b9d2ff;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(0,0,0,0.06);
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    div[data-testid="stMetricLabel"] { font-size: 1rem; }

    .result-card {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-top: 0.4rem;
        margin-bottom: 0.9rem;
    }
    .result-high { background: linear-gradient(120deg, #ff5f6d22, #ffc37122); border: 1px solid #ff5f6d55; }
    .result-low  { background: linear-gradient(120deg, #11998e22, #38ef7d22); border: 1px solid #11998e55; }
    .result-card h2 { margin: 0 0 0.3rem 0; font-size: 1.4rem; }
    .result-card p { margin: 0.15rem 0; font-size: 1.05rem; }

    section[data-testid="stSidebar"] { font-size: 1.02rem; }
    section[data-testid="stSidebar"] h3 { font-size: 1.15rem; }

    .stButton>button { border-radius: 12px; padding: 0.7rem 1rem; font-weight: 700; font-size: 1.05rem; }
    label[data-testid="stWidgetLabel"] p { font-size: 1rem !important; font-weight: 600; }

    /* keep radios compact so three side-by-side cards don't force a page scrollbar */
    div[role="radiogroup"] { gap: 0.1rem; }
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
    age = st.number_input(T["age"], min_value=10, max_value=55, value=25, step=1)
    weight = st.number_input(T["weight"], min_value=30.0, max_value=130.0, value=60.0, step=0.5)
    height = st.number_input(T["height"], min_value=130.0, max_value=190.0, value=157.0, step=0.5)
    cycle_len = st.number_input(T["cycle_len"], min_value=1, max_value=15, value=5, step=1)
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
