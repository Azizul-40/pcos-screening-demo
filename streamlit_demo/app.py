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
    "SVM (highest mean accuracy, Table 9)": os.path.join(BASE_DIR, "models", "svm_merged20.joblib"),
    "KNN (headline best single cell, 89.55%)": os.path.join(BASE_DIR, "models", "knn_merged20.joblib"),
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

YES_NO = {"না (No)": 0, "হ্যাঁ (Yes)": 1}


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1100px; }

    .hero {
        background: linear-gradient(120deg, #7b2ff7 0%, #f107a3 100%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(123, 47, 247, 0.25);
    }
    .hero h1 { margin: 0; font-size: 1.8rem; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

    .section-card {
        background: var(--secondary-background-color, #f6f3fb);
        border-radius: 16px;
        padding: 1.2rem 1.4rem 0.6rem 1.4rem;
        margin-bottom: 1.1rem;
        border: 1px solid rgba(123, 47, 247, 0.12);
    }
    .section-card h3 {
        margin-top: 0;
        font-size: 1.05rem;
        color: #7b2ff7;
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(0,0,0,0.06);
    }

    .result-card {
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }
    .result-high {
        background: linear-gradient(120deg, #ff5f6d22, #ffc37122);
        border: 1px solid #ff5f6d55;
    }
    .result-low {
        background: linear-gradient(120deg, #11998e22, #38ef7d22);
        border: 1px solid #11998e55;
    }
    .result-card h2 { margin: 0 0 0.3rem 0; font-size: 1.35rem; }
    .result-card p { margin: 0.15rem 0; }

    .stButton>button {
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🩺 PCOS Risk Screening — Research Prototype</h1>
        <p>এটা একটা academic research prototype — ক্লিনিক্যাল ডায়াগনসিস টুল না। reproduced ও locked
        Table-9 (Merged, 20% split) pipeline-এর উপর বানানো, যেটা Step 44D-তে deployment-feasibility
        হিসেবে latency/size পরীক্ষা করা হয়েছিল।</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Model সেটিংস")
    model_label = st.selectbox("Classifier বেছে নিন", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown("**📦 উৎস**")
    st.caption(
        "`Pcos Rerun 01` Drive প্রজেক্ট — frozen Table 9 (Merged, split=20%) "
        "models, কোনো নতুন train করা হয়নি।"
    )
    st.markdown("**⚠️ Disclaimer**")
    st.caption("গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।")

model = load_model(MODEL_FILES[model_label])

st.markdown('<div class="section-card"><h3>🧍 মৌলিক তথ্য</h3>', unsafe_allow_html=True)
b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    age = st.number_input("বয়স (বছর)", min_value=10, max_value=55, value=25, step=1)
with b2:
    weight = st.number_input("ওজন (kg)", min_value=30.0, max_value=130.0, value=60.0, step=0.5)
with b3:
    height = st.number_input("উচ্চতা (cm)", min_value=130.0, max_value=190.0, value=157.0, step=0.5)
with b4:
    cycle_len = st.number_input("মাসিকের সময়কাল (দিন)", min_value=1, max_value=15, value=5, step=1)
with b5:
    blood_group = st.selectbox("রক্তের গ্রুপ", list(BLOOD_GROUPS.keys()), index=4)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><h3>🏃 জীবনযাত্রা ও স্বাস্থ্য অভ্যাস</h3>', unsafe_allow_html=True)
l1, l2, l3, l4 = st.columns(4)
with l1:
    reg_cycle = st.radio("মাসিক নিয়মিত?", list(YES_NO.keys()), horizontal=False)
with l2:
    reg_exercise = st.radio("নিয়মিত ব্যায়াম?", list(YES_NO.keys()), horizontal=False, index=1)
with l3:
    fast_food = st.radio("ঘন ঘন ফাস্ট ফুড?", list(YES_NO.keys()), horizontal=False)
with l4:
    weight_gain = st.radio("সম্প্রতি ওজন বৃদ্ধি?", list(YES_NO.keys()), horizontal=False)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-card"><h3>✨ ত্বক / চুল / লোম সংক্রান্ত লক্ষণ</h3>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
with s1:
    skin = st.radio("ত্বক কালচে হওয়া", list(YES_NO.keys()), horizontal=False, key="skin")
with s2:
    hair_loss = st.radio("চুল পড়া", list(YES_NO.keys()), horizontal=False, key="hairloss")
with s3:
    acne = st.radio("ব্রণ (Acne)", list(YES_NO.keys()), horizontal=False, key="acne")
with s4:
    hirsutism = st.radio("অতিরিক্ত লোম", list(YES_NO.keys()), horizontal=False, key="hirsutism")
st.markdown("</div>", unsafe_allow_html=True)

predict_clicked = st.button("🔍 Predict / পূর্বাভাস দিন", type="primary", use_container_width=True)

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

    if pred == 1:
        st.markdown(
            f"""
            <div class="result-card result-high">
                <h2>⚠️ PCOS-এর ঝুঁকি আছে বলে মডেল অনুমান করছে</h2>
                <p>Predicted probability: <b>{pcos_prob*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-low">
                <h2>✅ PCOS-এর ঝুঁকি কম বলে মডেল অনুমান করছে</h2>
                <p>Predicted probability: <b>{pcos_prob*100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("PCOS Probability", f"{pcos_prob*100:.1f}%")
    m2.metric("Prediction Latency", f"{latency_ms:.2f} ms")
    m3.metric("Model Used", model_label.split(" (")[0])

    st.progress(min(max(pcos_prob, 0.0), 1.0))
    st.caption(
        "এই single prediction-এর real latency উপরে দেখানো হয়েছে "
        "(Step 44D-তে measured real-row latency-র সাথে তুলনীয়, একই frozen pipeline)।"
    )

    with st.expander("🔎 ইনপুট যাচাই করুন (what was sent to the model)"):
        st.dataframe(row.T.rename(columns={0: "value"}), use_container_width=True)

    st.warning(
        "⚠️ এটা একটা গবেষণা প্রোটোটাইপ, পূর্বে reproduce ও audit করা একটা ML pipeline দিয়ে বানানো — "
        "এটা ডাক্তারি নির্ণয় (medical diagnosis) না এবং কোনো চিকিৎসা সিদ্ধান্তে এটার উপর ভরসা করা উচিত না। "
        "PCOS সন্দেহ হলে দয়া করে একজন gynecologist-এর সাথে পরামর্শ করুন।"
    )
