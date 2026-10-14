import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PCOS Risk Screening — Prototype", page_icon="🩺", layout="centered")

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


st.title("🩺 PCOS Risk Screening — Research Prototype")
st.caption(
    "এটা একটা academic research prototype — ক্লিনিক্যাল ডায়াগনসিস টুল না। "
    "এটা আমাদের reproduced ও locked Table-9 (Merged, 20% split) pipeline-এর উপর বানানো, "
    "যেটা Step 44D-তে deployment-feasibility হিসেবে latency/size পরীক্ষা করা হয়েছিল।"
)

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Classifier বেছে নিন", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown(
        "**উৎস:** `Pcos Rerun 01` Drive প্রজেক্ট — frozen Table 9 (Merged, split=20%) "
        "models, কোনো নতুন train করা হয়নি।\n\n"
        "**Disclaimer:** গবেষণার জন্য তৈরি প্রোটোটাইপ, কোনো ডাক্তারি পরামর্শ বা ডায়াগনসিস না।"
    )

model = load_model(MODEL_FILES[model_label])

st.subheader("রোগীর তথ্য দিন")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("বয়স (Age, বছর)", min_value=10, max_value=55, value=25, step=1)
    weight = st.number_input("ওজন (Weight, kg)", min_value=30.0, max_value=130.0, value=60.0, step=0.5)
    height = st.number_input("উচ্চতা (Height, cm)", min_value=130.0, max_value=190.0, value=157.0, step=0.5)
    cycle_len = st.number_input("মাসিকের সময়কাল (দিন)", min_value=1, max_value=15, value=5, step=1)
    blood_group = st.selectbox("রক্তের গ্রুপ (Blood Group)", list(BLOOD_GROUPS.keys()), index=4)

with col2:
    reg_cycle = st.radio("মাসিক নিয়মিত? (Regular periods)", list(YES_NO.keys()), horizontal=False)
    reg_exercise = st.radio("নিয়মিত ব্যায়াম করেন?", list(YES_NO.keys()), horizontal=False, index=1)
    fast_food = st.radio("ঘন ঘন ফাস্ট ফুড খান?", list(YES_NO.keys()), horizontal=False)
    weight_gain = st.radio("সম্প্রতি ওজন বেড়েছে?", list(YES_NO.keys()), horizontal=False)

st.markdown("**ত্বক/চুল/লোম সংক্রান্ত লক্ষণ**")
col3, col4, col5, col6 = st.columns(4)
with col3:
    skin = st.radio("ত্বক কালচে হওয়া", list(YES_NO.keys()), horizontal=False, key="skin")
with col4:
    hair_loss = st.radio("চুল পড়া", list(YES_NO.keys()), horizontal=False, key="hairloss")
with col5:
    acne = st.radio("ব্রণ (Acne)", list(YES_NO.keys()), horizontal=False, key="acne")
with col6:
    hirsutism = st.radio("অতিরিক্ত লোম (Hirsutism)", list(YES_NO.keys()), horizontal=False, key="hirsutism")

if st.button("Predict / পূর্বাভাস দিন", type="primary", use_container_width=True):
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

    st.markdown("---")
    if pred == 1:
        st.error(f"**ফলাফল: PCOS-এর ঝুঁকি আছে বলে মডেল অনুমান করছে** (probability ≈ {pcos_prob*100:.1f}%)")
    else:
        st.success(f"**ফলাফল: PCOS-এর ঝুঁকি কম বলে মডেল অনুমান করছে** (probability ≈ {pcos_prob*100:.1f}%)")

    st.caption(
        f"এই single prediction-এর real latency: **{latency_ms:.2f} ms** "
        "(Step 44D-তে measured real-row latency-র সাথে তুলনীয়, একই frozen pipeline)।"
    )

    with st.expander("ইনপুট যাচাই করুন (what was sent to the model)"):
        st.dataframe(row.T.rename(columns={0: "value"}))

    st.warning(
        "⚠️ এটা একটা গবেষণা প্রোটোটাইপ, পূর্বে reproduce ও audit করা একটা ML pipeline দিয়ে বানানো — "
        "এটা ডাক্তারি নির্ণয় (medical diagnosis) না এবং কোনো চিকিৎসা সিদ্ধান্তে এটার উপর ভরসা করা উচিত না। "
        "PCOS সন্দেহ হলে দয়া করে একজন gynecologist-এর সাথে পরামর্শ করুন।"
    )
