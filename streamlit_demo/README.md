# PCOS Risk Screening — Streamlit Prototype

এটা আমাদের **locked Table-9 (Merged, 20% split)** pipeline দিয়ে বানানো একটা real-time prediction demo।
কোনো নতুন model train করা হয়নি — এই app শুধু আগে থেকে frozen/locked `.joblib` model দুটো
(SVM ও KNN, Drive-এর `models/` ফোল্ডার থেকে) load করে, আর প্রতিটা prediction-এর real wall-clock
latency মেপে দেখায় — এটাই Step 44D-এর "deployment feasibility" claim-কে একটা কাজ-করা prototype-এ পরিণত করে।

## চালানোর নিয়ম (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

তারপর browser-এ `http://localhost:8501` খুলবে।

## ফোল্ডার গঠন

```
streamlit_demo/
├── app.py               # মূল Streamlit app
├── requirements.txt
├── README.md
└── models/
    ├── svm_merged20.joblib   # Table 9, Merged, split=20%, SVM (mean accuracy সবচেয়ে বেশি)
    └── knn_merged20.joblib   # Table 9, Merged, split=20%, KNN (headline best single cell 89.55%)
```

## এটা কী প্রমাণ করে (পেপারের জন্য)

- Pipeline শুধু "feasible" (Step 44D-এর measurement) না, বাস্তবে **deployable** — একটা কাজ-করা interactive
  prototype আছে যেটা locked model দিয়ে সত্যিকারের real-time prediction দেয়।
- প্রতিটা prediction-এর latency app নিজেই মাপে ও দেখায় — Step 44D-র measured latency-র (real-row ~14.6 ms গড়)
  সাথে সরাসরি তুলনীয়, কারণ একই pipeline/model ব্যবহার হচ্ছে, নতুন কিছু train হচ্ছে না।
- এটা দিয়ে title/abstract-এ সততার সাথে লেখা যায়: *"a deployable Streamlit prototype"* বা
  *"a working real-time screening interface"* — কারণ এখন সত্যিই এটা বানানো ও পরীক্ষা করা হয়েছে,
  শুধু feasibility measurement না।

## সীমাবদ্ধতা (honesty — পেপারে/Discussion-এ উল্লেখ করা উচিত)

- এটা একটা **local prototype**, কোনো public URL-এ hosted না (Streamlit Community Cloud বা অন্য কোথাও deploy
  করা হয়নি) — চাইলে পরে করা যায়, কিন্তু এখন claim শুধু "a working prototype exists and was tested locally"
  পর্যন্ত সীমাবদ্ধ রাখা উচিত, "publicly deployed" বলা যাবে না।
- Input validation ন্যূনতম (শুধু reasonable range বেঁধে দেওয়া হয়েছে) — production-grade input sanitization নেই।
- Blood-group encoding (A+/A-/.../AB- → 11–18) মূল Kaggle dataset-এর convention অনুযায়ী ধরে নেওয়া হয়েছে,
  dataset-এর codebook দিয়ে আলাদাভাবে cross-check করা হয়নি — paper-এ লিখলে এই ধরে নেওয়াটা উল্লেখ করা ভালো।
