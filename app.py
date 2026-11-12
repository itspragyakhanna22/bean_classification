import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("bean_classifier.pkl")


bundle = load_bundle()
model, le = bundle["pipeline"], bundle["label_encoder"]
features, fstats, examples = bundle["features"], bundle["feature_stats"], bundle["class_examples"]

st.title("🫘 Dry Bean Type Classifier")
st.write("Enter the physical measurements of a bean (as produced by the camera system) and get the predicted bean type.")

st.sidebar.header("Quick start")
choice = st.sidebar.selectbox("Pre-fill with a typical bean of type:", ["Manual entry"] + list(examples.index))
st.sidebar.caption("Values are class medians from the training data. Edit any value afterwards.")

defaults = fstats["50%"] if choice == "Manual entry" else examples.loc[choice]

cols = st.columns(4)
values = {}
for i, f in enumerate(features):
    lo, hi = float(fstats.loc[f, "min"]), float(fstats.loc[f, "max"])
    step = max((hi - lo) / 500, 1e-6)
    values[f] = cols[i % 4].number_input(
        f, min_value=0.0, value=float(defaults[f]), step=step,
        format="%.4f" if hi <= 10 else "%.2f", key=f"{choice}_{f}",
        help=f"Training range: {lo:.4g} to {hi:.4g}")

if st.button("Predict bean type", type="primary"):
    X_new = pd.DataFrame([values], columns=features)
    label = le.inverse_transform(model.predict(X_new))[0]
    st.success(f"Predicted bean type: **{label}**")
    if hasattr(model, "predict_proba"):
        proba = pd.Series(model.predict_proba(X_new)[0], index=le.classes_, name="Probability")
        st.bar_chart(proba)
        st.caption(f"Confidence: {proba.max():.1%}")
