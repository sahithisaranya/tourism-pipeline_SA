import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Tourism RF Model Demo")
uploaded = st.file_uploader("Upload CSV with features (same columns used in training)", type=["csv"])
model_path = Path("best_model.joblib")

if uploaded is not None:
  df = pd.read_csv(uploaded)
  st.write("Input preview:", df.head())
  try:
    model = joblib.load(model_path)
    preds = model.predict(pd.get_dummies(df, drop_first=True))
    st.write("Predictions:", preds)
  except Exception as e:
    st.error("Model load/predict error: " + str(e))
else:
  st.info("Upload a CSV file to get predictions. Place 'best_model.joblib' in the deploy folder before pushing to HF Space.")
