"""Clinical - dự đoán bệnh tim bằng clinical_model.pkl.

Ví dụ:
  python mohinh/Clinical/predict.py                 # demo 5 dòng đầu dataset
  python mohinh/Clinical/predict.py --csv patients.csv
  python mohinh/Clinical/predict.py --age 63 --sex 1 --cp 1 --trestbps 145 --chol 233 \
      --fbs 1 --restecg 2 --thalach 150 --exang 0 --oldpeak 2.3 --slope 3 --ca 0 --thal 6
"""
import argparse

import joblib
import pandas as pd

from preprocess import FEATURES, SCALER_PATH, load_dataset
from train import MODEL_PATH


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    prep = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    X = prep.transform(df[FEATURES])
    out = df.copy()
    out["probability"] = model.predict_proba(X)[:, 1].round(4)
    out["label"] = model.predict(X)
    out["label"] = out["label"].map({0: "Không bệnh", 1: "Có bệnh"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV có header gồm: " + ", ".join(FEATURES))
    for f in FEATURES:
        ap.add_argument(f"--{f}", type=float)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, na_values="?")
    elif all(getattr(args, f) is not None for f in FEATURES):
        df = pd.DataFrame([{f: getattr(args, f) for f in FEATURES}])
    else:
        print("Không có input -> demo với 5 dòng đầu của dataset\n")
        df = load_dataset().head(5)

    res = predict_df(df)
    cols = ["probability", "label"] + (["target"] if "target" in res else [])
    print(res[cols].to_string())


if __name__ == "__main__":
    main()