"""Dự đoán nguy cơ bệnh tim từ 13 đặc trưng Clinical.
Dùng predict(sample) với dict theo tên cột hoặc list/array đúng thứ tự FEATURE_NAMES.
Nhãn 0: không mắc bệnh; nhãn 1: có bệnh (diagnosis gốc 1–4 gộp thành 1).
"""
from pathlib import Path
import joblib
import numpy as np

try:
    from .preprocess import FEATURE_NAMES
except ImportError:
    from preprocess import FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ppg_model.pkl"
_BUNDLE = None


def _load_bundle(model_path=MODEL_PATH):
    global _BUNDLE
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}. Hãy chạy preprocess.py và train.py trước.")
    if path.resolve() == MODEL_PATH.resolve() and _BUNDLE is not None:
        return _BUNDLE
    obj = joblib.load(path)
    bundle = obj if isinstance(obj, dict) and "model" in obj else {"model": obj, "feature_names": FEATURE_NAMES}
    if list(bundle.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Thứ tự đặc trưng trong model không khớp preprocess.py.")
    if path.resolve() == MODEL_PATH.resolve():
        _BUNDLE = bundle
    return bundle


def predict(sample, model_path=MODEL_PATH, return_probability=False):
    """Trả về nhãn dự đoán; có thể trả thêm xác suất lớp bệnh."""
    if isinstance(sample, dict):
        missing = [name for name in FEATURE_NAMES if name not in sample]
        if missing:
            raise ValueError(f"Thiếu đặc trưng: {', '.join(missing)}")
        row = [sample[name] for name in FEATURE_NAMES]
    else:
        row = np.asarray(sample, dtype=float).reshape(-1).tolist()
    if len(row) != len(FEATURE_NAMES):
        raise ValueError(f"Cần đúng {len(FEATURE_NAMES)} giá trị theo thứ tự FEATURE_NAMES.")
    x = np.asarray(row, dtype=float).reshape(1, -1)
    if np.isinf(x).any():
        raise ValueError("Dữ liệu đầu vào không được chứa Inf.")
    model = _load_bundle(model_path)["model"]
    label = int(model.predict(x)[0])
    if return_probability:
        probs = model.predict_proba(x)[0]
        classes = list(model.classes_)
        probability = float(probs[classes.index(1)]) if 1 in classes else 0.0
        return {"prediction": label, "label": "Có bệnh" if label else "Không bệnh", "probability_disease": round(probability, 4)}
    return label


if __name__ == "__main__":
    # Ví dụ minh họa theo thứ tự 13 đặc trưng trong preprocess.py.
    example = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]
    try:
        print("Kết quả:", predict(example, return_probability=True))
    except FileNotFoundError as exc:
        print("Lỗi:", exc)
