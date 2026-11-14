PPG – CLINICAL DATABASE CLASSIFICATION

Cau truc:
- preprocess.py: doc processed.cleveland.data, lam sach, kham pha, tao X_ppg.npy/y_ppg.npy.
- train.py: huan luyen RandomForestClassifier, danh gia Accuracy/Precision/Recall/F1, luu ppg_model.pkl.
- predict.py: ham predict() nhan 13 dac trung Clinical.
- dataset/Clinical/: du lieu Cleveland goc va tai lieu mo ta.
- X_ppg.npy, y_ppg.npy: du lieu da xu ly.
- feature_names_ppg.npy: thu tu dac trung.
- ppg_model.pkl: mo hinh da huan luyen.
- ppg_metrics.json: ket qua danh gia.
- ppg_confusion_matrix.png, explore/: bieu do.
- _pycache_/: bytecode Python da compile (theo cau truc thu muc mau ECG).

Cach chay (Python):
1) pip install numpy pandas scikit-learn matplotlib joblib
2) python preprocess.py
3) python train.py
4) python predict.py

Nhãn: 0 = diagnosis goc bang 0 (khong co benh); 1 = diagnosis goc 1–4 (co benh).
6 dong co gia tri thieu '?' duoc loai, con 297 mau.
Luu y: dataset la Clinical tabular, khong phai raw PPG waveform.
