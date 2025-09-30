import pandas as pd
import numpy as np

def rca_report(df, sensor_name, feature_cols=None, score_col="if_score", flag_col="if_anom", top_k=1):
    """
    Root Cause Analysis sederhana:
    - Identifikasi kolom (fitur sensor) mana yang paling menyumbang anomali.
    - Gunakan z-score untuk mengukur deviasi terhadap distribusi normal.
    
    Args:
        df : DataFrame hasil anomaly detection
        sensor_name : nama sensor
        feature_cols : daftar kolom fitur numerik
        score_col : nama kolom skor anomaly
        flag_col : nama kolom flag anomali
        top_k : jumlah penyebab utama ditampilkan
    
    Returns:
        report (list of str) : laporan penyebab anomali
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes("number").columns.drop([score_col], errors="ignore")

    anomalies = df[df[flag_col] == True]
    reports = []

    for t, row in anomalies.iterrows():
        z_scores = {}
        for col in feature_cols:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                z = abs((row[col] - mean) / std)
                z_scores[col] = z
        # sort penyebab utama
        main_causes = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        main_text = ", ".join([f"{c} (z={z:.2f})" for c, z in main_causes])
        reports.append(f"[{t}] RCA {sensor_name}: {main_text} → kemungkinan penyebab utama")

    return reports
