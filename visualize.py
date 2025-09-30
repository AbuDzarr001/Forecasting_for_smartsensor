import os
import pandas as pd
import matplotlib.pyplot as plt
from config import OUT_DIR

def plot_anomalies(sensor_name, file_suffix="_if.csv", feature=None, save=True, show_table=True, max_rows=20):
    """
    Visualisasi hasil deteksi anomali dari file CSV dan tampilkan tabel anomali.
    """
    fname = f"anomalies_{sensor_name}{file_suffix}"
    path = os.path.join(OUT_DIR, fname)

    if not os.path.exists(path):
        print(f"[WARNING] File {path} tidak ditemukan.")
        return

    df = pd.read_csv(path, parse_dates=['time'], index_col='time')

    # Tentukan feature default jika tidak diberikan
    if feature is None:
        feature = df.select_dtypes("number").columns[0]

    if feature not in df.columns:
        print(f"[WARNING] Kolom {feature} tidak ditemukan dalam {fname}. Kolom tersedia: {df.columns}")
        return

    anomalies = df[df['if_anom'] == True]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[feature], label=f"{feature} ({sensor_name})", color="blue")
    plt.scatter(anomalies.index, anomalies[feature], color="red", label="Anomaly", marker="x")
    plt.title(f"Anomaly Detection - {sensor_name} ({feature})")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        out_path = os.path.join(OUT_DIR, f"plot_{sensor_name}_{feature.replace(' ', '_')}.png")
        plt.savefig(out_path)
        print(f"[INFO] Plot disimpan di {out_path}")
    else:
        plt.show()

    # Print ringkasan anomali
    print(f"[INFO] Jumlah anomali pada {sensor_name} ({feature}): {len(anomalies)}")

    if show_table and not anomalies.empty:
        print("\n[INFO] Data anomali (max", max_rows, "rows):")
        cols_to_show = [feature]
        if "if_score" in anomalies.columns:
            cols_to_show.append("if_score")
        print(anomalies[cols_to_show].head(max_rows))


def plot_all_sensors(file_suffix="_if.csv", max_features=2, save=True):
    """
    Batch mode: Visualisasi semua sensor & fitur utama secara otomatis.

    Parameters
    ----------
    file_suffix : str
        Jenis file hasil anomali, default "_if.csv".
    max_features : int
        Jumlah fitur numeric teratas yang divisualisasikan per sensor.
    save : bool
        Simpan plot ke file PNG (default True).
    """
    for fname in os.listdir(OUT_DIR):
        if not fname.endswith(file_suffix):
            continue
        sensor_name = fname.replace("anomalies_", "").replace(file_suffix, "")
        path = os.path.join(OUT_DIR, fname)
        df = pd.read_csv(path, parse_dates=['time'], index_col='time')
        num_cols = df.select_dtypes("number").columns
        for feature in num_cols[:max_features]:
            print(f"\n=== Sensor {sensor_name} | Feature {feature} ===")
            plot_anomalies(sensor_name, file_suffix=file_suffix, feature=feature, save=save, show_table=True)
