import pandas as pd
import os
from config import FILE_PATH
from data_loader import load_data
from preprocessing import preprocess_data
from isolation_forest_module import isolation_forest_detection
from pca_module import pca_window_detection
from severity import severity_scoring
from utils import save_results_csv, notify_telegram
from visualize import plot_all_sensors
from rca import rca_report
from forecast import batch_forecast
from sarima_forecast import sarima_batch_forecast
from prophet_forecast import prophet_batch_forecast
from print_forecast_summary import print_forecast_summary

def main():
    dfs = load_data(FILE_PATH)
    summary = []
    severity_all = {}

    for sheet, df_raw in dfs.items():
        # Pilih fitur sesuai sensor
        if sheet.upper() == "S2103":
            features = ["CO2","Temperature","Humidity"]
        elif sheet.upper() == "S2120":
            features = ["Air Temperature","Air Humidity","Light Intensity","Uv Index","Wind Speed","Rain Gauge","Barometric Pressure"]
        elif sheet.upper() == "WS302":
            features = [c for c in df_raw.columns if "Noise" in c or "LA" in c][:1]
        else:
            features = [c for c in df_raw.select_dtypes("number").columns][:5]

        if not features: continue
        df_proc = preprocess_data(df_raw, features=features)

        # Isolation Forest
        df_if, clf, sc = isolation_forest_detection(df_proc, features, contamination=0.01)
        fname = f"anomalies_{sheet}_if.csv"
        save_results_csv(df_if, fname)
        summary.append((sheet, "IF", df_if['if_anom'].sum(), len(df_if)))
        severity_all[sheet] = df_if['if_anom']

        # PCA khusus weather
        if sheet.upper() == "S2120":
            pca_out, pca, sc, th = pca_window_detection(df_proc, features)
            save_results_csv(pca_out, f"anomalies_{sheet}_pca.csv")
            summary.append((sheet, "PCA", pca_out['anom_pca'].sum(), len(pca_out)))

    # Severity scoring untuk CO2 (contoh)
    if "S2103" in dfs:
        df_co2 = pd.read_csv("D:/S2/Smart City/iot_anomaly_pipeline/if_pipeline_outputs/anomalies_S2103_if.csv", parse_dates=['time'], index_col='time')
        multi_flags = {k:v for k,v in severity_all.items() if k!="S2103"}
        df_sev = severity_scoring(df_co2, score_col='if_score', flag_col='if_anom',
                                  multi_sensor_flags=multi_flags)
        save_results_csv(df_sev, "anomalies_S2103_with_severity.csv")

    # Summary
    df_sum = pd.DataFrame(summary, columns=["Sheet","Method","Anomalies","Total"])
    save_results_csv(df_sum, "if_pipeline_summary.csv")
    print(df_sum)

    print("\n=== Batch Visualisasi Semua Sensor ===")
    plot_all_sensors(file_suffix="_if.csv", max_features=2)
    
    print("\n=== RCA & Severity Analysis ===")

    df = pd.read_csv("D:/S2/Smart City/iot_anomaly_pipeline/if_pipeline_outputs/anomalies_S2103_if.csv", parse_dates=["time"], index_col="time")

    # RCA
    rca_results = rca_report(df, "S2103")
    for r in rca_results[:5]:  # tampilkan 5 sampel
        print(r)
    
        
    print("\n=== BATCH FORECASTING ===")

    # Ambil langsung dari dfs hasil load_data()
    df_S2103 = dfs.get("S2103")
    df_S2120 = dfs.get("S2120")
    df_WS302 = dfs.get("WS302")

    # Dict all_data untuk batch_forecast
    all_data = {
        "S2103": df_S2103,
        "S2120": df_S2120,
        "WS302": df_WS302
    }

    # tentukan threshold untuk fitur tertentu
    thresholds = {
        "Temperature": 30,
        "CO2": 1500,
        "Noise_dBA": 85
    }

    df_forecasts, warnings = batch_forecast(all_data, steps=12, thresholds=thresholds)

    if warnings:
        print("\n=== EARLY WARNINGS ===")
        for w in warnings:
            print(w)

    print("\n=== SARIMA BATCH FORECASTING ===")

    all_data = {name: df for name, df in dfs.items()}

    thresholds = {
        "Temperature": 30,
        "CO2": 1500,
        "Humidity": 90,
        "Noise_dBA": 85
    }

    # Panggil fungsi batch, bukan fungsi tunggal
    df_forecasts, warnings = sarima_batch_forecast(all_data, steps=24, thresholds=thresholds)

    if warnings:
        print("\n=== EARLY WARNINGS (SARIMA) ===")
        for w in warnings:
            print(w)
            
    print("\n=== PROPHET BATCH FORECASTING ===")

    all_data = {name: df for name, df in dfs.items()}

    thresholds = {
        "Temperature": 30,
        "CO2": 1500,
        "Humidity": 90,
        "Noise_dBA": 85
    }

    df_forecasts, warnings = prophet_batch_forecast(all_data, steps=24, thresholds=thresholds)

    if warnings:
        print("\n=== EARLY WARNINGS (PROPHET) ===")
        for w in warnings:
            print(w)
            

if __name__ == "__main__":
    main()
