import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from config import OUT_DIR

def forecast_sensor(df, feature="Temperature", steps=12, order=(2,1,2), threshold=None, sensor_name="Sensor"):
    """
    Forecast nilai sensor menggunakan ARIMA.
    """
    series = df[feature].dropna()
    if len(series) < 10:
        print(f"[WARNING] Data {sensor_name}-{feature} terlalu sedikit untuk ARIMA.")
        return None, []

    # fit ARIMA
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
    except Exception as e:
        print(f"[ERROR] ARIMA gagal untuk {sensor_name}-{feature}: {e}")
        return None, []

    # prediksi ke depan
    forecast = model_fit.forecast(steps=steps)
    freq = pd.infer_freq(series.index) or (series.index[-1] - series.index[-2])
    forecast_index = pd.date_range(series.index[-1], periods=steps+1, freq=freq)
    forecast = pd.Series(forecast.values, index=forecast_index[1:])

    # plot hasil
    plt.figure(figsize=(12,5))
    plt.plot(series.index, series, label="Historical")
    plt.plot(forecast.index, forecast, label="Forecast", color="red")
    if threshold:
        plt.axhline(y=threshold, color="orange", linestyle="--", label="Threshold")
    plt.title(f"Forecast {sensor_name} - {feature}")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_plot = os.path.join(OUT_DIR, f"forecast_{sensor_name}_{feature}.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"[INFO] Plot disimpan di {out_plot}")

    # early warning check
    warnings = []
    if threshold:
        exceed = forecast[forecast > threshold]
        if not exceed.empty:
            t_exceed = exceed.index[0]
            val = exceed.iloc[0]
            warnings.append(f"⚠️ Early Warning: {feature} diprediksi {val:.2f} > threshold {threshold} pada {t_exceed}")

    return forecast, warnings


def batch_forecast(all_data, steps=12, thresholds=None, order=(2,1,2)):
    """
    Jalankan forecasting untuk semua sensor & semua fitur numerik.
    
    Args:
        all_data : dict {sensor_name: DataFrame}
        steps : langkah prediksi ke depan
        thresholds : dict {feature: threshold}
        order : parameter ARIMA
    """
    all_warnings = []
    all_forecasts = {}

    for sensor, df in all_data.items():
        numeric_cols = df.select_dtypes("number").columns
        for feature in numeric_cols:
            threshold = thresholds.get(feature) if thresholds else None
            forecast, warnings = forecast_sensor(df, feature=feature, steps=steps, order=order, threshold=threshold, sensor_name=sensor)
            if forecast is not None:
                all_forecasts[(sensor, feature)] = forecast
            if warnings:
                all_warnings.extend([f"{sensor}-{feature}: {w}" for w in warnings])

    # simpan forecast ke CSV
    out_csv = os.path.join(OUT_DIR, "batch_forecasts.csv")
    df_forecasts = pd.DataFrame(all_forecasts)
    df_forecasts.to_csv(out_csv)
    print(f"[INFO] Semua forecast disimpan di {out_csv}")

    return df_forecasts, all_warnings
