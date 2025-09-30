import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from config import OUT_DIR

def prophet_forecast(df, feature="Temperature", steps=24, threshold=None, sensor_name="Sensor"):
    """
    Forecast 1 fitur sensor dengan Prophet.
    """

    if "if_anom" in df.columns:
        df_train = df.loc[~df["if_anom"], [feature]].dropna()
    else:
        df_train = df[[feature]].dropna()

    if len(df_train) < 30:
        print(f"[WARNING] Data {sensor_name}-{feature} terlalu sedikit untuk Prophet.")
        return None, []

    # Ubah ke format Prophet: ds, y
    df_prophet = df_train.reset_index().rename(columns={"time": "ds", feature: "y"})

    # Model Prophet
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_prophet)

    # Buat horizon
    future = model.make_future_dataframe(periods=steps, freq=pd.infer_freq(df_train.index) or "H")
    forecast = model.predict(future)

    # Ambil hanya kolom yhat
    forecast_series = forecast.set_index("ds")["yhat"].iloc[-steps:]

    # Plot hasil
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df[feature], label="Actual", color="black")
    plt.plot(forecast_series.index, forecast_series.values, label="Prophet Forecast", color="blue")
    if threshold:
        plt.axhline(y=threshold, color="orange", linestyle="--", label="Threshold")
    plt.title(f"Prophet Forecast {sensor_name} - {feature}")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)

    out_plot = os.path.join(OUT_DIR, f"prophet_{sensor_name}_{feature}.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"[INFO] Plot Prophet disimpan di {out_plot}")

    # Early warning
    warnings = []
    if threshold:
        exceed = forecast_series[forecast_series > threshold]
        if not exceed.empty:
            t_exceed = exceed.index[0]
            val = exceed.iloc[0]
            warnings.append(
                f"⚠️ {sensor_name}-{feature}: prediksi {val:.2f} > threshold {threshold} pada {t_exceed}"
            )

    return forecast_series, warnings


def prophet_batch_forecast(all_data, steps=24, thresholds=None):
    """
    Forecast semua sensor & parameter numerik dengan Prophet.
    """
    all_forecasts = {}
    all_warnings = []

    for sensor, df in all_data.items():
        numeric_cols = df.select_dtypes("number").columns
        for feature in numeric_cols:
            threshold = thresholds.get(feature) if thresholds else None
            forecast, warnings = prophet_forecast(
                df, feature=feature, steps=steps, threshold=threshold, sensor_name=sensor
            )
            if forecast is not None:
                all_forecasts[(sensor, feature)] = forecast
            if warnings:
                all_warnings.extend(warnings)

    out_csv = os.path.join(OUT_DIR, "prophet_batch_forecasts.csv")
    df_forecasts = pd.DataFrame(all_forecasts)
    df_forecasts.to_csv(out_csv)
    print(f"[INFO] Semua hasil forecast Prophet disimpan di {out_csv}")

    return df_forecasts, all_warnings
