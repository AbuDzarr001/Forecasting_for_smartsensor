import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import OUT_DIR

def sarima_forecast(df, feature="Temperature", steps=12, order=(1,0,1),
                    seasonal_order=(1,1,1,24), threshold=None, sensor_name="Sensor"):
    """
    Forecast 1 fitur sensor dengan SARIMA.
    """

    if "if_anom" in df.columns:
        df_train = df.loc[~df["if_anom"], feature].dropna()
    else:
        df_train = df[feature].dropna()

    if len(df_train) < 30:
        print(f"[WARNING] Data {sensor_name}-{feature} terlalu sedikit untuk SARIMA.")
        return None, []

    try:
        model = SARIMAX(df_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
    except Exception as e:
        print(f"[ERROR] SARIMA gagal untuk {sensor_name}-{feature}: {e}")
        return None, []

    freq = pd.infer_freq(df_train.index) or (df_train.index[-1] - df_train.index[-2])
    forecast_index = pd.date_range(df_train.index[-1], periods=steps+1, freq=freq)[1:]
    forecast = pd.Series(forecast.values, index=forecast_index)

    # Plot hasil
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df[feature], label="Actual", color="black")
    plt.plot(forecast.index, forecast, label="SARIMA Forecast", color="blue")
    if threshold:
        plt.axhline(y=threshold, color="orange", linestyle="--", label="Threshold")
    plt.title(f"SARIMA Forecast {sensor_name} - {feature}")
    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)

    out_plot = os.path.join(OUT_DIR, f"sarima_{sensor_name}_{feature}.png")
    plt.savefig(out_plot)
    plt.close()
    print(f"[INFO] Plot SARIMA disimpan di {out_plot}")

    warnings = []
    if threshold:
        exceed = forecast[forecast > threshold]
        if not exceed.empty:
            t_exceed = exceed.index[0]
            val = exceed.iloc[0]
            warnings.append(
                f"⚠️ {sensor_name}-{feature}: prediksi {val:.2f} > threshold {threshold} pada {t_exceed}"
            )

    return forecast, warnings


def sarima_batch_forecast(all_data, steps=12, thresholds=None, order=(1,0,1), seasonal_order=(1,1,1,24)):
    """
    Forecast semua sensor & parameter numerik dengan SARIMA.
    """
    all_forecasts = {}
    all_warnings = []

    for sensor, df in all_data.items():
        numeric_cols = df.select_dtypes("number").columns
        for feature in numeric_cols:
            threshold = thresholds.get(feature) if thresholds else None
            forecast, warnings = sarima_forecast(
                df, feature=feature, steps=steps,
                order=order, seasonal_order=seasonal_order,
                threshold=threshold, sensor_name=sensor
            )
            if forecast is not None:
                all_forecasts[(sensor, feature)] = forecast
            if warnings:
                all_warnings.extend(warnings)

    out_csv = os.path.join(OUT_DIR, "sarima_batch_forecasts.csv")
    df_forecasts = pd.DataFrame(all_forecasts)
    df_forecasts.to_csv(out_csv)
    print(f"[INFO] Semua hasil forecast SARIMA disimpan di {out_csv}")

    return df_forecasts, all_warnings
