import pandas as pd

def print_forecast_summary(df_forecasts, steps=3):
    """
    Cetak ringkasan forecast n langkah ke depan (default 3) 
    untuk semua sensor dan parameter.
    
    df_forecasts = DataFrame hasil batch_forecast (SARIMA/Prophet)
    """
    print(f"\n=== RINGKASAN FORECAST {steps} LANGKAH KE DEPAN ===")
    
    # Ambil hanya n langkah terakhir
    tail_forecasts = df_forecasts.tail(steps)
    
    for (sensor, feature) in df_forecasts.columns:
        print(f"\n--- {sensor} - {feature} ---")
        vals = tail_forecasts[(sensor, feature)]
        for t, v in vals.items():
            print(f"{t}: {v:.2f}")

    print("\n=== SELESAI ===")
