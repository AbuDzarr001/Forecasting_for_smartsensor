import pandas as pd

def preprocess_data(df, features=None, resample_rule='10T'):
    df2 = df.copy()
    if features is None:
        features = [c for c in df2.select_dtypes("number").columns]
    df2 = df2[features].apply(pd.to_numeric, errors='coerce')
    df2 = df2.resample(resample_rule).mean()
    df2 = df2.interpolate(method='time').ffill().bfill()
    for c in features:
        df2[f'{c}_rmean'] = df2[c].rolling(3, min_periods=1).mean()
        df2[f'{c}_rstd'] = df2[c].rolling(3, min_periods=1).std().fillna(0)
        df2[f'{c}_delta'] = df2[c].diff().fillna(0)
    return df2
