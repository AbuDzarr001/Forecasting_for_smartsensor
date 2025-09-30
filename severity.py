def severity_scoring(df, score_col='if_score', flag_col='if_anom',
                     multi_sensor_flags=None, threshold_low=-0.1, threshold_high=-0.3):
    df = df.copy()
    df['severity'] = "Normal"
    for i in df.index:
        if not df.loc[i, flag_col]:
            continue
        score = df.loc[i, score_col]
        level = "Low"
        if score < threshold_low:
            level = "Medium"
        if score < threshold_high:
            level = "High"
        if multi_sensor_flags:
            count_anom = sum(bool(flags.loc[i]) for flags in multi_sensor_flags.values() if i in flags.index)
            if count_anom >= 2 and level == "Low":
                level = "Medium"
            if count_anom >= 3:
                level = "High"
        df.at[i, 'severity'] = level
    return df
