from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def isolation_forest_detection(df, feature_cols, contamination=0.01, train_frac=0.8):
    df_proc = df.copy()
    feat = [c for c in feature_cols if c in df_proc.columns]
    X = df_proc[feat].values
    n_train = int(len(df_proc) * train_frac)
    scaler = StandardScaler().fit(X[:n_train])
    X_scaled = scaler.transform(X)
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(scaler.transform(X[:n_train]))
    scores = clf.decision_function(X_scaled)
    preds = clf.predict(X_scaled)
    df_proc['if_score'] = scores
    df_proc['if_anom'] = (preds == -1)
    return df_proc, clf, scaler
