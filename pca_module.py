import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_windows(arr, window_size=6):
    n, d = arr.shape
    windows, idx = [], []
    for i in range(window_size, n+1):
        windows.append(arr[i-window_size:i].flatten())
        idx.append(i-1)
    return np.array(windows), idx

def pca_window_detection(df, feature_cols, window_size=6, var_threshold=0.95):
    arr = df[feature_cols].values
    W, idx = create_windows(arr, window_size)
    sc = StandardScaler().fit(W)
    W_s = sc.transform(W)
    pca = PCA(n_components=var_threshold, svd_solver='full').fit(W_s)
    W_p = pca.transform(W_s)
    W_rec = pca.inverse_transform(W_p)
    rec_err = ((W_s - W_rec)**2).mean(axis=1)
    threshold = rec_err.mean() + 3*rec_err.std()
    timestamps = [df.index[i] for i in idx]
    out = pd.DataFrame({'recon_error': rec_err, 'anom_pca': rec_err > threshold}, index=timestamps)
    return out, pca, sc, float(threshold)
