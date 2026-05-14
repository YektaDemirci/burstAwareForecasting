import pandas as pd
import numpy as np
# from sklearn.preprocessing import PowerTransformer

def get_binary_peaks(peak_path, SN):
    peak_idx = pd.read_csv(peak_path, header=None).to_numpy().flatten()
    # Shift according to the first peak, discard the last
    peak_idx = (peak_idx - peak_idx[0])[:-1]
    peaks = np.zeros(SN)
    peaks[peak_idx] = 1
    return peaks

def get_dist2peaks(peaks, SL):
    dist = np.full(SL, -1, dtype=int)
    last_special = -1
    for i in range(SL):
        if i in peaks:
            last_special = i
        if last_special != -1:
            dist[i] = i - last_special
    return dist

def get_scaled_peaks(peak_path, SL):
    peak_idx = pd.read_csv(peak_path, header=None).to_numpy().flatten()
    # Shift according to the first peak, discard the last
    peak_idx = (peak_idx - peak_idx[0])[:-1]
    dists2peak = get_dist2peaks(peak_idx, SL)

    dists = np.zeros(SL)
    for i in range(len(peak_idx)-1):

        start = peak_idx[i]
        end = peak_idx[i+1]

        segment = dists2peak[start:end]
        segment = np.log(1+segment)
        # pt = PowerTransformer(method='yeo-johnson')
        # segment = pt.fit_transform(segment.reshape(-1,1)).flatten()
        dists[start:end] = segment

    return dists