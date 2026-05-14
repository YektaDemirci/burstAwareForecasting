import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def peak_features2(peak_path, peak_dist_path, SN):
    p_df = pd.read_csv(peak_path, header=None)
    peaks = p_df.to_numpy().flatten()

    d_df = pd.read_csv(peak_dist_path, header=None)
    dist = d_df.to_numpy().flatten()

    dists = np.zeros(SN)
    # It was; pe.require_grad, Appareantly a typo 
    for i in range(len(peaks)-1):

        start = peaks[i]
        end = peaks[i+1]

        segment = dist[start:end]
        dists[start:end:2]   = np.sin(segment[0::2])
        dists[start+1:end:2] = np.cos(segment[1::2])

        # div_term = np.exp(np.arange(0, d_segment, 2, dtype=np.float32) * -(np.log(10000.0) / d_segment))
        # # Following will fail, fix
        # dists[start:end:2] = np.sin(segment * div_term)
        # dists[start+1:end:2] = np.cos(segment * div_term)

    return dists

def peak_idx(peak_path, SN):
    p_df = pd.read_csv(peak_path, header=None)
    peak_idx = p_df.to_numpy().flatten()
    peaks = np.zeros(SN)
    peaks[peak_idx] = 1
    return peaks

def get_quantiles(peaks, quant_percents):
    quants = []
    for quant_per in quant_percents:
        quants.append(np.quantile(peaks, quant_per))
    return quants


def peak_features(peak_path, peak_dist_path, SN, training_idx):
    p_df = pd.read_csv(peak_path, header=None)
    peaks = p_df.to_numpy().flatten()

    d_df = pd.read_csv(peak_dist_path, header=None)
    dist = d_df.to_numpy().flatten()

    quant_percents = [70, 90, 95]
    quants = get_quantiles(peaks[peaks < training_idx], quant_percents)

    dists = [np.zeros(SN)*len(quant_percents)]
    # Max is 2172 for my data
    segment_normalizer = 1107
    # It was; pe.require_grad, Appareantly a typo 
    for i in range(len(peaks)-1):

        start = peaks[i]
        end = peaks[i+1]

        for idx, quant in enumerate(quants):
            segment = dist[start:end]
            # 0,1 scaling
            segment = segment/segment_normalizer
            segment = np.clip(segment, None, 1)
            # -1,1 scaling
            # segment = 2 * (segment - segment.min()) / (segment.max() - segment.min()) - 1
            # We wont know when the next peak is actually coming!
            segment = segment[::-1]
            dists[start:end]   = segment

    return dists