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

def get_scaled_peaks(peak_path, SL, extend_tail=False):
    # extend_tail=False keeps the original behaviour, where every position after
    # the last retained peak has dist 0 (indistinguishable from a burst).
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

    if extend_tail:
        dists[peak_idx[-1]:] = np.log(1 + dists2peak[peak_idx[-1]:])

    return dists


def causal_burst_events(x, train_end, k=128, h=2.5, gamma=2.5):
    """Online analogue of the burst definition in get_bursts.py: x_t is compared
    with its k *past* samples only, and all thresholds are fitted on x[:train_end].
    Whether t is an event is therefore decidable at time t."""
    x = np.asarray(x, dtype=float).flatten()
    csum = np.concatenate([[0.], np.cumsum(x)])
    score = np.zeros(len(x))
    score[k:] = x[k:] - (csum[k:-1] - csum[:-k-1]) / k

    train_score = score[k:train_end]
    pos = train_score[train_score > 0]
    tau_score = pos.mean() + h * pos.std()
    tau_x = x[:train_end].mean() + gamma * x[:train_end].std()
    return (score > tau_score) & (x > tau_x)


def get_causal_dists(x, train_end, k=128, h=2.5, gamma=2.5):
    """log(1 + samples since the last causal event). Before the first event the
    distance is counted from the start of the series."""
    events = causal_burst_events(x, train_end, k, h, gamma)
    dist = np.zeros(len(events))
    last = 0
    for t, is_event in enumerate(events):
        if is_event:
            last = t
        dist[t] = t - last
    return np.log1p(dist)


def decoder_dists(dists, r_begin, s_end, r_end, mode):
    """Distance feature for the decoder window. Only mode='leak' reads
    dists[s_end:r_end], i.e. the distance values over the forecast horizon."""
    if mode in ('leak', 'none'):  # 'none' is all zeros, so slicing cannot leak
        return dists[r_begin:r_end]
    known = dists[r_begin:s_end]
    horizon = r_end - s_end
    # extrap/causal: assume no burst happens over the horizon
    future = np.log1p(np.expm1(dists[s_end - 1]) + np.arange(1, horizon + 1))
    return np.concatenate([known, future])