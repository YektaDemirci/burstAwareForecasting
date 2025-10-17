import pandas as pd
import numpy as np

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
    # Lets make x2 of %99 for outliers
    quants.append(quants[-1]*2)
    return quants

def get_dist2peaks(peaks, SL):
    dist = np.full(SL, -1, dtype=int)
    last_special = -1
    for i in range(SL):
        if i in peaks:
            last_special = i
        if last_special != -1:
            dist[i] = i - last_special
    return dist

def peak_features(peak_path, SL):
    peaks = pd.read_csv(peak_path, header=None).to_numpy().flatten()
    # Values before the first were discarded, shift peaks accordingly
    peaks = peaks - peaks[0]
    dists2peak = get_dist2peaks(peaks, SL)

    dists = np.zeros(SL)
    for i in range(len(peaks)-1):

        start = peaks[i]
        end = peaks[i+1]

        segment = dists2peak[start:end]
        segment = np.log(1+segment)
        dists[start:end]   = segment

    return dists

# def peak_features(peak_path, peak_dist_path, SN, training_idx):
#     p_df = pd.read_csv(peak_path, header=None)
#     peaks = p_df.to_numpy().flatten()

#     d_df = pd.read_csv(peak_dist_path, header=None)
#     dist = d_df.to_numpy().flatten()

#     peak_distances = [peaks[i+1] - peaks[i] for i in range(len(peaks[peaks < training_idx])-1)]
#     quant_percents = [0.6, 0.90, 0.99]
#     quants = get_quantiles(peak_distances, quant_percents)

#     dists = np.zeros(SN)
#     # Max is 2172 for my data
#     segment_normalizer = 1107
#     # It was; pe.require_grad, Appareantly a typo 
#     for i in range(len(peaks)-1):

#         start = peaks[i]
#         end = peaks[i+1]

#         segment = dist[start:end]
#         # 0,1 scaling
#         segment = np.log(1+segment)
#         # segment = segment/segment_normalizer
#         # segment = np.clip(segment, None, 1)
#         # -1,1 scaling
#         # segment = 2 * (segment - segment.min()) / (segment.max() - segment.min()) - 1
#         # We wont know when the next peak is actually coming!
#         # segment = segment[::-1]
#         dists[start:end]   = segment

#     return dists

    # dists = np.zeros((len(quants), SN))

    # # It was; pe.require_grad, Appareantly a typo 
    # for i in range(len(peaks)-1):

    #     start = peaks[i]
    #     end = peaks[i+1]

    #     for idx, quant in enumerate(quants):
    #         segment = dist[start:end]
    #         # 0,1 scaling
    #         segment = np.exp(-segment/quant)
    #         # segment = np.clip(segment, None, 1)
    #         # -1,1 scaling
    #         # segment = 2 * (segment - segment.min()) / (segment.max() - segment.min()) - 1
    #         # We wont know when the next peak is actually coming!
    #         # segment = segment[::-1]
    #         dists[idx][start:end] = (-1)*(segment)

    # return dists.transpose(1,0)