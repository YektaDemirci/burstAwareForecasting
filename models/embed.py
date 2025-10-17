import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        # It was; pe.require_grad, Appareantly a typo 
        pe.requires_grad = False # Not learnable

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Position actually works as [5000, 256]
        # div_term actually works as [5000, 256] very tricky!!!(5000,1) × (256,) → (5000,256)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    # Positional Embedding
    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Value embedding
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    # Value embedding
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x # 32 x 96 x 512 (from 7(c_in)->512)

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

# Temporal embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    # Temporal embedding
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h', detail_freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # Some changes here; Yekta
        if detail_freq == "10L":
            d_inp = 3
            self.embed = nn.Linear(d_inp, d_model)
        elif detail_freq == "100L":
            d_inp = 4
            self.embed = nn.Linear(d_inp, d_model)
        else:
            freq_map = {'h':4, 't':7, 's':3, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3, 'L':3}
            d_inp = freq_map[freq]
            self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

# class PeakEmbedding(nn.Module):
#     def __init__(self, d_model):
#         super(PeakEmbedding, self).__init__()
#         SN = 57927

#         p_df = pd.read_csv('/home/yektad/Desktop/informer/p2p_peaks.csv', header=None)
#         peaks = p_df.to_numpy().flatten()

#         d_df = pd.read_csv('/home/yektad/Desktop/informer/p2p_peaks_dist2peak.csv', header=None)
#         dist = d_df.to_numpy().flatten()

#         # Sinusoidal embedding
#         pe = torch.zeros(SN, d_model).float()
#         # It was; pe.require_grad, Appareantly a typo 
#         pe.requires_grad = False # Not learnable
#         for i in range(len(peaks)-1):

#             start = peaks[i]
#             end = peaks[i+1]
#             segment  = torch.from_numpy(dist[start:end]).float()

#             div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#             pe[start:end, 0::2] = torch.sin(segment[:, None] * div_term[None, :])
#             pe[start:end, 1::2] = torch.cos(segment[:, None] * div_term[None, :])
        
#         self.register_buffer('pe', pe)

#     def forward(self, x_size, indexes):
#         #if training:
#         # return torch.stack([self.pe[idx:idx+x_size, :] for idx in indexes], dim=0)
#         #if pred:
#         return torch.stack([self.pe[(idx+46342):(idx+x_size+46342), :] for idx in indexes], dim=0)

class PeakEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PeakEmbedding, self).__init__()
        # The number of peak feautres we need, needs to be updated manually for now!
        self.embed = nn.Linear(1, d_model)
        # self.embed = nn.Linear(4, d_model)

    def forward(self, batch_x_peak):
        # return self.embed(batch_x_peak)
        return self.embed(batch_x_peak.unsqueeze(-1))

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, detail_freq='h'):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq, detail_freq=detail_freq)
        self.peak_embedding = PeakEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, batch_x_peak=None, idx=None):
        # if (batch_x_peak is not None) and (batch_x_peak==1).any().item():
        #     print("HERE!")
        # The following is the original one
        # x = self.value_embedding(x) + self.position_embedding(x)
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + self.peak_embedding(batch_x_peak)
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) + self.peak_embedding(x.size(1), idx)

        # x = self.value_embedding(x) + self.position_embedding(x)
        # All the same size 32x96x512 + 32x96x512 + 32x96x512 
        
        return self.dropout(x)