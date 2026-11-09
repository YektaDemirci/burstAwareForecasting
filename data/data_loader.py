import os
import numpy as np
import pandas as pd
import math

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.peakfeatures import get_scaled_peaks, get_binary_peaks, get_causal_dists, decoder_dists

import warnings
warnings.filterwarnings('ignore')

def build_dists(peak_feature, peak_full_path, n, series, num_train):
    if peak_feature == 'none':
        return np.zeros(n)
    if peak_feature == 'causal':
        return get_causal_dists(series, num_train)
    # leak/extrap: history comes from the centered labels, which look k samples ahead.
    return get_scaled_peaks(peak_full_path, n, extend_tail=peak_feature != 'leak')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None, org_mean=0, org_std=1):
        print("ETT_min is triggered")
        print(f'freq in Dataset type {freq}')
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # I wonder what these borders are, ok got it
        # Every 15 minutes !
        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Takes all features
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_mark, seq_y_mark, seq_y_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# freq is custom-made to "L" to support millisecond
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='PARETOm.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='L', cols=None, org_mean=0, org_std=1, peak_path="peaks.csv", peak_feature='causal'):
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.peak_path = peak_path
        self.peak_feature = peak_feature
        self.__read_data__()

    def __read_data__(self):
        
        self.scaler = StandardScaler()
        peaks_raw = pd.read_csv(os.path.join(self.root_path, self.peak_path), header=None).to_numpy().flatten()
        # Consider only from 1st peak to the last
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path)).iloc[peaks_raw[0]:peaks_raw[-1]+1]
        # Shift peak index
        peaks_raw = peaks_raw - peaks_raw[0]
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        # Determines what part of the data you are interested in!
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # It scales everything according to the training datas mean and variance
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        peak_full_path = os.path.join(self.root_path, self.peak_path)
        dists = build_dists(self.peak_feature, peak_full_path, df_raw.shape[0], data[:, -1], num_train)
        peaks = get_binary_peaks(peak_full_path, df_raw.shape[0])
        self.dists = dists[border1:border2]
        self.peaks = peaks[border1:border2]

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    # Gets train, val, test seperately 
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_peaks = self.dists[s_begin:s_end]

        # Reads the true future distance only when peak_feature='leak'.
        seq_y_peaks = decoder_dists(self.dists, r_begin, s_end, r_end, self.peak_feature)

        seq_x_p = self.peaks[s_begin:s_end]
        # Burst labels: the training target and scoring only, never a model input.
        seq_y_p = self.peaks[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_peaks, seq_y_peaks, seq_x_p, seq_y_p
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, org_mean=0., org_std=1., peak_path="peaks.csv", peak_feature='causal'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.scale = True
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.peak_path = peak_path
        self.peak_feature = peak_feature
        self.__read_data__(org_mean, org_std)

    def __read_data__(self, org_mean, org_std):
        self.scaler = StandardScaler()
        peaks_raw = pd.read_csv(os.path.join(self.root_path, self.peak_path), header=None).to_numpy().flatten()
        # Consider only from 1st peak to the last
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path)).iloc[peaks_raw[0]:peaks_raw[-1]+1]
        # Shift peak index
        peaks_raw = peaks_raw - peaks_raw[0]
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        # The border shall end considering the seq_len
        border2s = [num_train, num_train+num_vali, len(df_raw)]

        # 2 comes from type_map = {'train':0, 'val':1, 'test':2}
        border1 = border1s[2]
        border2 = border2s[2]

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Our case is 10 millisecond!
        # This is determined from exp_informer when the Dataset model is called      
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        # tmp_stamp = df_raw[['date']][border1:border2]
        # tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)        
        # df_stamp = pd.DataFrame(columns = ['date'])
        # df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        # data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # This is scaled anyways
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        peak_full_path = os.path.join(self.root_path, self.peak_path)
        dists = build_dists(self.peak_feature, peak_full_path, df_raw.shape[0], data[:, -1], num_train)
        peaks = get_binary_peaks(peak_full_path, df_raw.shape[0])
        self.dists = dists[border1:border2]
        self.peaks = peaks[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            # data_x and data_y are the same, ok data_y is used later to compare, so unscaled is used that is why
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            # The following was somehow ignored
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_peaks = self.dists[s_begin:s_end]

        # Reads the true future distance only when peak_feature='leak'.
        seq_y_peaks = decoder_dists(self.dists, r_begin, s_end, r_end, self.peak_feature)

        seq_x_p = self.peaks[s_begin:s_end]
        # Burst labels: the training target and scoring only, never a model input.
        seq_y_p = self.peaks[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_peaks, seq_y_peaks, seq_x_p, seq_y_p
    
    def __len__(self):
        # return len(self.data_x) - self.seq_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
