from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F


###########################################################
### Self implemented losses ###
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

def _pinball_loss(input: Tensor, target: Tensor, tau: float, reduction: str = "mean") -> Tensor:
    err = target - input
    tau = 0.5
    loss = torch.maximum(tau * err, (tau - 1) * err)
    if reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class PinballLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, tau: float = 0.5, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(PinballLoss, self).__init__(size_average, reduce, reduction)
        if not (0 < tau < 1):
            raise ValueError("tau should be in (0,1)")
        self.tau = 0.5

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _pinball_loss(input, target, self.tau, reduction=self.reduction)

def mixed_pinball_loss(input: Tensor, target: Tensor, tau: float, reduction: str = "mean") -> Tensor:
    err = target - input
    tau1 = 0.9
    loss1 = torch.maximum(tau1 * err, (tau1 - 1) * err)
    # tau2 = 0.5
    loss2 = torch.pow(err, 2)
    # loss2 = torch.maximum(tau2 * err, (tau2 - 1) * err)
    loss = (loss1+loss2)/2
    if reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class MixedPinballLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, tau: float = 0.8, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(MixedPinballLoss, self).__init__(size_average, reduce, reduction)
        if not (0 < tau < 1):
            raise ValueError("tau should be in (0,1)")
        self.tau = 0.8

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return mixed_pinball_loss(input, target, self.tau, reduction=self.reduction)
    

# def _qlike_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
#     """
#     QLIKE loss for variance/volatility forecasting.
#     input: forecasted variance (positive)
#     target: true variance (positive)
#     """
#     if torch.any(input <= 0) or torch.any(target <= 0):
#         raise ValueError("QLIKE requires positive inputs and targets.")

#     loss = (target / input) - torch.log(target / input) - 1.0

#     if reduction == "mean":
#         return loss.mean()
#     else:
#         raise ValueError(f"Invalid reduction: {reduction}")

# class QLIKELoss(_Loss):
#     __constants__ = ['reduction']

#     def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
#         super(QLIKELoss, self).__init__(size_average, reduce, reduction)

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         return _qlike_loss(input, target, reduction=self.reduction)
    
def _exp_bregman_loss(input: Tensor, target: Tensor, a: float = -2.0, reduction: str = "mean") -> Tensor:
    if a == 0:
        raise ValueError("a cannot be zero for exponential Bregman loss.")

    # loss = (2/a**2)*(torch.exp(a*torch.tensor(target))-torch.exp(a*input))-(2/a*torch.exp(a*input)*(torch.tensor(target)-input)) 
    loss = (2/a**2)*(torch.exp(a*target)-torch.exp(a*input))-(2/a*torch.exp(a*input)*(target-input)) 

    if reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class ExpBregmanLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, a: float = -2.0, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(ExpBregmanLoss, self).__init__(size_average, reduce, reduction)
        self.a = a

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _exp_bregman_loss(input, target, a=self.a, reduction=self.reduction)


def _hom_bregman_loss(input: Tensor, target: Tensor, k: float = 1.1, reduction: str = "mean") -> Tensor:
    if k <= 1:
        raise ValueError("k must be greater than 1 for homogeneous Bregman loss.")
    if torch.any(input <= 0) or torch.any(target <= 0):
        raise ValueError("Homogeneous Bregman loss requires positive inputs.")

    loss = abs(target).pow(k) - abs(input).pow(k)  - (k* torch.sign(input)*abs(input).pow(k-1)*(target-input))
    # exprLin = (torch.abs(target)**k) - (torch.abs(inputs)**k) - (k*torch.sign(inputs)*(torch.abs(inputs)**(k-1))*(target-inputs))

    if reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class HomBregmanLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, k: float = 1.1, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(HomBregmanLoss, self).__init__(size_average, reduce, reduction)
        if k <= 1:
            raise ValueError("k must be > 1")
        self.k = k

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return _hom_bregman_loss(input, target, k=self.k, reduction=self.reduction)

###########################################################

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.detail_freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        # print("!!HERE!!")
        # print(self.args.detail_freq)
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        # See affect of 0 and 1
        timeenc = 0 if args.embed!='timeF' else 1
        
        # Detail flag is including full 10L!
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.detail_freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            # Pred doesnt have train/test/val at all
            Data = Dataset_Pred
        else:
            # shuffle_flag is SUCH A GAME CHANGER HERE YEKTA !!!
            # print("HERE!")
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.detail_freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            org_mean=float(args.org_data_mean),
            org_std=float(args.org_data_std),
        )
        # print(flag, len(data_set))
        # print(f'Batch size is {batch_size}')
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    # Yekta loss function change here
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        # criterion = PinballLoss()
        # criterion = ExpBregmanLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx) in enumerate(vali_loader):
            pred, true, peak_logits = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        criterionMixed = MixedPinballLoss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_batch = []
        loss_epoch = []

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # This is where the training happens!

            def dilate_binary_mask(mask, k=5):
                # mask: [B, L] with {0,1}
                # use max-pool to grow a window around peaks
                x = mask.unsqueeze(1).float()                   # [B,1,L]
                y = F.max_pool1d(x, kernel_size=k, stride=1, padding=k//2)
                return (y.squeeze(1) > 0.5).float()             # [B,L]

            def asymmetric_mse(pred, true, y_peak, gamma=3.0, under_w=2.0, k=5):
                # pred,true: [B,L,C], y_peak: [B,L] in {0,1}
                err = (pred - true).float()
                w_peak = 1.0 + gamma * dilate_binary_mask(y_peak, k=k).unsqueeze(-1)  # [B,L,1]
                # extra penalty if under-pred (err < 0)
                dtype, device = err.dtype, err.device
                w_asym = torch.where(
                                    err < 0,
                                    torch.full_like(err, fill_value=under_w, dtype=dtype, device=device),
                                    torch.ones_like(err, dtype=dtype, device=device)
                )
                return (w_peak * w_asym * err.pow(2)).mean()
            
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx) in enumerate(train_loader):

                iter_count += 1
                
                model_optim.zero_grad()
                pred, true, peak_logits= self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx)
                
                # # Mixed loss
                y_peak_gt = batch_y_idx[:, -self.args.pred_len:].float().to(self.device)

                loss_forecast = asymmetric_mse(pred, true, y_peak_gt, gamma=3.0, under_w=2.0, k=5)
                # loss_forecast = criterion(pred, true)

                # cls_loss = F.binary_cross_entropy_with_logits(peak_logits.squeeze(-1), y_peak_gt)

                pos = y_peak_gt.sum()
                neg = y_peak_gt.numel() - pos

                pos_w = (neg / pos.clamp_min(1.)).detach()
                cls_loss2 = F.binary_cross_entropy_with_logits(
                    peak_logits.squeeze(-1), y_peak_gt, pos_weight=pos_w
                )

                alpha = 0.5
                loss = loss_forecast + (alpha*cls_loss2)

                # original way!
                # loss = criterion(pred, true)
                
                train_loss.append(loss.item())
                loss_batch.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            loss_epoch.append(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx) in enumerate(test_loader):
            pred, true, peak_logits = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        # folder_path = './results/' + setting +'/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        # 64_32 -> 48 fails because first 64 is used in encoder, use max_length-seq_len
        cut_off = (11850-455)
        joined_path = os.path.join(pred_loader.dataset.root_path, pred_loader.dataset.data_path)
        data = pd.read_csv(joined_path, skiprows=1, header=None)
        dates = data[0].iloc[self.args.seq_len: cut_off + self.args.seq_len + 1]
        dates = dates.to_numpy().reshape(-1, 1)


        if load:
            # path = os.path.join(self.args.checkpoints, setting)
            # print("LOAD")
            path = self.args.checkpoints
            best_model_path = './'+path+'/'+'checkpoint.pth'
            # print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        trues = []
        preds_inversed = []
        trues_inversed = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx) in enumerate(pred_loader):

            pred, true, peak_logits = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_peak, batch_y_peak, batch_x_idx, batch_y_idx)

            pred_inversed = pred_data.inverse_transform(pred)
            true_inversed = pred_data.inverse_transform(true)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            preds_inversed.append(pred_inversed.detach().cpu().numpy())
            trues_inversed.append(true_inversed.detach().cpu().numpy())

            if i == cut_off:
                break

        preds = np.array(preds)
        trues = np.array(trues)
        preds_inversed = np.array(preds_inversed).astype(int)
        trues_inversed = np.array(trues_inversed).astype(int)

        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        preds_inversed = preds_inversed.reshape(-1, preds_inversed.shape[-2], preds_inversed.shape[-1])
        trues_inversed = trues_inversed.reshape(-1, trues_inversed.shape[-2], trues_inversed.shape[-1])
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open("resINF.txt", "a") as f:
            f.write(f"{setting},{mse:.4f},{mae:.4f} \n")

        preds_2d = preds.reshape(preds.shape[-3], preds.shape[-2]).astype(object)
        trues_2d = trues.reshape(preds.shape[-3], preds.shape[-2]).astype(object)
        preds_inv_2d = preds_inversed.reshape(preds.shape[-3], preds.shape[-2]).astype(object)
        trues_inv_2d = trues_inversed.reshape(preds.shape[-3], preds.shape[-2]).astype(object)

        preds_2d = pd.DataFrame(preds_2d)
        trues_2d = pd.DataFrame(trues_2d)
        preds_inv_2d = pd.DataFrame(preds_inv_2d)
        trues_inv_2d = pd.DataFrame(trues_inv_2d)

        os.makedirs("res/"+setting, exist_ok=True)
        preds_2d.to_csv("res/"+setting+"/INF_pred.csv", index=False, header=False)
        trues_2d.to_csv("res/"+setting+"/INF_true.csv", index=False, header=False)
        preds_inv_2d.to_csv("res/"+setting+"/INF_pred_inv.csv", index=False, header=False)
        trues_inv_2d.to_csv("res/"+setting+"/INF_trues_inv.csv", index=False, header=False)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_peak=None, batch_y_peak=None, batch_x_idx=None, batch_y_idx=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        batch_x_peak = batch_x_peak.float().to(self.device)
        batch_y_peak = batch_y_peak.float().to(self.device)

        batch_x_idx = batch_x_idx.to(self.device)
        batch_y_idx = batch_y_idx.to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs, logits = self.model(batch_x,
                                     batch_x_mark,
                                     dec_inp,
                                     batch_y_mark,
                                     batch_x_peak=batch_x_peak,
                                     batch_y_peak=batch_y_peak,
                                     batch_x_idx=batch_x_idx,
                                     batch_y_idx=batch_y_idx)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            # print( dataset_object.inverse_transform(outputs) )
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y, logits
