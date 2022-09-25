#!/usr/bin/env python

"""
Applies the pretrained SimCLR model onto the consolidated waveform file to retrieve embeddings. 

Example: python apply_pretrained_simclr.py -i /deep/group/pulmonary-embolism/v2/waveforms/20sec-125hz-0norm-1wpp/Pleth -m /deep/group/pulmonary-embolism/v2/pretrained-models/simclr/pleth-epoch=9-step=2149.ckpt -f /deep/group/mimic3wdb-matched/files/20sec-125hz-1wpp/PLETH/waveforms.pt
"""

import argparse

"""
Based on https://code.engineering.queensu.ca/pritam/SSL-ECG/-/blob/master/implementation/signal_transformation_task.py
"""

import cv2
from sklearn.model_selection import train_test_split


def add_noise(signal, noise_amount):
    """ 
    adding noise
    """
    noise = np.random.normal(1, noise_amount, np.shape(signal)[0])
    noised_signal = signal+noise
    return noised_signal
    
def add_noise_with_SNR(signal, noise_amount):
    """ 
    adding noise
    created using: https://stackoverflow.com/a/53688043/10700812 
    """
    
    target_snr_db = noise_amount #20
    x_watts = signal ** 2                       # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)   # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))     # Generate an sample of white noise
    noised_signal = signal + noise_volts        # noise added signal

    return noised_signal 

def scaled(signal, factor):
    """"
    scale the signal
    """
    scaled_signal = signal * factor
    return scaled_signal
 

def negate(signal):
    """ 
    negate the signal 
    """
    negated_signal = signal * (-1)
    return negated_signal

    
def hor_flip(signal):
    """ 
    flipped horizontally 
    """
    hor_flipped = np.flip(signal).copy()
    return hor_flipped


def permute(signal, pieces):
    """ 
    signal: numpy array (batch x window)
    pieces: number of segments along time    
    """
    pieces       = int(np.ceil(np.shape(signal)[0]/(np.shape(signal)[0]//pieces)).tolist())
    piece_length = int(np.shape(signal)[0]//pieces)
    
    sequence = list(range(0,pieces))
    np.random.shuffle(sequence)
    
    permuted_signal = np.reshape(signal[:(np.shape(signal)[0]//pieces*pieces)], (pieces, piece_length)).tolist() + [signal[(np.shape(signal)[0]//pieces*pieces):]]
    permuted_signal = np.asarray(permuted_signal)[sequence]
    permuted_signal = np.hstack(permuted_signal)
        
    return permuted_signal

     
    
def time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
    """ 
    signal: numpy array (batch x window)
    sampling freq
    pieces: number of segments along time
    stretch factor
    squeeze factor
    """
    
    total_time = np.shape(signal)[0]//sampling_freq
    segment_time = total_time/pieces
    sequence = list(range(0,pieces))
    stretch = np.random.choice(sequence, math.ceil(len(sequence)/2), replace = False)
    squeeze = list(set(sequence).difference(set(stretch)))
    initialize = True
    for i in sequence:
        orig_signal = signal[int(i*np.floor(segment_time*sampling_freq)):int((i+1)*np.floor(segment_time*sampling_freq))]
        orig_signal = orig_signal.reshape(np.shape(orig_signal)[0],1)
        if i in stretch:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*stretch_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
        elif i in squeeze:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*squeeze_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
    return time_warped
   

from torch.utils.data import Dataset
import math
import random

class WaveformDataLoader(Dataset):
    def __init__(self, waveforms, is_train=True, verbose=False):
        self.waveforms = waveforms

    def __len__(self):
        return len(self.waveforms)
        # return 50

    def __getitem__(self, idx):
        waveform1 = self.waveforms[idx, :]
        waveform2 = waveform1.copy()
        if random.randint(0, 1) == 1:
            waveform2 = add_noise(waveform2, noise_amount=0.2)
        if random.randint(0, 1) == 1:
            waveform2 = negate(waveform2)
        if random.randint(0, 1) == 1:
            waveform2 = hor_flip(waveform2)
        # if random.randint(0, 1) == 1:
        #     waveform2 = time_warp(waveform2, 250, 10, 2, 1)
        return np.expand_dims(waveform1, axis=0), np.expand_dims(waveform2, axis=0)

"""
Adapted from https://github.com/hsd1503/resnet1d/blob/master/resnet1d.py
"""

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)
        
        return out

class SimclrModel(nn.Module):
    def __init__(self, embed_dim=128):
        super(SimclrModel, self).__init__()

        kernel_size = 16
        stride = 2
        n_block = 48
        downsample_gap = 6
        increasefilter_gap = 12
        self.model = ResNet1D(
            in_channels=1, 
            base_filters=64, # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=kernel_size, 
            stride=stride, 
            groups=32, 
            n_block=n_block, 
            n_classes=4, 
            downsample_gap=downsample_gap, 
            increasefilter_gap=increasefilter_gap, 
            use_do=True)

        # encoder
        self.model.dense = nn.Identity()

        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, embed_dim, bias=True))

    def forward(self, x):
        x = self.model(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(out, dim=-1)


import pytorch_lightning as pl
import csv
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SimclrModule(pl.LightningModule):

    def __init__(self, 
                 input_file,
                 train_only=False,
                 feature_dim=128, 
                 temperature=0.5,
                 batch_size=64,
                 lr=1e-3,
                 weight_decay=1e-6
    ):

        super().__init__()

        self.input_file = input_file
        self.train_only = train_only
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.training_losses = []
        self.val_losses = []
        self.test_losses = []
        self.cumulative_training_losses = []
        self.cumulative_val_losses = []
        self.batch_size = batch_size
        self.epoch = 0
        
        # Define PyTorch model
        self._setup()
        self.model = SimclrModel(embed_dim=feature_dim)

    def forward(self, x1, x2):
        print(f"x1={x1.shape}")
        out_1 = self.model(x1)
        out_2 = self.model(x2)
        return out_1, out_2

    def training_step(self, batch, batch_idx):
        w1, w2 = batch
        w1 = w1.to(device=device, dtype=torch.float)
        w2 = w2.to(device=device, dtype=torch.float)

        out_1, out_2 = self(w1, w2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        if batch_idx == 0:
            self.training_losses = [loss.cpu().detach().numpy()]
        else:
            self.training_losses.append(loss.cpu().detach().numpy())

        self.log_dict({"train_loss": loss})
        # wandb.log({"train_loss": loss})
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = np.array([x["loss"].item() for x in outputs]).mean()
        self.cumulative_training_losses.append(avg_loss)

    def validation_step(self, batch, batch_idx):
        w1, w2 = batch
        w1 = w1.to(device=device, dtype=torch.float)
        w2 = w2.to(device=device, dtype=torch.float)

        out_1, out_2 = self(w1, w2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        if batch_idx == 0:
            self.val_losses = [loss.cpu().detach().numpy()]
        else:
            self.val_losses.append(loss.cpu().detach().numpy())

        self.log_dict({"val_loss": loss})
        # wandb.log({"val_loss": loss})
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = np.array([x.item() for x in outputs]).mean()
        self.cumulative_val_losses.append(avg_loss)

    def test_step(self, batch, batch_idx):
        w1, w2 = batch
        w1 = w1.to(device=device, dtype=torch.float)
        w2 = w2.to(device=device, dtype=torch.float)

        out_1, out_2 = self(w1, w2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
        if batch_idx == 0:
            self.test_losses = [loss.cpu().detach().numpy()]
        else:
            self.test_losses.append(loss.cpu().detach().numpy())

        self.log_dict({"test_loss": loss})
        # wandb.log({"test_loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_losses(self):
        return self.cumulative_training_losses, self.cumulative_val_losses

    def get_model(self):
        return self.model

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # Nothing needs to happen here
        pass

    def setup(self, stage=None):
        self._setup()

    def _setup(self):
        waveforms = torch.load(self.input_file)
        waveforms = waveforms.cpu().detach().numpy()
        print(f"Loaded waveforms of shape: {waveforms.shape}")

        if self.train_only:
            print(f"Training only")
            self.waveforms_train = waveforms
        else:
            self.waveforms_train, waveforms_test = train_test_split(waveforms, train_size=0.7, random_state=42)
            self.waveforms_val, self.waveforms_test = train_test_split(waveforms_test, train_size=0.5, random_state=42)
            print(f"Loaded waveforms_train of shape: {self.waveforms_train.shape}")
            print(f"Loaded waveforms_val of shape: {self.waveforms_val.shape}")
            print(f"Loaded waveforms_test of shape: {self.waveforms_test.shape}")

    def train_dataloader(self):
        dl = WaveformDataLoader(self.waveforms_train)
        return DataLoader(dl, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=4)

    def val_dataloader(self):
        if self.train_only:
            return None
        else:
            dl = WaveformDataLoader(self.waveforms_val, is_train=False)
            return DataLoader(dl, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=4)

    def test_dataloader(self):
        if self.train_only:
            return None
        else:
            dl = WaveformDataLoader(self.waveforms_test, is_train=False)
            return DataLoader(dl, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=4)


def process_record(input_args):
    i, total_rows, waveform_df, patient_id, waveforms, model = input_args
    print(f"[{i}/{total_rows}] {patient_id} waveform...")
    
    try:
        waveforms_for_patient = []
        for j, row in waveform_df[waveform_df["record_name"] == patient_id].iterrows():
            waveforms_for_patient.append([waveforms[j]])

        input_tensor = torch.tensor(np.array(waveforms_for_patient), dtype=torch.float32)
#         print(f"input={input_tensor.shape}")
        embeddings = model(input_tensor).detach().numpy()
#         print(f"output1={embeddings.shape}")
        embeddings = np.mean(embeddings, axis=0)
#         print(f"output2={embeddings.shape}")

        return {
            "record_name": patient_id,
            "subject_id": waveform_df[waveform_df["record_name"] == patient_id]["subject_id"] if "subject_id" in waveform_df.columns else "",
            "embeddings": embeddings
        }
    except Exception as e:
        print("Unexpected error:", e)
        return None
    

def run(args):
    input_folder = args.waveform_folder
    model_path = args.model_path
    original_input_file = args.original_input_file
    max_patients = int(args.max_patients) if args.max_patients is not None else None
    
    output_folder = f"{input_folder}/simclr"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Loading file from {input_folder}")

    df = pd.read_csv(f"{input_folder}/summary.csv")
    waveforms_numpy = np.load(f"{input_folder}/waveforms.dat.npy")
    patient_ids = set(df["record_name"].tolist())
    
    print(f"Loading model from {model_path}")
    simclr_module = SimclrModule.load_from_checkpoint(model_path, input_file=original_input_file, feature_dim=322)
    simclr_module.eval()
    model = simclr_module.get_model()
    print(f"Loaded model from {model_path}")
    
    print(f"Found {input_folder} with shape {df.shape}")
    total_rows = len(df)

    waveforms = []
    for i, patient_id in tqdm(enumerate(patient_ids), disable=True):
        input_args = [i, total_rows, df, patient_id, waveforms_numpy, model]
        result = process_record(input_args)
        if result is not None:
            waveforms.append(result)
        if max_patients is not None and i >= (max_patients - 1):
            break

    output_embeddings = []
    with open(f"{output_folder}/embeddings_summary.csv", "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        headers = []
        if "subject_id" in df.columns:
            headers.append("subject_id")
        if "record_name" in df.columns:
            headers.append("record_name")
        writer.writerow(headers)
        for row in waveforms:
            new_row = []
            if "subject_id" in df.columns:
                new_row.append(row["subject_id"])
            if "record_name" in df.columns:
                new_row.append(row["record_name"])
                
            writer.writerow(new_row)
            output_embeddings.append(row["embeddings"])

    output_tensor = np.array(output_embeddings)
    np.save(f"{output_folder}/embeddings.dat", output_tensor)
    print(f"Output is written to: {output_folder}/summary.csv")
    print(f"Output is written to: {output_folder}/embeddings.dat.npy")

#
# Main
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts waveforms from the patient dir to create a single file')
    parser.add_argument('-i', '--waveform-folder',
                        required=True,
                        help='The path to the consolidated waveforms folder containing the summary file and the waveform NumPy')
    parser.add_argument('-f', '--original-input-file',
                        required=True,
                        help='The original input file used to train the model')
    parser.add_argument('-m', '--model-path',
                        default=15,
                        help='Pre-trained model location')
    parser.add_argument('-p', '--max-patients',
                        default=None,
                        help='Maximum number of patients to use')

    args = parser.parse_args()

    run(args)

    print("DONE")
