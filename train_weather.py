import torch
import torch.nn as nn 
import torch.optim as optim
import random

from torch.utils.data.dataloader import DataLoader
from dataset.data_hyper import hyper_inputs_collate, hyper_dataset
from dataset.data_process import load_weather_data

from model import HyperNetTSF

def build_model(args):
    model = HyperNetTSF(args.hyper_input_dim, args.h_hat_dim, args.infer_input_dim, args.infer_hidden_dim, args.z_dim, args.num_labels)
    return model

def build_dataloader(args):
    train_loader_list,valid_loader,test_loader = load_weather_data(args.file_path, args.batch_size, args.station)
    return train_loader_list,valid_loader,test_loader

def build_hyper_raw(args):
    hyper_raw = hyper_inputs_collate(args.file_path, args.station, args.hyper_input_len)
    return hyper_raw
   

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, optimizer, train_loader, hyper_raw):
    model.train()
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    hyper_data = random.sample(list(hyper_raw),817) # 817 is the total number of trainset.
    hyper_datasets = hyper_data(hyper_data)
    hyper_data_loader = DataLoader(hyper_data, 32, shuffle =True)
    for idx, instance in enumerate(zip(train_loader,hyper_data_loader)):
        infer_inputs = instance[0][0].cuda()
        label_reg_t = instance[0][1].cuda()
        label_reg_s = instance[0][2].cuda()
        hyper_inputs= instance[1][0].cuda()
        hyper_labels = instance[1][1].cuda()


    pass