import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import torch.nn as nn 
import argparse
import configparser
from transformers import AdamW, get_linear_schedule_with_warmup
from model.htsf import HyperRNN as Model
from model.BasicTrainer import Trainer
from lib.dataloader import get_loaders

DEVICE = 'cuda:0'
DEBUG = 'True'
config_file = 'Air.conf'
config = configparser.ConfigParser()
config.read(config_file)

args = argparse.ArgumentParser(description='arguments')
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
#data
args.add_argument('--file_dir', default=config['data']['file_dir'],type=str)
args.add_argument('--meta_length', default=config['data']['meta_length'], type=int, help='the length of meta input sequences')
args.add_argument('--station', default=config['data']['station'],type=str)
#model
args.add_argument('--meta_input_dim',default=config['model']['meta_input_dim'],type=int)
args.add_argument('--meta_dim',default=config['model']['meta_dim'],type=int)
args.add_argument('--main_input_dim',default=config['model']['main_input_dim'],type=int)
args.add_argument('--main_dim',default=config['model']['main_dim'],type=int)
args.add_argument('--z_dim',default=config['model']['z_dim'],type=int)
args.add_argument('--cell',default=config['model']['cell'],type=str)
#train
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--val_times', default=config['train']['val_times'], type=int)
#test
args.add_argument('--test_times', default=config['test']['test_times'], type=int)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
#args.add_argument('--plot', default=config['log']['plot'], type=eval)

args = args.parse_args()
model = Model(args)
model = model.to(args.device)

train_loader, val_loader, test_loader = get_loaders(args)

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), args.lr_init)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = len(train_loader)*15, num_training_steps = len(train_loader)*300)
trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, test_loader, scheduler, args)
trainer.train()
