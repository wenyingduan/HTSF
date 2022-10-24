import torch
import torch.utils.data as torch_data
import pandas as pd
import numpy as np
import random
import datetime
import time


def create_dataset(df, station, start_date, end_date, mean=None, std=None):
    data=df[station]
    feat, label, label_reg =data[0], data[1], data[2]
    referece_start_time=datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time=datetime.datetime(2017, 2, 28, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start=(pd.to_datetime(start_date) - referece_start_time).days
    index_end=(pd.to_datetime(end_date) - referece_start_time).days
    feat=feat[index_start: index_end + 1]
    label=label[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]

    #ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    #feat=feat.reshape(-1, feat.shape[2])
    #feat=(feat - mean) / std
    #feat=feat.reshape(-1, ori_shape_1, ori_shape_2)

    return feat, label, label_reg

def np2tensor(raw):
    if isinstance(raw,tuple) or isinstance(raw,list):
        raw = [torch.from_numpy(i) for i in raw]
    else:
        raw = torch.from_numpy(raw)
    return raw


class Weather(torch_data.Dataset):
    def __init__(self,data_file, station, meta_len, trainable = 'train'):
        self.trainable = trainable
        self.meta_len = meta_len
        self.time_interval = 1
        self.main_len = 1 # one day, 24 hours
        if self.trainable =='train':
            self.raw = np2tensor(create_dataset(pd.read_pickle(data_file),station, '2013-3-1 0:0','2016-6-30 23:0'))
            self.trainset = self.raw[0]
            self.label_s = self.raw[1]
            self.label_t = self.raw[2]
        elif self.trainable =='eval':
            self.raw = np2tensor(create_dataset(pd.read_pickle(data_file),station, '2013-3-1 0:0','2016-10-31 23:0'))  # we can use all history data for making hyper inputs
            self.test_raw = np2tensor(create_dataset(pd.read_pickle(data_file),station,'2016-7-1 0:0','2016-10-31 23:0'))
        # we need historic data for parameter generation
            
        else: 
            self.raw = np2tensor(create_dataset(pd.read_pickle(data_file),station, '2013-3-1 0:0','2017-2-28 23:0'))
            self.test_raw = np2tensor(create_dataset(pd.read_pickle(data_file),station, '2016-11-2 0:0','2017-2-28 23:0'))
    
        #self.real_length_long_term = int(long_term_seq_day*(24/time_interval))
        self.hyper_seq_idx = []
        if self.trainable == 'train':
            self.start_idx =list(range(self.trainset.size(0)-self.meta_len-self.main_len))
           
            
            self.start_idx = torch.Tensor(self.start_idx)
          
            self.end_idx = self.start_idx+self.meta_len
        else:
            total_len = self.raw[0].size(0)
            test_len = self.test_raw[0].size(0)
            self.extend_len = total_len - test_len
            self.main_inputs = self.test_raw[0]
            self.label_s = self.test_raw[1]
            self.label_t = self.test_raw[2]
            
    def  __len__(self):
        if self.trainable == 'train':
            return len(self.start_idx)
        else:
            return self.test_raw[0].size(0)
       

    def __getitem__(self, index):
       
        
        if self.trainable=='train':
            meta_start = self.start_idx.long()[index]
            meta_end = self.end_idx.long()[index]
            meta_inputs =self.trainset[meta_start:meta_end]
            max_range = self.trainset.size(0)
            main_inputs_idx = random.randint(meta_end,max_range)
            main_index = random.randint(meta_end ,max_range-1)
            main_inputs=self.trainset[main_index]
            label_t = self.label_t[main_index]
         
        else:
            
            main_inputs= self.main_inputs[index]
            label_t= self.label_t[index]
            meta_end= random.randint(self.meta_len,index+self.extend_len-1)
            meta_start =  meta_end-self.meta_len
            meta_inputs =self.raw[0][meta_start:meta_end]
        
        
        return (meta_inputs.mean(1),  main_inputs, label_t)


def get_loaders(args):
    trainsets = Weather(args.file_dir,args.station, args.meta_length,'train')
    valsets =  Weather(args.file_dir,args.station, args.meta_length,'val')
    testsets = Weather(args.file_dir,args.station, args.meta_length,'test')
    trainloader = torch_data.DataLoader(trainsets,batch_size=args.batch_size,shuffle = True)
    valloader = torch_data.DataLoader(valsets,batch_size=args.batch_size) 
    testloader = torch_data.DataLoader(testsets,batch_size=36) 
    return trainloader, valloader, testloader