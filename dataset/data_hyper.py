import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
class hyper_inputs_collate(object):
    def __init__(self,file_path, station,seq_len):
        self.raw = pd.read_pickle(file_path)[station][0][:-118]# delete test seq
        self.seq_len = seq_len
        
        
    def __call__(self): # raw:(seq_len,24,6)
        start_list = torch.arange(0,len(self.raw)-self.seq_len)
        inputs_list =[]
        for start in start_list:
    
            end = start+self.seq_len
            df = torch.from_numpy(self.raw[start:end])
            df = df.mean(-2)
            inputs_list.append(df.unsqueeze(0))
        hyper_inputs = torch.cat(inputs_list,0)
        return hyper_inputs.float()


class hyper_dataset(Dataset):
    def __init__(self, raw):
        self.dataset = raw
    def __getitem__(self, index):
        inputs = self.dataset[index][0:512]
        labels = self.dataset[index][1:]
        return inputs, labels
    def __len__(self):
        return len(self.dataset)
