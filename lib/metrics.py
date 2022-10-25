import torch
import numpy as np

def MAE_torch(pred, true):
    return torch.mean(torch.abs(true-pred))
    
def RMSE_torch(pred, true):
    return torch.sqrt(torch.mean((pred-true)**2))