import torch
import torch.nn as nn
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import MAE_torch, RMSE_torch

class Trainer(object):
    def __init__(self, 
    model, 
    criterion, 
    optimizer, 
    train_loader, 
    val_loader, 
    test_loader,
    scheduler,
    args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.args = args
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.logger = get_logger(args.log_dir, name = args.station, debug = args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
    
    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for idx, x in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            meta_inputs, main_inputs, labels = x
            pred ,z= self.model(meta_inputs.float().to(self.args.device), main_inputs.float().to(self.args.device))
            loss = self.criterion(pred.squeeze(),labels.float().to(self.args.device))
            total_loss = total_loss+loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if idx % self.args.log_step == 0:
                 self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, idx, len(self.train_loader), loss.item()))
        train_epoch_loss = total_loss/len(self.train_loader)
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        return train_epoch_loss
    
    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for idx, x in enumerate(self.val_loader):
                self.optimizer.zero_grad()
                meta_inputs, main_inputs, labels = x
                pred , _= self.model(meta_inputs.float().to(self.args.device), main_inputs.float().to(self.args.device))
                loss = self.criterion(pred.squeeze(),labels.float().to(self.args.device))
                total_val_loss = total_val_loss+loss.item()
        val_loss = total_val_loss / len(self.val_loader)
       
        return val_loss
    def validation(self, epoch):
        total_val_loss = 0
        for _ in range(self.args.val_times):
            val_loss = self.val_epoch(epoch)
            total_val_loss = total_val_loss + val_loss
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, total_val_loss/self.args.val_times))
        return total_val_loss/self.args.val_times
        
    def test(self):
        rmse =0
        mae = 0
        criterion_1 = nn.L1Loss()
        self.model.eval()
        for idx, x in enumerate(self.test_loader):
            with torch.no_grad():
                meta_inputs, main_inputs, labels = x
                pred,z = self.model(meta_inputs.float().to(self.args.device), main_inputs.float().to(self.args.device))
                #loss = self.criterion(pred.squeeze(),labels.float().cuda())
                #loss_1 = criterion_1(pred.squeeze(),labels.float().cuda())
                #loss_r = torch.sqrt(loss)
                loss_1 = MAE_torch(pred.squeeze(),labels.float().to(self.args.device))
                loss_r = RMSE_torch(pred.squeeze(),labels.float().to(self.args.device))
                rmse = rmse+loss_r.item()
                mae = mae+loss_1.item()
        return rmse/len(self.test_loader), mae/len(self.test_loader)
       
    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list =[]
        val_loss_list = []
        start_time = time.time()
        for epoch in range(self.args.epochs):
            train_epoch_loss = self.train_epoch(epoch)
            val_epoch_loss = self.validation(epoch)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
            
        self.model.load_state_dict(best_model)
        RMSE= 0
        MAE = 0
        for _ in range(self.args.test_times):
          rmse, mae = self.test()
          RMSE = RMSE+rmse
          MAE = MAE+mae
          
        self.logger.info("Sation: {}, RMSE：{:.6f}, MAE：{:.6f}".format(self.args.station, RMSE/self.args.test_times, MAE/self.args.test_times))




