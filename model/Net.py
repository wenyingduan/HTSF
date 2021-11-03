import torch
import torch.nn as nn
from Modules import TransformerTSF, AttnInferenceRNN
class HyperNetTSF(nn.Module):
    def __init__(self, hyper_input_dim, h_hat_dim, infer_input_dim, infer_hidden_dim, z_dim, num_labels =3):
        super(HyperNetTSF,self).__init__()
        self.hyper_input_dim = hyper_input_dim
        self.h_hat_dim = h_hat_dim
        self.infer_input_dim = infer_input_dim
        self.infer_hidden_dim = infer_hidden_dim
      
        self.Hyper = TransformerTSF(hyper_input_dim,h_hat_dim,hyper_input_dim,4,3)
        self.infer_h_layer = nn.Linear(h_hat_dim,infer_hidden_dim)
        self.infer_c_layer = nn.Linear(h_hat_dim,infer_hidden_dim)
        self.Infer = AttnInferenceRNN(z_dim, infer_input_dim, h_hat_dim, infer_hidden_dim, "lstm")
        self.classifier =nn.Linear(infer_hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid(dim=-1)

    def forward(self, hyper_inputs,mask, infer_inputs):
        hyper_predict, hyper_outputs = self.Hyper(hyper_inputs,mask)
        h_hat_e = hyper_outputs.mean(1)
        h = self.infer_h_layer(h_hat_e)
        c = self.infer_c_layer(h_hat_e)
        state = (h,c)
        infer_inputs = infer_inputs.transpose(0,1).contiguous()
        infer_outputs = self.Infer(state, hyper_outputs, infer_inputs)
        logits = self.classifier(infer_outputs)
        
        return  hyper_predict, self.sigmoid(logits)
