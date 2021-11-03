import torch
import torch.nn as nn
from utils import generate_original_PE
class HyberRnnNet(nn.Module):
    def __init__(self, 
    inputs_dim:int, 
    hidden_hat_dim:int, 
    layer_nums:int, 
    rnn_type:str,
    bidirectional =False):

        super(HyberRnnNet, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_hat_dim = hidden_hat_dim
        self.layer_nums = layer_nums
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        
        if self.rnn_type == 'rnn':
            self.RNN = nn.RNN(input_size = self.inputs_dim, 
            hidden_size = self.hidden_hat_dim,
            num_layers = self.layer_nums,
            batch_first = True,
            dropout = 0.2,
            bidirectional = bidirectional
                            )

        elif self.rnn_type == 'gru':
            self.RNN = nn.GRU(input_size = self.inputs_dim, 
            hidden_size = self.hidden_hat_dim,
            num_layers = self.layer_nums,
            batch_first = True,
            dropout = 0.2,
            bidirectional = bidirectional
                             )

        elif self.rnn_type == 'lstm':
            self.RNN = nn.LSTM(input_size = self.inputs_dim, 
            hidden_size = self.hidden_hat_dim,
            num_layers = self.layer_nums,
            batch_first = True,
            dropout = 0.2,
            bidirectional = bidirectional
                              )
            #torch.nn.init.orthogonal_()

        else:
            raise ValueError(
        "rnn_type should be rnn, gru or lstm!"
        )

    def forward(self,inputs, state_0): # inputs:(batch, seq_len, input_size) ; h_0:(num_layers * num_directions, batch, hidden_size)
        if self.rnn_type == 'lstm':
            (h_0, c_0) = state_0
            output, (h_t, c_t) = self.RNN(inputs, (h_0, c_0))
            return output, (h_t, c_t)
        
        else:
            output, h_t = self.RNN(inputs, state_0) #h_t: h^{hat}_{t}
            return output, h_t

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        a= 2 if self.bidirectional else 1
        if self.rnn_type == 'lstm':
            h_0 = torch.randn(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            c_0 = torch.randn(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            return (h_0, c_0)

        else:
            h_0 = torch.zeros(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            return h_0
        
        
class InferenceNetRnnCell(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int, 
    activate = 'tanh'
   ):
        super(InferenceNetRnnCell, self).__init__()
        self.w_hh = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hx = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hb = nn.Linear(hidden_hat_dim, z_dim, bias=True)
        self.W_hz = nn.Linear(z_dim, hidden_dim,bias=False)
        self.W_xz = nn.Linear(z_dim, hidden_dim, bias=False)
        self.b = nn.Linear(z_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wx = nn.Linear(input_dim ,hidden_dim)
        self.activate = activate
        self.dropout = nn.Dropout(p=0.1, inplace=True)
       
    def forward(self, h_t, h_t_hat, inf_inputs):

        z_h = self.w_hh(h_t_hat) #z_{h}
        z_x = self.w_hx(h_t_hat) # z_{x}
        z_bias = self.w_hb(h_t_hat) #z_{b}
        d_z_h = self.W_hz(z_h) #d_{h}(z_{h})
        d_z_x = self.W_xz(z_x) #d_{x}(z_{x})
        b_z_b = self.b(z_bias) #d_{b}{z_{b}}
        h_t_new = d_z_h*self.Wh(h_t)+d_z_x*self.Wx(inf_inputs)+ b_z_b
        h_t_new = self.dropout(h_t_new)
        
        if self.activate =='relu':
            return torch.relu(h_t_new)
        elif self.activate =='tanh':
            return torch.tanh(h_t_new)
        elif self.activate =='sigmoid':
            return torch.sigmoid(h_t_new)
    
class InferenceNetLSTMCell(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int
   ):

        super(InferenceNetLSTMCell, self).__init__()
        self.w_hh = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hx = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hb = nn.Linear(hidden_hat_dim, z_dim)
        self.W_hz = nn.Linear(z_dim, 4*hidden_dim,bias=False)
        self.W_xz = nn.Linear(z_dim, 4*hidden_dim, bias=False)
        self.b = nn.Linear(z_dim, 4*hidden_dim)
        self.Wh = nn.Linear(hidden_dim, 4*hidden_dim)
        self.Wx = nn.Linear(input_dim ,4*hidden_dim)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_c = nn.LayerNorm(hidden_dim)
    def forward(self, h_t, c, h_t_hat, inf_inputs):

        z_h = self.w_hh(h_t_hat) #z_{h}   size = (b, z_dim)
        z_x = self.w_hx(h_t_hat) #z_{x}  size = (b,z_dim)
        z_bias = self.w_hb(h_t_hat) # z_{b}  size = (b, z_dim)
        d_z_h = self.W_hz(z_h) #d_{h}(z_{h}) size = (b, 4*hidden_dim)
        d_z_x = self.W_xz(z_x) #d_{x}(z_{x})  size = (b, 4*hidden_dim)
        b_z_b = self.b(z_bias) #d_{b}{z_{b}}  size = (b, 4*hidden_dim)
        ifgo = d_z_h*self.Wh(h_t)+d_z_x*self.Wx(inf_inputs)+ b_z_b    #size = (b, 4*hidden_dim)
        i,f,g,o = torch.chunk(ifgo,4,-1)  #i ,f,g,o, size = (b, hidden_dim)
        i = torch.sigmoid(i) 
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c =f*c+i*g
        new_h = o*torch.tanh(new_c)
        new_h  = self.dropout(new_h)
        new_h = self.norm_h(new_h)
        new_c = self.norm_c(new_c)
        return new_h, new_c

class InferenceRNN(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int,
    cell_type: str
     ):
        super(InferenceRNN, self).__init__()
        self.cell_type = cell_type
        if cell_type =='rnn':
            self.cell = InferenceNetRnnCell(z_dim, input_dim, hidden_hat_dim, hidden_dim)
        elif cell_type =='lstm':
            self.cell = InferenceNetLSTMCell(z_dim, input_dim, hidden_hat_dim, hidden_dim)
        else:
            raise ValueError("need gru?")
            
    def forward(self, state, h_hat_t, inf_inputs):
        outputs = []
        if self.cell_type =='lstm':
            h,c = state
            for t in range(inf_inputs.size(0)):
                h,c = self.cell(h,c, h_hat_t, inf_inputs[t])
                outputs.append(h)
        else:
            h = state
            for t in range(inf_inputs.size(0)):
                h = self.cell(h, h_hat_t, inf_inputs[t])
                outputs.append(h)
        
        return torch.stack(outputs,1)

class AttnInferenceNetLSTMCell(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int
   ):

        super(InferenceNetLSTMCell, self).__init__()
        self.atten = nn.MultiheadAttention(hidden_hat_dim, 1)
        self.w_hh = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hx = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hb = nn.Linear(hidden_hat_dim, z_dim)
        self.W_hz = nn.Linear(z_dim, 4*hidden_dim,bias=False)
        self.W_xz = nn.Linear(z_dim, 4*hidden_dim, bias=False)
        self.b = nn.Linear(z_dim, 4*hidden_dim)
        self.Wh = nn.Linear(hidden_dim, 4*hidden_dim)
        self.Wx = nn.Linear(input_dim ,4*hidden_dim)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_c = nn.LayerNorm(hidden_dim)
    def forward(self, h_t, c, h_t_hat, inf_inputs):
        z_t,_ = self.atten(h_t.unsqueeze(1), h_t_hat,h_t_hat)
        z_h = self.w_hh(z_t.squeeze()) #z_{h}   size = (b, z_dim)
        z_x = self.w_hx(z_t.squeeze()) #z_{x}  size = (b,z_dim)
        z_bias = self.w_hb(z_t.squeeze()) # z_{b}  size = (b, z_dim)
        d_z_h = self.W_hz(z_h) #d_{h}(z_{h}) size = (b, 4*hidden_dim)
        d_z_x = self.W_xz(z_x) #d_{x}(z_{x})  size = (b, 4*hidden_dim)
        b_z_b = self.b(z_bias) #d_{b}{z_{b}}  size = (b, 4*hidden_dim)
        ifgo = d_z_h*self.Wh(h_t)+d_z_x*self.Wx(inf_inputs)+ b_z_b    #size = (b, 4*hidden_dim)
       
        i,f,g,o = torch.chunk(ifgo,4,-1)  #i ,f,g,o, size = (b, hidden_dim)
        i = torch.sigmoid(i) 
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c =f*c+i*g
        new_h = o*torch.tanh(new_c)
        new_h  = self.dropout(new_h)
        new_h = self.norm_h(new_h)
        new_c = self.norm_c(new_c)
        return new_h, new_c

class AttnInferenceRNN(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int,
    cell_type: str
     ):
        super(InferenceRNN, self).__init__()
        self.cell_type = cell_type
        if cell_type =='rnn':
            self.cell = InferenceNetRnnCell(z_dim, input_dim, hidden_hat_dim, hidden_dim)
        elif cell_type =='lstm':
            self.cell = AttnInferenceNetLSTMCell(z_dim, input_dim, hidden_hat_dim, hidden_dim)
        else:
            raise ValueError("need gru?")
            
    def forward(self, state, h_hat, inf_inputs):
        outputs = []
        if self.cell_type =='lstm':
            h,c = state
            for t in range(inf_inputs.size(0)):
                h,c = self.cell(h,c, h_hat, inf_inputs[t])
                outputs.append(h)
        else:
            h = state
            for t in range(inf_inputs.size(0)):
                h = self.cell(h, h_hat, inf_inputs[t])
                outputs.append(h)
        
        return torch.stack(outputs,1)


class TransformerTSF(nn.Module):
   
    def __init__(self,input_dim, model_dim, output_dim, head, N):
        super().__init__()
        self.input_emb =  nn.Linear(input_dim, model_dim)
        self.position_emb = generate_original_PE(512, model_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=model_dim, nhead=head), num_layers=N)
        self.project = nn.Linear(model_dim, output_dim)
        self.act = nn.Sigmoid()
    def forward(self,x,mask):
        inputs = self.input_emb(x[:,:512,]).add_(self.position_emb.to(x.device))
        outputs = self.transformer_encoder(inputs.transpose(0,1), mask= mask)
        
        logits = self.project(outputs)
        return self.act(logits.transpose(0,1).contiguous()), outputs.transpose(0,1).contiguous()