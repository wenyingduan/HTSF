import torch
import torch.nn as nn 
from onmt.modules import GlobalAttention
import math




def HadamardLinear(inputs,W,b=None):
    
    return inputs*W+b if b is not None else inputs*W


        
class AttentionLayer(nn.Module):
    def __init__(self,q_dim, k_dim):
        super(AttentionLayer, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        if q_dim == k_dim:
            self.attention = GlobalAttention(q_dim, attn_type = 'mlp')
        else:
            self.linear_context = nn.Linear(k_dim, q_dim, bias=False)
            self.linear_query = nn.Linear(q_dim, q_dim, bias=False)
            self.attention = GlobalAttention(q_dim, attn_type = 'mlp')
        
    def forward(self,q ,s):
        if self.q_dim == self.k_dim:
            attn_h, align_vectors = self.attention(q, s)
        else:
            s = self.linear_context(s)
            q = self.linear_query(q)
            attn_h, align_vectors = self.attention(q, s)
        return attn_h, align_vectors
    
class LatentLayer(nn.Module):
    def __init__(self, meta_dim, main_dim, z_dim, use_attention =True):
        super(LatentLayer, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.Attn = AttentionLayer(main_dim,meta_dim)
            self.W_z = nn.Linear(main_dim, z_dim)
        else:
            self.W_z = nn.Linear(meta_dim, z_dim)
    def forward(self, meta_outputs, main_output_t= None):
        if self.use_attention:
            attn_h, align_vectors = self.Attn(main_output_t, meta_outputs)
            z = self.W_z(attn_h)
        
        else:
            z = self.W_z(meta_outputs[:,-1,:])
        return z

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.weight_ih = nn.Parameter(torch.randn(4*hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.randn(4*hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(4*hidden_dim))

    def forward(self,inputs, h, c):
        ifgo = F.linear(inputs, self.weight_ih, self.bias)  + F.linear(h, self.weight_hh)
        i, f, g, o =torch.chunk(ifgo, 4,-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f*c +i*g
        new_h = o*torch.tanh(new_c)
        return (new_h, new_c)
    
class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, inputs, h):
        h = F.linear(inputs,self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        return h
    
class GRUCell(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(GRUCell, self).__init__()
            self.weight_ih = nn.Parameter(torch.randn(3*hidden_dim, input_dim))
            self.weight_hh = nn.Parameter(torch.randn(3*hidden_dim, hidden_dim))
            self.bias = nn.Parameter(torch.zeros(3*hidden_dim))

        def forward(self, inputs ,h):
            rzn_i = F.linear(inputs, self.weight_ih, self.bias) 
            rzn_h = F.linear(h, self.weight_hh)
            r_i, z_i, n_i = torch.chunk(rzn_i, 3, -1)
            r_h, z_h, n_h = torch.chunk(rzn_h, 3, -1)
            r = torch.sigmoid(r_i+r_h)
            z = torch.sigmoid(z_i+z_h)
            n = torch.tanh(r*n_h+n_i)
            new_h = (1-z)*h+z*n
            return new_h

        
class MetaRnnNet(nn.Module):
    def __init__(self, 
    inputs_dim:int, 
    hidden_hat_dim:int, 
    layer_nums:int, 
    rnn_type:str,
    bidirectional =False):

        super(MetaRnnNet, self).__init__()
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

        else:
            raise ValueError(
        "rnn_type should be rnn, gru or lstm!"
        )
        

    def forward(self,inputs): # inputs:(batch, seq_len, input_size) ; h_0:(num_layers * num_directions, batch, hidden_size)
        state_0 = self.init_state(inputs)
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
            h_0 = torch.zeros(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            c_0 = torch.randn(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            return (h_0, c_0)

        else:
            h_0 = torch.zeros(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            return h_0
        
class MainGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(MainGRUCell, self).__init__()
        self.weight_ih = nn.Linear(z_dim, 3*hidden_dim, bias=False)
        self.weight_hh = nn.Linear(z_dim, 3*hidden_dim, bias=False)
        self.bias = nn.Linear(z_dim, 3*hidden_dim, bias=False)
        self.W_ih = nn.Linear(input_dim, 3*hidden_dim)
        self.W_hh = nn.Linear(hidden_dim, 3*hidden_dim)
        
    def forward(self, z,inputs ,h):
        weight_ih = self.weight_ih(z)
        weight_hh = self.weight_hh(z)
        bias = self.bias(z)
        inputs = self.W_ih(inputs)
        h_ = self.W_hh(h)
    
        rzn_i = HadamardLinear(inputs, weight_ih, bias) 
        rzn_h = HadamardLinear(h_, weight_hh)
        r_i, z_i, n_i = torch.chunk(rzn_i, 3, -1)
        r_h, z_h, n_h = torch.chunk(rzn_h, 3, -1)
        r = torch.sigmoid(r_i+r_h)
        z = torch.sigmoid(z_i+z_h)
           
        n = torch.tanh(r*n_h+n_i)
        new_h = (1-z)*h+z*n
        
        return new_h
    
class MainRNN(nn.Module):
    def __init__(self,
             meta_dim,
             main_input_dim, 
             main_dim,
             z_dim,
             cell_type,
             use_attention = True):
        super(MainRNN, self).__init__()
        self.cell_type = cell_type
        self.calculate_z = LatentLayer(meta_dim, main_dim, z_dim, use_attention)
        if cell_type =='rnn':
               pass
        elif cell_type =='gru':
            self.cell = MainGRUCell(main_input_dim, main_dim, z_dim)
    def forward(self, meta_outputs,state, main_inputs):
        outputs =[]
        z_s = []
        h = state
        main_inputs = main_inputs.transpose(0,1)
        for t in range(main_inputs.size(0)):
            z= self.calculate_z(meta_outputs,h)
            h = self.cell(z,main_inputs[t],h)
            outputs.append(h)
            z_s.append(z)
        return outputs, z_s
