import torch
import torch.nn as nn
from model.modules import MainGRUCell, LatentLayer,AttentionLayer, MetaRnnNet,MainRNN

class HyperRNN(nn.Module):
    def __init__(self,
             args,
             bidirectional = True,
             use_attention = True):
        super(HyperRNN, self).__init__()
        self.cell_type = args.cell
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.meta_rnn = MetaRnnNet(args.meta_input_dim,
            args.meta_dim,
            1,
            rnn_type = args.cell,
            bidirectional = bidirectional
        )
        self.main_rnn = MainRNN(args.meta_dim,
                                args.main_input_dim,
                                args.main_dim,
                                args.z_dim,
                                args.cell,
                                use_attention
        )
        if self.bidirectional:
            self.bridge = nn.Linear(2*args.meta_dim, args.main_dim)
            self.meta_bridge = nn.Linear(2*args.meta_dim,args.meta_dim, )
        else:
            self.bridge = nn.Linear(args.meta_dim, args.main_dim)
        self.output_layer =nn.Linear(64,1)
    def forward(self, meta_inputs, main_inputs):
        meta_outputs, meta_h = self.meta_rnn(meta_inputs)
        if self.bidirectional:
            meta_h = torch.cat([meta_h[0], meta_h[1]],-1)
            meta_outputs= self.meta_bridge(meta_outputs)
        h = self.bridge(meta_h).squeeze()
    
        main_outputs,zs = self.main_rnn(meta_outputs, h, main_inputs)
        main_outputs= torch.stack(main_outputs)
        main_outputs= main_outputs.transpose(0,1)
        outputs= self.output_layer(main_outputs[:,-1,:])
        return outputs,zs