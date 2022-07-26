from modules import MainGRUCell, LatentLayer,AttentionLayer, MetaRnnNet,MainRNN

class HyperRNN(nn.Module):
    def __init__(self,
             meta_input_dim,
             meta_dim,
             main_input_dim, 
             main_dim,
             z_dim,
             cell_type,
             bidirectional = True,
             use_attention = True):
        super(HyperRNN, self).__init__()
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.meta_rnn = MetaRnnNet(meta_input_dim,
            meta_dim,
            1,
            rnn_type = cell_type,
            bidirectional = bidirectional
        )
        self.main_rnn = MainRNN(meta_dim,
                                main_input_dim,
                                main_dim,
                                z_dim,
                                cell_type,
                                use_attention
        )
        if self.bidirectional:
            self.bridge = nn.Linear(2*meta_dim, main_dim)
            self.meta_bridge = nn.Linear(2*meta_dim,meta_dim, )
        else:
            self.bridge = nn.Linear(meta_dim, main_dim)
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
