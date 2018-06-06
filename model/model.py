import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import shutil
from pathlib import Path

class RNNPredictor(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, enc_inp_size, rnn_inp_size, rnn_hid_size, dec_out_size, nlayers, dropout=0.5,
                 tie_weights=False,res_connection=False):
        super(RNNPredictor, self).__init__()
        self.enc_input_size = enc_inp_size

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(enc_inp_size, rnn_inp_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(rnn_inp_size, rnn_hid_size, nlayers, dropout=dropout)
        elif rnn_type == 'SRU':
            from cuda_functional import SRU, SRUCell
            self.rnn = SRU(input_size=rnn_inp_size,hidden_size=rnn_hid_size,num_layers=nlayers,dropout=dropout,
                           use_tanh=False,use_selu=True,layer_norm=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'SRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(rnn_inp_size, rnn_hid_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(rnn_hid_size, dec_out_size)


        if tie_weights:
            if rnn_hid_size != rnn_inp_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.res_connection=res_connection
        self.init_weights()
        self.rnn_type = rnn_type
        self.rnn_hid_size = rnn_hid_size
        self.nlayers = nlayers
        #self.layerNorm1=nn.LayerNorm(normalized_shape=rnn_inp_size)
        #self.layerNorm2=nn.LayerNorm(normalized_shape=rnn_hid_size)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_hiddens=False, noise=False):
        emb = self.drop(self.encoder(input.contiguous().view(-1,self.enc_input_size))) # [(seq_len x batch_size) * feature_size]
        emb = emb.view(-1, input.size(1), self.rnn_hid_size) # [ seq_len * batch_size * feature_size]
        if noise:
            # emb_noise = Variable(torch.randn(emb.size()))
            # hidden_noise = Variable(torch.randn(hidden[0].size()))
            # if next(self.parameters()).is_cuda:
            #     emb_noise=emb_noise.cuda()
            #     hidden_noise=hidden_noise.cuda()
            # emb = emb+emb_noise
            hidden = (F.dropout(hidden[0],training=True,p=0.9),F.dropout(hidden[1],training=True,p=0.9))

        #emb = self.layerNorm1(emb)
        output, hidden = self.rnn(emb, hidden)
        #output = self.layerNorm2(output)

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # [(seq_len x batch_size) * feature_size]
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) # [ seq_len * batch_size * feature_size]
        if self.res_connection:
            decoded = decoded + input
        if return_hiddens:
            return decoded,hidden,output

        return decoded, hidden




    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.rnn_hid_size).zero_())

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def save_checkpoint(self,state, is_best):
        print("=> saving checkpoint ..")
        args = state['args']
        checkpoint_dir = Path('save',args.data,'checkpoint')
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint = checkpoint_dir.joinpath(args.filename).with_suffix('.pth')

        torch.save(state, checkpoint)
        if is_best:
            model_best_dir = Path('save',args.data,'model_best')
            model_best_dir.mkdir(parents=True,exist_ok=True)

            shutil.copyfile(checkpoint, model_best_dir.joinpath(args.filename).with_suffix('.pth'))

        print('=> checkpoint saved.')

    def extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer

    def initialize(self,args,feature_dim):
        self.__init__(rnn_type = args.model,
                           enc_inp_size=feature_dim,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=feature_dim,
                           nlayers = args.nlayers,
                           dropout = args.dropout,
                           tie_weights= args.tied,
                           res_connection=args.res_connection)
        self.to(args.device)

    def load_checkpoint(self, args, checkpoint, feature_dim):
        start_epoch = checkpoint['epoch'] +1
        best_val_loss = checkpoint['best_loss']
        args_ = checkpoint['args']
        args_.resume = args.resume
        args_.pretrained = args.pretrained
        args_.epochs = args.epochs
        args_.save_interval = args.save_interval
        args_.prediction_window_size=args.prediction_window_size
        self.initialize(args_, feature_dim=feature_dim)
        self.load_state_dict(checkpoint['state_dict'])

        return args_, start_epoch, best_val_loss