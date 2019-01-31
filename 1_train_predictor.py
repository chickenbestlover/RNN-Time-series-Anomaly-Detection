import argparse
import time
import torch
import torch.nn as nn
import preprocess_data
from model import model
from torch import optim
from matplotlib import pyplot as plt
from pathlib import Path
from anomalyDetector import fit_norm_distribution_param

parser = argparse.ArgumentParser(description='PyTorch RNN Prediction Model on Time-series Dataset')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                    help='filename of the dataset')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--augment', type=bool, default=True,
                    help='augment')
parser.add_argument('--emsize', type=int, default=32,
                    help='size of rnn input features')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--res_connection', action='store_true',
                    help='residual connection')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=10,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval_batch size')
parser.add_argument('--bptt', type=int, default=50,
                    help='sequence length')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.7,
                    help='teacher forcing ratio (deprecated)')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights (deprecated)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                    help='save interval')
parser.add_argument('--save_fig', action='store_true',
                    help='save figure')
parser.add_argument('--resume','-r',
                    help='use checkpoint model parameters as initial parameters (default: False)',
                    action="store_true")
parser.add_argument('--pretrained','-p',
                    help='use checkpoint model parameters and do not train anymore (default: False)',
                    action="store_true")
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename,
                                                augment_test_data=args.augment)
train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData, args.batch_size)
test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, args.eval_batch_size)
gen_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, 1)


###############################################################################
# Build the model
###############################################################################
feature_dim = TimeseriesData.trainData.size(1)
model = model.RNNPredictor(rnn_type = args.model,
                           enc_inp_size=feature_dim,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=feature_dim,
                           nlayers = args.nlayers,
                           dropout = args.dropout,
                           tie_weights= args.tied,
                           res_connection=args.res_connection).to(args.device)
optimizer = optim.Adam(model.parameters(), lr= args.lr,weight_decay=args.weight_decay)
criterion = nn.MSELoss()
###############################################################################
# Training code
###############################################################################
def get_batch(args,source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len] # [ seq_len * batch_size * feature_size ]
    target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
    return data, target

def generate_output(args,epoch, model, gen_dataset, disp_uncertainty=True,startPoint=500, endPoint=3500):
    if args.save_fig:
        # Turn on evaluation mode which disables dropout.
        model.eval()
        hidden = model.init_hidden(1)
        outSeq = []
        upperlim95 = []
        lowerlim95 = []
        with torch.no_grad():
            for i in range(endPoint):
                if i>=startPoint:
                    # if disp_uncertainty and epoch > 40:
                    #     outs = []
                    #     model.train()
                    #     for i in range(20):
                    #         out_, hidden_ = model.forward(out+0.01*Variable(torch.randn(out.size())).cuda(),hidden,noise=True)
                    #         outs.append(out_)
                    #     model.eval()
                    #     outs = torch.cat(outs,dim=0)
                    #     out_mean = torch.mean(outs,dim=0) # [bsz * feature_dim]
                    #     out_std = torch.std(outs,dim=0) # [bsz * feature_dim]
                    #     upperlim95.append(out_mean + 2.58*out_std/np.sqrt(20))
                    #     lowerlim95.append(out_mean - 2.58*out_std/np.sqrt(20))

                    out, hidden = model.forward(out, hidden)

                    #print(out_mean,out)

                else:
                    out, hidden = model.forward(gen_dataset[i].unsqueeze(0), hidden)
                outSeq.append(out.data.cpu()[0][0].unsqueeze(0))


        outSeq = torch.cat(outSeq,dim=0) # [seqLength * feature_dim]

        target= preprocess_data.reconstruct(gen_dataset.cpu(), TimeseriesData.mean, TimeseriesData.std)
        outSeq = preprocess_data.reconstruct(outSeq, TimeseriesData.mean, TimeseriesData.std)
        # if epoch>40:
        #     upperlim95 = torch.cat(upperlim95, dim=0)
        #     lowerlim95 = torch.cat(lowerlim95, dim=0)
        #     upperlim95 = preprocess_data.reconstruct(upperlim95.data.cpu().numpy(),TimeseriesData.mean,TimeseriesData.std)
        #     lowerlim95 = preprocess_data.reconstruct(lowerlim95.data.cpu().numpy(),TimeseriesData.mean,TimeseriesData.std)

        plt.figure(figsize=(15,5))
        for i in range(target.size(-1)):
            plt.plot(target[:,:,i].numpy(), label='Target'+str(i),
                     color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)
            plt.plot(range(startPoint), outSeq[:startPoint,i].numpy(), label='1-step predictions for target'+str(i),
                     color='green', marker='.', linestyle='--', markersize=1.5, linewidth=1)
            # if epoch>40:
            #     plt.plot(range(startPoint, endPoint), upperlim95[:,i].numpy(), label='upperlim'+str(i),
            #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
            #     plt.plot(range(startPoint, endPoint), lowerlim95[:,i].numpy(), label='lowerlim'+str(i),
            #              color='skyblue', marker='.', linestyle='--', markersize=1.5, linewidth=1)
            plt.plot(range(startPoint, endPoint), outSeq[startPoint:,i].numpy(), label='Recursive predictions for target'+str(i),
                     color='blue', marker='.', linestyle='--', markersize=1.5, linewidth=1)

        plt.xlim([startPoint-500, endPoint])
        plt.xlabel('Index',fontsize=15)
        plt.ylabel('Value',fontsize=15)
        plt.title('Time-series Prediction on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.text(startPoint-500+10, target.min(), 'Epoch: '+str(epoch),fontsize=15)
        save_dir = Path('result',args.data,args.filename).with_suffix('').joinpath('fig_prediction')
        save_dir.mkdir(parents=True,exist_ok=True)
        plt.savefig(save_dir.joinpath('fig_epoch'+str(epoch)).with_suffix('.png'))
        #plt.show()
        plt.close()
        return outSeq

    else:
        pass



def evaluate_1step_pred(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    with torch.no_grad():
        hidden = model.init_hidden(args.eval_batch_size)
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):

            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            outSeq, hidden = model.forward(inputSeq, hidden)

            loss = criterion(outSeq.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))
            hidden = model.repackage_hidden(hidden)
            total_loss+= loss.item()

    return total_loss / nbatch

def train(args, model, train_dataset,epoch):

    with torch.enable_grad():
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0
        start_time = time.time()
        hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args,train_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.repackage_hidden(hidden)
            hidden_ = model.repackage_hidden(hidden)
            optimizer.zero_grad()

            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals=[]
            hids1 = []
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals,dim=0)
            hids1 = torch.cat(hids1,dim=0)
            loss1 = criterion(outSeq1.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.view(args.batch_size, -1), targetSeq.view(args.batch_size, -1))

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.view(args.batch_size,-1), hids2.view(args.batch_size,-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1+loss2+loss3
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.4f} | '
                      'loss {:5.2f} '.format(
                    epoch, batch, len(train_dataset) // args.bptt,
                                  elapsed * 1000 / args.log_interval, cur_loss))
                total_loss = 0
                start_time = time.time()

def evaluate(args, model, test_dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model.init_hidden(args.eval_batch_size)
        nbatch = 1
        for nbatch, i in enumerate(range(0, test_dataset.size(0) - 1, args.bptt)):
            inputSeq, targetSeq = get_batch(args,test_dataset, i)
            # inputSeq: [ seq_len * batch_size * feature_size ]
            # targetSeq: [ seq_len * batch_size * feature_size ]
            hidden_ = model.repackage_hidden(hidden)
            '''Loss1: Free running loss'''
            outVal = inputSeq[0].unsqueeze(0)
            outVals=[]
            hids1 = []
            for i in range(inputSeq.size(0)):
                outVal, hidden_, hid = model.forward(outVal, hidden_,return_hiddens=True)
                outVals.append(outVal)
                hids1.append(hid)
            outSeq1 = torch.cat(outVals,dim=0)
            hids1 = torch.cat(hids1,dim=0)
            loss1 = criterion(outSeq1.view(args.batch_size,-1), targetSeq.view(args.batch_size,-1))

            '''Loss2: Teacher forcing loss'''
            outSeq2, hidden, hids2 = model.forward(inputSeq, hidden, return_hiddens=True)
            loss2 = criterion(outSeq2.view(args.batch_size, -1), targetSeq.view(args.batch_size, -1))

            '''Loss3: Simplified Professor forcing loss'''
            loss3 = criterion(hids1.view(args.batch_size,-1), hids2.view(args.batch_size,-1).detach())

            '''Total loss = Loss1+Loss2+Loss3'''
            loss = loss1+loss2+loss3

            total_loss += loss.item()

    return total_loss / (nbatch+1)


# Loop over epochs.
if args.resume or args.pretrained:
    print("=> loading checkpoint ")
    checkpoint = torch.load(Path('save', args.data, 'checkpoint', args.filename).with_suffix('.pth'))
    args, start_epoch, best_val_loss = model.load_checkpoint(args,checkpoint,feature_dim)
    optimizer.load_state_dict((checkpoint['optimizer']))
    del checkpoint
    epoch = start_epoch
    print("=> loaded checkpoint")
else:
    epoch = 1
    start_epoch = 1
    best_val_loss = 0
    print("=> Start training from scratch")
print('-' * 89)
print(args)
print('-' * 89)

if not args.pretrained:
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch, args.epochs+1):

            epoch_start_time = time.time()
            train(args,model,train_dataset,epoch)
            val_loss = evaluate(args,model,test_dataset)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | '.format(epoch, (time.time() - epoch_start_time),                                                                                        val_loss))
            print('-' * 89)

            generate_output(args,epoch,model,gen_dataset,startPoint=1500)

            if epoch%args.save_interval==0:
                # Save the model if the validation loss is the best we've seen so far.
                is_best = val_loss > best_val_loss
                best_val_loss = max(val_loss, best_val_loss)
                model_dictionary = {'epoch': epoch,
                                    'best_loss': best_val_loss,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'args':args
                                    }
                model.save_checkpoint(model_dictionary, is_best)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


# Calculate mean and covariance for each channel's prediction errors, and save them with the trained model
print('=> calculating mean and covariance')
means, covs = list(),list()
train_dataset = TimeseriesData.batchify(args, TimeseriesData.trainData, bsz=1)
for channel_idx in range(model.enc_input_size):
    mean, cov = fit_norm_distribution_param(args,model,train_dataset[:TimeseriesData.length],channel_idx)
    means.append(mean), covs.append(cov)
model_dictionary = {'epoch': max(epoch,start_epoch),
                    'best_loss': best_val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'means': means,
                    'covs': covs
                    }
model.save_checkpoint(model_dictionary, True)
print('-' * 89)