import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import preprocess_data
from model import model
from torch import optim
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR
from anomalyDetector import fit_norm_distribution_param
from anomalyDetector import anomalyScore
parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model on ecg Dataset')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args_ = parser.parse_args()
print("=> loading checkpoint ")
checkpoint = torch.load('./save/ecg/checkpoint.pth.tar')
print("=> loaded checkpoint")
args = checkpoint['args']
args.prediction_window_size= args_.prediction_window_size

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
TimeseriesData = preprocess_data.DataLoad(args.data)
train_dataset = preprocess_data.batchify(args,TimeseriesData.trainData, 1)[:5000]
test_dataset = preprocess_data.batchify(args,TimeseriesData.testData, 1)



###############################################################################
# Build the model
###############################################################################

model = model.RNNPredictor(rnn_type = args.model, enc_inp_size=2, rnn_inp_size = args.emsize, rnn_hid_size = args.nhid,
                           dec_out_size=2,
                           nlayers = args.nlayers,)
model.load_state_dict(checkpoint['state_dict'])
del checkpoint

if args.cuda:
    model.cuda()


for channel_idx in range(TimeseriesData.trainData.size(-1)):
    mean, cov = fit_norm_distribution_param(args, model, train_dataset, channel_idx=channel_idx)
    train_scores, _, _, hiddens,_ = anomalyScore(args, model, train_dataset, mean, cov, 3000)
    score_predictor = SVR(C=1.0,epsilon=0.2)
    score_predictor.fit(torch.cat(hiddens,dim=0).numpy(),train_scores)

    scores, sorted_predictions,sorted_errors, _, predicted_scores = anomalyScore(args, model, test_dataset, mean, cov, endPoint,
                                                                                score_predictor=score_predictor,channel_idx=1)

    sorted_predictions = torch.cat(sorted_predictions, dim=0)
    sorted_errors = torch.cat(sorted_errors,dim=0)

    scores = np.array(scores)

    target= preprocess_data.reconstruct(test_dataset.cpu()[:, 0,1].numpy(),
                                         TimeseriesData.mean,
                                         TimeseriesData.std)
    sorted_predictions_mean = preprocess_data.reconstruct(sorted_predictions.mean(dim=1).numpy(),
                                         TimeseriesData.trainData['seqData2_mean'],
                                         TimeseriesData.trainData['seqData2_std'])

    sorted_predictions_1step = preprocess_data.reconstruct(sorted_predictions[:,-1].numpy(),
                                         TimeseriesData.trainData['seqData2_mean'],
                                         TimeseriesData.trainData['seqData2_std'])

    sorted_predictions_Nstep = preprocess_data.reconstruct(sorted_predictions[:,0].numpy(),
                                         TimeseriesData.trainData['seqData2_mean'],
                                         TimeseriesData.trainData['seqData2_std'])
    #sorted_errors_mean = sorted_errors.mean(dim=1).abs().cpu().numpy()
    sorted_errors_mean = sorted_errors.abs().mean(dim=1).cpu().numpy()

    sorted_errors_mean *=TimeseriesData.trainData['seqData2_std']

    fig, ax1 = plt.subplots(figsize=(15,5))
    ax1.plot(target,label='Target', color='black', marker='.', linestyle='--', markersize=1, linewidth=0.5)

    ax1.plot(sorted_predictions_mean,label='Mean predictions', color='purple', marker='.', linestyle='--', markersize=1, linewidth=0.5)
    ax1.plot(sorted_predictions_1step,label='1-step predictions', color='green', marker='.', linestyle='--', markersize=1, linewidth=0.5)
    ax1.plot(sorted_predictions_Nstep,label=str(args.prediction_window_size)+'-step predictions', color='blue', marker='.', linestyle='--', markersize=1, linewidth=0.5)
    ax1.plot(sorted_errors_mean,label='Absolute mean prediction errors', color='orange', marker='.', linestyle='--', markersize=1, linewidth=1)

    ax1.legend(loc='upper left')
    ax1.set_ylabel('Value',fontsize=15)
    ax1.set_xlabel('Index',fontsize=15)

    ax2 = ax1.twinx()
    ax2.plot(scores.reshape(3500,1)/predicted_scores,label='Anomaly scores from \nmultivariate normal distribution', color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
    #ax2.plot(predicted_scores,label='Predicted anomaly scores from SVR', color='cyan', marker='.', linestyle='--', markersize=1, linewidth=1)

    ax2.legend(loc='upper right')
    ax2.set_ylabel('anomaly score',fontsize=15)
    plt.axvspan(2830,2900 , color='yellow', alpha=0.3)
    plt.xlim([0, endPoint])
    plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.xlim([1500,endPoint])
    #plt.xlim([2700,3000])
    plt.savefig('result/'+args.data+'/fig_scores.png')
    plt.show()
