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
parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model on gesture Dataset')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
args_ = parser.parse_args()
print("=> loading checkpoint ")
checkpoint = torch.load('./save/gesture/checkpoint.pth.tar')
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
train_dataset = preprocess_data.batchify(args,TimeseriesData.trainData, 1)[:10000]
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


# def anomalyScore(args,model,test_dataset,mean,cov,endPoint=10000):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     pasthidden = model.init_hidden(1)
#     predictions = []
#     organized = []
#     errors = []
#     # out = Variable(test_dataset[0].unsqueeze(0))
#     for t in range(endPoint):
#         out, hidden = model.forward(Variable(test_dataset[t].unsqueeze(0), volatile=True), pasthidden)
#         predictions.append([])
#         organized.append([])
#         errors.append([])
#         predictions[t].append(out.data.cpu()[0][0][0])
#         pasthidden = model.repackage_hidden(hidden)
#         for prediction_step in range(1, args.prediction_window_size):
#             out, hidden = model.forward(out, hidden)
#             predictions[t].append(out.data.cpu()[0][0][0])
#
#         if t >= args.prediction_window_size:
#             for step in range(args.prediction_window_size):
#                 organized[t].append(
#                     predictions[step + t - args.prediction_window_size][args.prediction_window_size - 1 - step])
#             organized[t] =torch.FloatTensor(organized[t]).unsqueeze(0)
#             errors[t] = organized[t] - test_dataset[t][0][0]
#             if args.cuda:
#                 errors[t] = errors[t].cuda()
#         else:
#             organized[t] = torch.zeros(1,args.prediction_window_size)
#             errors[t] = torch.zeros(1,args.prediction_window_size)
#             if args.cuda:
#                 errors[t] = errors[t].cuda()
#
#     scores = []
#     for error in errors:
#         mult1 = error-mean.unsqueeze(0) # [ 1 * prediction_window_size ]
#         reg = torch.eye(args.prediction_window_size)
#         if args.cuda:
#             reg = reg.cuda()
#         mult2 = torch.inverse(cov+0.0001*reg) # [ prediction_window_size * prediction_window_size ]
#         mult3 = mult1.t() # [ prediction_window_size * 1 ]
#         score = torch.mm(mult1,torch.mm(mult2,mult3)) # [ 1 * 1 ]
#         scores.append(score[0][0])
#     return scores, organized, errors


# At any point you can hit Ctrl + C t
endPoint=3000
try:

   mean, cov = fit_norm_distribution_param(args, model, train_dataset, endPoint,channel_idx=0)
   # scores, sorted_predictions,sorted_errors = anomalyScore(args, model, test_dataset, mean, cov, endPoint)
   #
   # sorted_predictions = torch.cat(sorted_predictions, dim=0)
   # sorted_errors = torch.cat(sorted_errors,dim=0)
   #
   # scores = np.array(scores)
   train_scores, _, _, hiddens,_ = anomalyScore(args, model, train_dataset, mean, cov, 3000)
   score_predictor = SVR(C=1.0,epsilon=0.2)
   score_predictor.fit(torch.cat(hiddens,dim=0).numpy(),train_scores)

   scores, sorted_predictions,sorted_errors, _, predicted_scores = anomalyScore(args, model, test_dataset, mean, cov, endPoint,score_predictor=score_predictor)

   sorted_predictions = torch.cat(sorted_predictions, dim=0)
   sorted_errors = torch.cat(sorted_errors,dim=0)
   scores = np.array(scores)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

target= preprocess_data.reconstruct(test_dataset.cpu()[:, 0,0].numpy(),
                                     TimeseriesData.trainData['seqData1_mean'],
                                     TimeseriesData.trainData['seqData1_std'])
sorted_predictions_mean = preprocess_data.reconstruct(sorted_predictions.mean(dim=1).numpy(),
                                     TimeseriesData.trainData['seqData1_mean'],
                                     TimeseriesData.trainData['seqData1_std'])

sorted_predictions_1step = preprocess_data.reconstruct(sorted_predictions[:,-1].numpy(),
                                     TimeseriesData.trainData['seqData1_mean'],
                                     TimeseriesData.trainData['seqData1_std'])

sorted_predictions_Nstep = preprocess_data.reconstruct(sorted_predictions[:,0].numpy(),
                                     TimeseriesData.trainData['seqData1_mean'],
                                     TimeseriesData.trainData['seqData1_std'])
#sorted_errors_mean = sorted_errors.mean(dim=1).abs().cpu().numpy()
sorted_errors_mean = sorted_errors.abs().mean(dim=1).cpu().numpy()

sorted_errors_mean *=TimeseriesData.trainData['seqData1_std']

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
ax2.plot(scores,label='Anomaly scores from \nmultivariate normal distribution', color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
ax2.plot(predicted_scores,label='Predicted anomaly scores from SVR', color='cyan', marker='.', linestyle='--', markersize=1, linewidth=1)

ax2.legend(loc='upper right')
ax2.set_ylabel('anomaly score',fontsize=15)
plt.axvspan(2100,2500, color='yellow', alpha=0.3)
plt.xlim([0, endPoint])
plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.xlim([0,endPoint])
#plt.xlim([1900,2600])
plt.savefig('result/'+args.data+'/fig_scores.png')
plt.show()
