from torch.autograd import Variable
import torch
import numpy as np

def fit_norm_distribution_param(args, model, train_dataset, channel_idx=0):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    pasthidden = model.init_hidden(1)
    predictions = []
    organized = []
    errors = []
    #out = Variable(test_dataset[0].unsqueeze(0))
    for t in range(len(train_dataset)):
        out, hidden = model.forward(Variable(train_dataset[t].unsqueeze(0), volatile=True), pasthidden)
        predictions.append([])
        organized.append([])
        errors.append([])
        predictions[t].append(out.data.cpu()[0][0][channel_idx])
        pasthidden = model.repackage_hidden(hidden)
        for prediction_step in range(1,args.prediction_window_size):
            out, hidden = model.forward(out, hidden)
            predictions[t].append(out.data.cpu()[0][0][channel_idx])

        if t >= args.prediction_window_size:
            for step in range(args.prediction_window_size):
                organized[t].append(predictions[step+t-args.prediction_window_size][args.prediction_window_size-1-step])
            organized[t]= torch.FloatTensor(organized[t])
            if args.cuda:
                organized[t]= organized[t].cuda()
            errors[t] = organized[t] - train_dataset[t][0][channel_idx]
            if args.cuda:
                errors[t] = errors[t].cuda()
            errors[t] = errors[t].unsqueeze(0)

    errors_tensor = torch.cat(errors[args.prediction_window_size:],dim=0)
    mean = errors_tensor.mean(dim=0)
    cov = errors_tensor.t().mm(errors_tensor)/errors_tensor.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(0))
    # cov: positive-semidefinite and symmetric.

    return mean, cov


def anomalyScore(args, model, dataset, mean, cov, channel_idx=0, score_predictor=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    pasthidden = model.init_hidden(1)
    predictions = []
    organized = []
    errors = []
    hiddens = []
    predicted_scores = []
    # out = Variable(test_dataset[0].unsqueeze(0))
    for t in range(len(dataset)):
        out, hidden = model.forward(Variable(dataset[t].unsqueeze(0), volatile=True), pasthidden)
        predictions.append([])
        organized.append([])
        errors.append([])
        hiddens.append(model.extract_hidden(hidden))
        if score_predictor is not None:
            predicted_scores.append(score_predictor.predict(model.extract_hidden(hidden).numpy()))

        predictions[t].append(out.data.cpu()[0][0][channel_idx])
        pasthidden = model.repackage_hidden(hidden)
        for prediction_step in range(1, args.prediction_window_size):
            out, hidden = model.forward(out, hidden)
            predictions[t].append(out.data.cpu()[0][0][channel_idx])

        if t >= args.prediction_window_size:
            for step in range(args.prediction_window_size):
                organized[t].append(
                    predictions[step + t - args.prediction_window_size][args.prediction_window_size - 1 - step])
            organized[t] =torch.FloatTensor(organized[t]).unsqueeze(0)
            if args.cuda:
                organized[t] = organized[t].cuda()
            errors[t] = organized[t] - dataset[t][0][channel_idx]
            if args.cuda:
                errors[t] = errors[t].cuda()
        else:
            organized[t] = torch.zeros(1,args.prediction_window_size)
            errors[t] = torch.zeros(1,args.prediction_window_size)
            if args.cuda:
                organized[t] = organized[t].cuda()
                errors[t] = errors[t].cuda()
    predicted_scores = np.array(predicted_scores)
    scores = []
    for error in errors:
        mult1 = error-mean.unsqueeze(0) # [ 1 * prediction_window_size ]
        mult2 = torch.inverse(cov) # [ prediction_window_size * prediction_window_size ]
        mult3 = mult1.t() # [ prediction_window_size * 1 ]
        score = torch.mm(mult1,torch.mm(mult2,mult3))
        scores.append(score[0][0])
    return scores, organized, errors,hiddens, predicted_scores
