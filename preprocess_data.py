import os
import torch
import glob
import datetime
import numpy as np
import shutil
from pathlib import Path
import pickle

def normalization(seqData,max,min):
    return (seqData -min)/(max-min)

def standardization(seqData,mean,std):
    return (seqData-mean)/std

def reconstruct(seqData,mean,std):
    return seqData*std+mean


def DataLoad(data_type):
    if data_type == 'nyc_taxi':
        return NYCDataLoad('./dataset/'+ data_type +'/')
    elif data_type == 'gesture':
        return GestureDataLoad('./dataset/' + data_type + '/')
    elif data_type == 'ecg':
        return ECGDataLoad('./dataset/' + data_type + '/')


class PickleDataLoad(object):
    def __init__(self, data_type,filename,augment=True):
        self.augment=augment
        self.trainData, self.trainLabel = self.preprocessing(Path('dataset',data_type,'labeled','train',filename),train=True)
        self.testData, self.testLabel = self.preprocessing(Path('dataset',data_type,'labeled','test',filename),train=False)

    def augmentation(self,data,label):
        noiseSeq = torch.randn(data.size())
        augmentedData = data.clone()
        augmentedLabel = label.clone()
        noise_ratio = 0.1
        for i in np.arange(0, noise_ratio, 0.005):
            scaled_noiseSeq = noise_ratio * self.std.expand_as(data) * noiseSeq
            augmentedData = torch.cat([augmentedData, data + scaled_noiseSeq], dim=0)
            augmentedLabel = torch.cat([augmentedLabel, label])
        return augmentedData, augmentedLabel

    def preprocessing(self, path, train=True):
        """ Read, Standardize, Augment """

        with open(str(path), 'rb') as f:
            data = torch.FloatTensor(pickle.load(f))
            label = data[:,-1]
            data = data[:,:-1]
        if train:
            self.mean = data.mean(dim=0)
            self.std= data.std(dim=0)

        if self.augment:
            data,label = self.augmentation(data,label)
        data = standardization(data,self.mean,self.std)

        return data,label

    def batchify(self,args,data, bsz):
        nbatch = data.size(0) // bsz
        trimmed_data = data.narrow(0,0,nbatch * bsz)
        batched_data = trimmed_data.contiguous().view(bsz, -1, trimmed_data.size(-1)).transpose(0,1)
        if args.cuda:
            batched_data = batched_data.cuda()
        return batched_data




class NYCDataLoad(object):
    def __init__(self, path,augumentation=True):
        self.augumentation=augumentation
        self.trainData = self.preprocessing(path + 'trainset/nyc_taxi.csv', trainData=True)
        self.testData = self.preprocessing(path + 'testset/nyc_taxi.csv', trainData=False)

    def preprocessing(self, path, trainData=True):
        """ Read, Standardize, Augment """
        seqData = []
        timeOfDay = []
        dayOfWeek = []
        dataset={}

        noise_val=0.8 if self.augumentation else 0.1
        for std in np.arange(0,noise_val,0.1):
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i > 0:
                        line_splited = line.strip().split(',')
                        # print line_splited
                        seqData.append(float(line_splited[1]) + np.random.normal(0, std))
                        timeOfDay.append(float(line_splited[2]))
                        dayOfWeek.append(float(line_splited[3]))

        if trainData:
            dataset['seqData_mean'] = np.mean(seqData)
            dataset['seqData_std']= np.std(seqData, ddof=1)
            dataset['timeOfDay_mean'] = np.mean(timeOfDay)
            dataset['timeOfDay_std'] = np.std(timeOfDay,ddof=1)
            dataset['dayOfWeek_mean'] = np.mean(dayOfWeek)
            dataset['dayOfWeek_std'] = np.std(dayOfWeek,ddof=1)

            #print 'seqData_mean =', seqData_mean
            #print 'seqData_std =', seqData_std
        else:
            dataset['seqData_mean'] = self.trainData['seqData_mean']
            dataset['seqData_std'] = self.trainData['seqData_std']
            dataset['timeOfDay_mean'] = self.trainData['timeOfDay_mean']
            dataset['timeOfDay_std'] = self.trainData['timeOfDay_std']
            dataset['dayOfWeek_mean'] = self.trainData['dayOfWeek_mean']
            dataset['dayOfWeek_std'] = self.trainData['dayOfWeek_std']

        seqData = torch.FloatTensor(seqData)
        seqData = standardization(seqData, dataset['seqData_mean'], dataset['seqData_std'])
        if self.augumentation:
            seqData_corrupted1 = seqData + 0.1*torch.randn(seqData.size(0))
            seqData_corrupted2 = seqData + 0.2*torch.randn(seqData.size(0))
            seqData = torch.cat([seqData,seqData_corrupted1],0)
            seqData = torch.cat([seqData, seqData_corrupted2], 0)
        dataset['seqData'] = seqData

        timeOfDay = torch.FloatTensor(timeOfDay)
        timeOfDay = standardization(timeOfDay, dataset['timeOfDay_mean'], dataset['timeOfDay_std'])

        if self.augumentation:
            timeOfDay_temp = torch.cat([timeOfDay, timeOfDay], 0)
            timeOfDay = torch.cat([timeOfDay, timeOfDay_temp], 0)
        dataset['timeOfDay'] = timeOfDay

        dayOfWeek = torch.FloatTensor(dayOfWeek)
        dayOfWeek = standardization(dayOfWeek, dataset['dayOfWeek_mean'], dataset['dayOfWeek_std'])

        if self.augumentation:
            dayOfWeek_temp = torch.cat([dayOfWeek, dayOfWeek], 0)
            dayOfWeek = torch.cat([dayOfWeek, dayOfWeek_temp], 0)
        dataset['dayOfWeek'] = dayOfWeek


        return dataset


class ECGDataLoad(object):
    def __init__(self, path,augumentation=True):
        self.augumentation=augumentation
        self.trainData = self.preprocessing(path + 'trainset/chfdb_chf13_45590.txt', trainData=True)
        self.testData = self.preprocessing(path + 'testset/chfdb_chf13_45590.txt', trainData=False)



    def preprocessing(self, path, trainData=True):
        """ Read, Standardize, Augment """
        seqData1 = []
        seqData2 = []

        dataset={}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    line_splited = line.strip().split(' ')
                    line_splited = [x for x in line_splited if x != '']
                    #print(line_splited)
                    seqData1.append(float(line_splited[1].strip()))
                    seqData2.append(float(line_splited[2].strip()))


        if trainData:
            dataset['seqData1_mean'] = np.mean(seqData1)
            dataset['seqData1_std']= np.std(seqData1, ddof=1)
            dataset['seqData2_mean'] = np.mean(seqData2)
            dataset['seqData2_std'] = np.std(seqData2, ddof=1)

            #print 'seqData_mean =', seqData_mean
            #print 'seqData_std =', seqData_std
        else:
            dataset['seqData1_mean'] = self.trainData['seqData1_mean']
            dataset['seqData1_std'] = self.trainData['seqData1_std']
            dataset['seqData2_mean'] = self.trainData['seqData2_mean']
            dataset['seqData2_std'] = self.trainData['seqData2_std']

        seqData1 = torch.FloatTensor(seqData1)
        seqData1_original = seqData1.clone()
        seqData2 = torch.FloatTensor(seqData2)
        seqData2_original = seqData2.clone()
        seq_length = len(seqData1)
        if self.augumentation:
            noise_ratio=0.2
            for i in np.arange(0,noise_ratio,0.01):
                tempSeq1 = noise_ratio*dataset['seqData1_std']* torch.randn(seq_length)
                seqData1 = torch.cat([seqData1,seqData1_original+tempSeq1])
                tempSeq2 = noise_ratio * dataset['seqData2_std'] * torch.randn(seq_length)
                seqData2 = torch.cat([seqData2, seqData2_original + tempSeq2])

        seqData1 = standardization(seqData1,dataset['seqData1_mean'],dataset['seqData1_std'])
        seqData2 = standardization(seqData2,dataset['seqData2_mean'],dataset['seqData2_std'])

        dataset['seqData1'] = seqData1
        dataset['seqData2'] = seqData2


        return dataset

class GestureDataLoad(object):
    def __init__(self, path,augumentation=True):
        self.augumentation=augumentation
        self.trainData = self.preprocessing(path + 'trainset/ann_gun_CentroidA.txt', trainData=True)
        self.testData = self.preprocessing(path + 'testset/ann_gun_CentroidA.txt', trainData=False)



    def preprocessing(self, path, trainData=True):
        """ Read, Standardize, Augment """
        seqData1 = []
        seqData2 = []

        dataset={}

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    line_splited = line.strip().split('\t')
                    line_splited = [x for x in line_splited if x != '']
                    if line_splited==[''] or line_splited==[]:
                        continue
                    #print(line_splited)
                    seqData1.append(float(line_splited[0].strip()))
                    seqData2.append(float(line_splited[1].strip()))


        if trainData:
            dataset['seqData1_mean'] = np.mean(seqData1)
            dataset['seqData1_std']= np.std(seqData1, ddof=1)
            dataset['seqData2_mean'] = np.mean(seqData2)
            dataset['seqData2_std'] = np.std(seqData2, ddof=1)

            #print 'seqData_mean =', seqData_mean
            #print 'seqData_std =', seqData_std
        else:
            dataset['seqData1_mean'] = self.trainData['seqData1_mean']
            dataset['seqData1_std'] = self.trainData['seqData1_std']
            dataset['seqData2_mean'] = self.trainData['seqData2_mean']
            dataset['seqData2_std'] = self.trainData['seqData2_std']

        seqData1 = torch.FloatTensor(seqData1)
        seqData1_original = seqData1.clone()
        seqData2 = torch.FloatTensor(seqData2)
        seqData2_original = seqData2.clone()
        seq_length = len(seqData1)
        if self.augumentation:
            noise_ratio=0.2
            for i in np.arange(0,noise_ratio,0.01):
                tempSeq1 = noise_ratio*dataset['seqData1_std']* torch.randn(seq_length)
                seqData1 = torch.cat([seqData1,seqData1_original+tempSeq1])
                tempSeq2 = noise_ratio * dataset['seqData2_std'] * torch.randn(seq_length)
                seqData2 = torch.cat([seqData2, seqData2_original + tempSeq2])

        seqData1 = standardization(seqData1,dataset['seqData1_mean'],dataset['seqData1_std'])
        seqData2 = standardization(seqData2,dataset['seqData2_mean'],dataset['seqData2_std'])

        dataset['seqData1'] = seqData1
        dataset['seqData2'] = seqData2


        return dataset




def batchify(args,data, bsz):

    if args.data == 'nyc_taxi':
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data['seqData'].size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        dataset = {}
        for key in ['seqData','timeOfDay','dayOfWeek']:
            dataset[key] = data[key].narrow(0, 0, nbatch * bsz)
            # Evenly divide the data across the bsz batches.
            dataset[key] = dataset[key].view(bsz, -1).t().contiguous().unsqueeze(2) # data: [ seq_length * batch_size * 1 ]

        batched_data = torch.cat([dataset['seqData'],dataset['timeOfDay'],dataset['dayOfWeek']],dim=2)
        # batched_data: [ seq_length * batch_size * feature_size ] , feature_size = 3
    elif args.data == 'ecg' or args.data =='gesture':
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data['seqData1'].size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        dataset = {}
        for key in ['seqData1', 'seqData2']:
            dataset[key] = data[key].narrow(0, 0, nbatch * bsz)
            # Evenly divide the data across the bsz batches.
            dataset[key] = dataset[key].view(bsz, -1).t().contiguous().unsqueeze(
                2)  # data: [ seq_length * batch_size * 1 ]

        batched_data = torch.cat([dataset['seqData1'], dataset['seqData2']], dim=2)
        # batched_data: [ seq_length * batch_size * feature_size ] , feature_size = 2

    if args.cuda:
        batched_data = batched_data.cuda()

    return batched_data