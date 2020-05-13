from __future__ import print_function
import warnings
import os
from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np


warnings.filterwarnings('ignore')

def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave, fs=sample_rate)[0]
    return mfcc_features

def build_data_set(dir):
    file_list = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for f in file_list:
        tmp = f.split('.')[0]
        label = tmp.split('_')[1]
        feature = extract_mfcc(dir + f)
        if label not in dataset.keys():
            dataset[label] = []
        dataset[label].append(feature)
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 5
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=1)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models

def length(dic):
    l = 0
    for key in dic.keys():
        #print(key)
        l += len(dic[key])
        #print(l)
    return l

train = './train_audio/'
train_data = build_data_set(train)
print("finish preparing the training data")

models = train_GMMHMM(train_data)
print("finish training model")

test = './test_audio/'
test_data = build_data_set(test)
print("finish preparing the testing data")

score_cnt = 0
for label in test_data.keys():
    for f in test_data[label]:
        score_list = {}
        for model_label in models.keys():
            model = models[model_label]
            score = model.score(f)
            score_list[model_label] = score
        predict = max(score_list, key=score_list.get)
        print("test on label ", label, ", predict result: ", predict)
        if predict == label:
            score_cnt += 1
            
print("Final recognition rate is %.2f"%(100.0*score_cnt/length(test_data)), "%")
