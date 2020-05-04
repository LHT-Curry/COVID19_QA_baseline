# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 00:21:58 2020

@author: 51106
"""
from keras_bert import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras_radam import RAdam
import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
import logging
from tqdm import tqdm
import time
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import log_loss
import heapq
file_path='log/'
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)
config_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'
MAX_LEN=512
epoch=2
EPSILON=1e-7
w1=0.9
w2=0.1
def seq_padding(X, padding=0):
    ML = MAX_LEN
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:MAX_LEN] for x in X
    ])

train_new=pd.read_csv('data/train_new_balance.csv')

data=train_new[['context','start','end','encode','len_question']].values
label=train_new['class'].values
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)
class data_generator:
    def __init__(self, data, batch_size=6):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, S1, S2, Class = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                start,end = d[1],d[2]
                x1=d[3][0]
                x2=d[3][1]
                if start!=-1 and end!=-1:
                    s1=to_categorical(start)
                    s2=to_categorical(end)
                    clss=1
                else:
                    s1=np.array([0])
                    s2=np.array([0])
                    clss=0
                X1.append(x1)
                X2.append(x2)
                S1.append(s1)
                S2.append(s2)
                Class.append([clss])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    S1=seq_padding(S1)
                    S2=seq_padding(S2)
                    X1 = np.array(X1)
                    X2=np.array(X2)
                    Class=np.array(Class)
                    yield [X1, X2, S1, S2], [Class, S1, S2]
                    X1, X2, S1, S2, Class = [], [], [], [], []
def LCS(str_a, str_b):
        lensum = float(len(str_a) + len(str_b))
        lengths = [[0 for j in range(len(str_b)+1)] for i in range(len(str_a)+1)]
        for i, x in enumerate(str_a):
            for j, y in enumerate(str_b):
                if x == y:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
        result = ""
        x, y = len(str_a), len(str_b)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else: #用到的从后向前的当前一个字符
                assert str_a[x-1] == str_b[y-1] #后面语句为真，类似于if(a[x-1]==b[y-1]),执行后条件下的语句
                result = str_a[x-1] + result #注意这一句，这是一个从后向前的过程
                x -= 1
                y -= 1
        longestdist = lengths[len(str_a)][len(str_b)]
#        ratio = longestdist/min(len(str_a),len(str_b))
        return longestdist
def Rouge_L(prediction, ground_truth, beta=1.2):  #ROUGE_L score
    lcs = LCS(prediction, ground_truth)
    len_g = len(ground_truth)
    len_p = len(prediction)
    R_lcs = lcs / float(len_g) if len_g > 0 else 0.
    P_lcs = lcs / float(len_p) if len_p > 0 else 0.
    F_lcs = (1 + beta ** 2) * R_lcs * P_lcs / (R_lcs + (beta ** 2) * P_lcs + EPSILON)
    return F_lcs
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import *
def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None,)) # 待识别句子输入
    x2_in = Input(shape=(None,)) # 待识别句子输入
    s1_in = Input(shape=(None,)) # 实体左边界（标签）
    s2_in = Input(shape=(None,)) # 实体右边界（标签）

    x1, x2, s1, s2= x1_in, x2_in, s1_in, s2_in
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

    x = bert_model([x1, x2])
    x1= Lambda(lambda x: x[:, 0])(x)
    pclss=Dense(1, activation='sigmoid',name='class')(x1)
    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps1=Activation('softmax',name='start')(ps1)
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])
    ps2=Activation('softmax',name='end')(ps2)
    model = Model([x1_in, x2_in], [pclss, ps1, ps2])
    train_model = Model([x1_in, x2_in, s1_in, s2_in], [pclss, ps1, ps2])
    train_model.compile(loss=[
            'binary_crossentropy',
            'categorical_crossentropy',
            'categorical_crossentropy'],loss_weights=[1,1,1],
            optimizer=Adam(1e-5))
    train_model.summary()
    return train_model,model
train_start=np.zeros((len(train_new),MAX_LEN), dtype=np.float32)
train_end=np.zeros((len(train_new),MAX_LEN), dtype=np.float32)
train_start_loc = np.zeros((len(train_new),), dtype=np.int32)
train_end_loc = np.zeros((len(train_new), ), dtype=np.int32)
train_class=np.zeros((len(train_new),),dtype=np.float32)
train_answer_score = np.zeros((len(train_new), ), dtype=np.float32)

test_new=pd.read_csv('data/test_new.csv')
test_data=test_new[['context','start','end','encode','len_question']].values
test_start=np.zeros((len(test_new),MAX_LEN), dtype=np.float32)
test_end=np.zeros((len(test_new),MAX_LEN), dtype=np.float32)
test_class=np.zeros((len(test_new),),dtype=np.float32)
def prediction(val_data):
    Class=[]
    Start=[]
    End=[]
    for i in tqdm(range(len(val_data))):
        d=val_data[i]
        x1,x2=d[3]
        T1, T1_ = np.array([x1]), np.array([x2])
        out_class, out_s1,out_s2 = model.predict([T1, T1_])
        out_class_, out_s1_,out_s2_=out_class[0][0], out_s1[0], out_s2[0]
        Start.append(out_s1_)
        End.append(out_s2_)
        Class.append(out_class_)
    return Class,Start,End
def score(val_data,Start,End,Class,w1,w2):
    epsilon = 1e-3
    Start_loc=[]
    End_loc=[]
    Score=[]
    Answer_score=[]
    for i in tqdm(range(len(val_data))):
        d=val_data[i]
        len_question=d[4]
        out_class_, out_s1_,out_s2_=Class[i], Start[i], End[i]
        start_new=heapq.nlargest(3,range(len(out_s1_[len_question+2:])),out_s1_[len_question+2:].take)
        end_new=heapq.nlargest(3,range(len(out_s2_[len_question+2:])),out_s2_[len_question+2:].take)
        max_answer_score=0
        for j in start_new:
            for k in end_new:
                if j<k:
                    start_=out_s1_[len_question+2:][j]
                    end_=out_s2_[len_question+2:][k]
                    answer_score=np.exp((0.5 * np.log(start_ + epsilon) + 0.5 * np.log(end_ + epsilon)) / (0.5 + 0.5))
                    if answer_score>max_answer_score:
                        max_start=j
                        max_end=k
                        max_answer_score=answer_score   
        Answer_score.append(max_answer_score)
        score=np.exp((w1 * np.log(out_class_ + epsilon) + w2 * np.log(max_answer_score + epsilon)) / (w1 + w2))
        Score.append(score)
        Start_loc.append(int(max_start))
        End_loc.append(int(max_end))
    return Start_loc,End_loc,Score,Answer_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=223344)
for fold, (train_index, valid_index) in enumerate(skf.split(data,label)):
    logger.info('================     fold {}        ==============='.format(fold))
    train_data = data[train_index]
    val_data = data[valid_index]
    train_D = data_generator(train_data)
    valid_D = data_generator(val_data)
    train_model,model = get_model()
    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.5, patience=1)
    checkpoint = ModelCheckpoint('model_save/' + str(fold) + '.w', monitor='val_loss', 
                                     verbose=2, save_best_only=True, mode='min',save_weights_only=True)
    history=train_model.fit_generator(train_D.__iter__(),
                        steps_per_epoch=len(train_D),
                        epochs=epoch,
                        validation_data=valid_D.__iter__(),
                        validation_steps=len(valid_D),
                        callbacks=[plateau, checkpoint])
    model.load_weights('model_save/{}.w'.format(fold))
    
    Class,Start,End=prediction(val_data)
    Start_loc,End_loc,Score,Answer_score=score(val_data,Start,End,Class,w1,w2)
    test_Class,test_Start,test_End=prediction(test_data)

    train_start_loc[valid_index] = Start_loc
    train_end_loc[valid_index] = End_loc
    train_class[valid_index] = Class
    train_start[valid_index] = Start
    train_end[valid_index] = End
    train_answer_score[valid_index] = Score

    test_class += test_Class
    test_start += test_Start
    test_end += test_End
    for i in range(epoch):
        logger.info('epoch: %d, class_loss: %.4f, start_loss: %.4f, end_loss: %.4f, val_class_loss: %.4f, val_start_loss: %.4f, val_end_loss: %.4f\n' % (i, history.history['class_loss'][i], history.history['start_loss'][i], history.history['end_loss'][i],history.history['val_class_loss'][i], history.history['val_start_loss'][i], history.history['val_end_loss'][i]))
    K.clear_session()
train_new['start_pre']=train_start_loc
train_new['end_pre']=train_end_loc
train_new['class_pre']=train_class
train_new['score']=train_answer_score
tmp=train_new.groupby('id')['score'].idxmax()
tmp=np.array(tmp)
train_new=train_new.loc[tmp,:]
train_new['result']=train_new.apply(lambda x:Rouge_L(x['context'][x['start_pre']:x['end_pre']], x['answer']),axis=1)

test_class/=5
test_start/=5
test_end/=5

test_start_loc,test_end_loc,test_score=score(test_data,test_start,test_end,test_class,0.9,0.1)
test_new['start_pre']=test_start_loc
test_new['end_pre']=test_end_loc
test_new['score']=test_score
tmp=test_new.groupby('id')['score'].idxmax()
tmp=np.array(tmp)
submit=test_new.loc[tmp,:]
submit=submit[['id','docid','answer']]
submit.to_csv('submitcsv',sep='\t',index=False)
