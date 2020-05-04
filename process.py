# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:19:41 2020

@author: 51106
"""
from keras_bert import Tokenizer
import numpy as np
import pandas as pd
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
def load_context(path):
    docid = []
    context=[]
    f = True
    for line in open(path,encoding='utf-8-sig'):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        docid.append(r[0])
        context.append('\t'.join(r[1:]).replace(u'\xa0','').replace(u'\u3000',''))
    output = pd.DataFrame(dtype=str)
    output['docid'] = docid
    output['context'] = context
    return output
def load_train(path):
    id=[]
    docid = []
    question=[]
    answer=[]
    f = True
    for line in open(path,encoding='utf-8-sig'):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        id.append(r[0])
        docid.append(r[1])
        question.append(r[2].replace(u'\xa0','').replace(u'\u3000',''))
        answer.append(r[3].replace(u'\xa0','').replace(u'\u3000',''))
    output = pd.DataFrame(dtype=str)
    output['id'] = id
    output['docid'] = docid
    output['question'] = question
    output['answer'] = answer
    return output
def load_test(path):
    query = []
    id=[]
    f = True
    for line in open(path, encoding='UTF-8'):
        if f:
            f = False
            continue
        r = line.strip().split('\t')
        query.append('\t'.join(r[1:]))
        id.append(r[0])
    output = pd.DataFrame(dtype=str)
    output['query'] = query
    output['id'] = id
    return output
def list_find(list1, list2):
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i,i+n_list2
    return -1,-1


import re
STOPS = (
    '\uFF01'  # Fullwidth exclamation mark
    '\uFF1F'  # Fullwidth question mark
    '\uFF61'  # Halfwidth ideographic full stop
    '\u3002'  # Ideographic full stop
            )

SPLIT_PAT = '([{}]”?)'.format(STOPS)
MULTI_PAT = '(哪[些两几]|分别是)'
def split_text(text, maxlen, split_pat=SPLIT_PAT, greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过maxlen；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        maxlen {int} -- 最大长度
    
    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式
    
    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表
    """
    if len(text) <= maxlen:
        return [text], [0]
    segs = re.split(split_pat, text)
    sentences = []
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]
    alls = []  # 所有满足约束条件的最长子片段
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= maxlen or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        if j == n_sentences - 1:
            if sub[-1] == j:
                break
#    
    if len(alls) == 1:
        return [text], [0]

    if greedy:  # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:  # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


def strip_punct(text, puncts=STOPS):
    return text.strip(puncts)


def is_multiple_answers(question, pat=MULTI_PAT):
    return bool(re.search(pat, question))

def data_strong(train):
    train_new=pd.DataFrame(columns=['id', 'docid', 'question', 'answer', 'context','start','end','tokenize','encode'])
    for i in range(len(train)):
        id=train.loc[i,'id']
        docid=train.loc[i,'docid']
        question=train.loc[i,'question']
        answer=train.loc[i,'answer']
        max_len=512-3-len(question)
        answer_start=train.loc[i,'start']
        answer_end=train.loc[i,'end']
        texts,starts=split_text(train.loc[i,'context'],max_len,greedy=False)
        for text,start in zip(texts,starts):
            if answer_start>=start and answer_end<=len(text)+start:
                token=(tokenizer.tokenize(question)+tokenizer.tokenize(text)[1:])
                encode=tokenizer.encode(first=question,second=text,max_len=512)
                start_new=answer_start-start+len(question)+2
                end_new=len(answer)+start_new
                temp=pd.DataFrame([id,docid,question,answer,text,start_new,end_new,token,encode]).T
                temp.columns=train_new.columns
                train_new=pd.concat([train_new,temp])
            elif answer_start>start+len(text) or answer_end<start:
                token=tokenizer.tokenize(question)+tokenizer.tokenize(text)[1:]
                encode=tokenizer.encode(first=question,second=text,max_len=512)
                start_new=-1
                end_new=-1
                temp=pd.DataFrame([id,docid,question,answer,text,start_new,end_new,token,encode]).T
                temp.columns=train_new.columns
                train_new=pd.concat([train_new,temp])
    train_new=train_new.reset_index(drop=True)
    return train_new
def split_test(test):
    test_new=pd.DataFrame(columns=['id', 'docid', 'query', 'context','start','end','encode','len_question'])
    for i in range(len(test)):
        id=test.loc[i,'id']
        docid=test.loc[i,'docid']
        question=test.loc[i,'query']
        max_len=512-3-len(question)
        texts,starts=split_text(test.loc[i,'context'],max_len,greedy=False)
        for text,start in zip(texts,starts):
            text_encode=tokenizer.encode(first=question,second=text,max_len=512)
            temp=pd.DataFrame([id,docid,question,text,0,0,text_encode,len(question)]).T
            temp.columns=test_new.columns
            test_new=pd.concat([test_new,temp])
    test_new=test_new.reset_index(drop=True)
    return test_new
if __name__ == '__main__':
    config_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
    checkpoint_path = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
    dict_path = 'vocab.txt'
    maxlen=512
    token_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    context=load_context('data/NCPPolicies_context_20200301.csv')
    train=load_train('data/NCPPolicies_train_20200301.csv')
    train = pd.merge(train, context, on='docid', how='left')
    question=train.question.apply(lambda x:len(x))
    start=[]
    end=[]
    data=train[['question','answer','context']].values
    for i in range(len(data)):
        d = data[i]
        context, question = d[2], d[0]
        answer = d[1]
        start_idx,end_idx=list_find(context, answer)
        start.append(start_idx)
        end.append(end_idx)
    start=np.array(start)
    end=np.array(end)
    train['start']=start
    train['end']=end
    train=train[train['start']!=-1]
    train=train.reset_index(drop=True)
    train.to_csv('data/train.csv',index=False)
    tokenizer = OurTokenizer(token_dict)
    train_new=data_strong(train)
    train_new['len_question']=train_new.question.apply(len)
    train_new['class']=train_new.start.apply(lambda x:0 if x==-1 else 1)
    train_new_class1=train_new[train_new['class']==1]
    train_new_class0=train_new[train_new['class']==0]
    train_new_class0=train_new_class0.sample(6012)
    train_new_balance=pd.concat([train_new_class1,train_new_class0],axis=0,ignore_index=True)
    train_new_balance.to_csv('data/train_new_balance.csv',index=False)
    test=pd.read_csv('data/test_top5.csv')
    test_new=split_test(test)
    test_new.to_csv('data/test_new.csv',index=False)
