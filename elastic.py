# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:19:27 2020

@author: 51106
"""

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
from process import load_context,load_test
class ElasticObj:
    def __init__(self, index_name,index_type,ip ="127.0.0.1"):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        '''
        self.index_name =index_name
        self.index_type = index_type
        # 无用户名密码状态
        self.es = Elasticsearch([ip])
        
    def Get_Data_By_Body(self, question, k):
        doc = {
            "size": k, 
            "query": {
                "match": {
                  "passage": question
                }
              }
        }
        try:
            _searched = self.es.search(index=self.index_name, doc_type=self.index_type, body=doc)
            answers = []
            for item in _searched['hits']['hits']:
                answers.append((item['_source']['passage'], item['_source']['docid'])) 
            return answers

        except:
            print('search not exist')
            print(question)

    def bulk_Index_Data(self, csvfile):
        '''
        用bulk将批量数据存储到es
        :return:
        '''
        df = load_context(csvfile)
        doc = []
        for item in df.values:
            dic = {}
            dic['docid'] = item[0]
            dic['passage'] = item[1]
            doc.append(dic)
        ACTIONS = []
        i = 0
        for line in doc:
            action = {
                "_index": self.index_name,
                "_type": self.index_type,
                "_source": {
                    "docid": line['docid'],
                    "passage": line['passage']}
            }
            i += 1
            ACTIONS.append(action)
            # 批量处理
        print('index_num:',i)
        success, _ = bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)
        

    def create_index(self,index_name,index_type):
        '''
        创建索引,创建索引名称为ott，类型为ott_type的索引
        :param ex: Elasticsearch对象
        :return:
        '''
        #创建映射
        _index_mappings = {
            "mappings": {
                    "properties":{
                          "passage":{
                            "type":"text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_max_word"
                          },
                          "docid":{
                            "type": "text"
                          }
                        }
            }
        }
        if self.es.indices.exists(index=self.index_name) is not True:
            res = self.es.indices.create(index=self.index_name, body=_index_mappings)
            print(res)

def read_examples(input_file,  k=1, es_index="passage", es_ip="localhost"):
    df=load_test(input_file)
    obj = ElasticObj(es_index,"_doc",ip =es_ip)
    examples=[]
    for val in df[['id','query']].values:
            answers = obj.Get_Data_By_Body(val[1], k)
            for passage, docid in answers:
                examples.append([val[0], docid, val[1], passage])
    return examples
obj = ElasticObj("passages","_doc",ip ='localhost')
obj.create_index("passages","_doc")
obj.bulk_Index_Data('data/NCPPolicies_context_20200301.csv')
test_examples = read_examples('data/NCPPolicies_test.csv', k=5, es_index="passages", es_ip='localhost')
test=pd.DataFrame(test_examples,columns=['id','docid','query','context'])
test.to_csv('data/test_top5.csv',index=False)
