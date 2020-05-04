# COVID19_QA_baseline
DataFountain疫情问答助手keras-bert实现，单模0.67，比赛链接https://www.datafountain.cn/competitions/424/submits?view=submit-records
## 数据预处理
* 运行elastic.py，利用Elastic Search导入文档索引，选出测试集中每个问题最相关的五个文档，输出为test_top5.csv  
* 运行process.py，bert最大序列长度为512，因此要对文本进行划分。答案通常在一个句子内部或者一个或多个完整的句子，通常不超过两句话，因此以句子为单位进行划分，保证覆盖全文以及段落长度限制条件的约束下，使分段结果具有最小的重复并且最小化丢失答案信息的风险，通过构建有向无环图进行规划，详情请见split_text函数。对训练集测试集进行文本分割，输出为train_new.csv和test_new.csv。
## 训练、验证、测试
* 运行base.py，利用roberta-wwm-ext预训练模型，模型下载地址：https://github.com/ymcui/Chinese-BERT-wwm  
* 多任务调和：分别预测出答案是否在该子文本中、答案开始的位置、答案结束的位置，五折交叉验证，计算每个子文本的得分，得分公式如下：  
![image](https://github.com/LHT-Curry/COVID19_QA_baseline/blob/master/score.png)  
其中w1、w2分别设置为0.9、0.1（在验证集中分数最高）  
* 选择分数最高的子文本，预测其答案，选择概率最高的三个答案开始位置和结束位置进行排列组合，满足结束位置>开始位置、answer_score最大的为最终位置，输出为submit.csv提交文件，模型权重保存至model_save文件夹，log文件夹为训练数据记录  
## 作者
511067474@qq.com
