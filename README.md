# COVID19_QA_baseline
DataFountain疫情问答助手keras-bert实现，单模0.67，比赛链接https://www.datafountain.cn/competitions/424/submits?view=submit-records
## 数据预处理
* 运行elastic.py，利用Elastic Search导入文档索引，选出测试集中每个问题最相关的五个文档，输出为test_top5.csv  
* 运行process.py，bert最大序列长度为512，答案通常不超过两句话，因此利用动态规划划分文本，子文本之间尽量少重复，详情请见split_text函数。对训练集测试集进行文本分割，输出为train_new.csv和test_new.csv。
## 训练、验证、测试
* 运行base.py，利用roberta-wwm-ext预训练模型，多任务调和：分别预测出答案是否在该子文本中、答案开始的位置、答案结束的位置，计算每个子文本的得分，得分公式如下：
![image](https://github.com/LHT-Curry/COVID19_QA_baseline/blob/master/score.png)
