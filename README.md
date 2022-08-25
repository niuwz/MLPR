# MLPR
数据集下载链接 https://openml.org/search?type=data&status=active&id=554
数据集处理部分在文件"PCA降维.py"中，最终生成一个csv文件，作为后面训练的数据。
"classification.py"中包含了本问题中所使用到的各种算法，均基于numpy实现。
"Dateset.py"中包含了数据集类，为本问题中各算法通用的数据集类
初步测试中对比了三个分类器算法，最终选择了KNN和MLP算法解决本问题