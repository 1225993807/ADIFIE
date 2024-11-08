# ADIFIE
**针对不完备数据的模糊信息熵理论及其异常检测**

**摘要**
粒计算理论能模仿人类处理大规模复杂问题时的思维方式，并已在异常检测领域得到了成功运用，但大多数基于粒计算的异常检测算法都无法处理不完备数据。针对这一问题，提出了一种面向不完备数据的模糊信息熵理论，并将其应用于不完备异常检测算法。首先，定义了不完备数据集下的模糊集与模糊相似关系。其次，提出了不完备数据集下的模糊熵理论及其相关度量，包括不完备模糊信息熵、不完备模糊联合熵、不完备模糊条件熵、不完备模糊互信息、不完备模糊补熵，并提出了不完备熵比率因子和不完备势差两种指标。接着，在所提理论基础上构建了异常检测模型。首先，通过不完备信息系统的模糊相似关系构建关系矩阵。接着，计算出指定属性集下的不完备熵比率因子和不完备势差。然后，构建出样本在特定属性集下的异常因子并计算出异常分数来衡量样本的异常程度。最后，设计了相应的异常检测算法ADIFIE（Anomaly Detection for Incomplete datasets based on Fuzzy Information Entropy）并在公开的UCI数据集以及仿真数据集上进行了实验，将所提算法与主流异常检测算法进行对比。结果表明，ADIFIE算法有着最优的实验效果，且在3/4的数据集中都取得了最好的效果。

使用
直接运行ADIFIE.py文件即可得到demo数据的结果 demo数据如下：

```
trandata = [["A", 6  , 0.1],
            ["C", "*", 0.9],
            ["D", 1  , 0.7],
            ["*", 9  , 0.3],
            ["B", 5  , "*"], 
            ["A", 3  , 0.5]])
```

结果如下：
```out_scores=
            0.62718698
            0.71977797
            0.87126917
            0.89652969
            0.60115125
            0.49254036
```
            
相关数据集位于https://github.com/BElloney/Outlier-detection
