# Lab01 PCA Algorithm

---

[toc]

Student ID: 19011418

Author: Sonqing Zhao, Minzu University of China

Written at Nov 11^th^, 2021

> 

---

## Lab Purpose

Master the principle of PCA algorithm

## Lab Requirements

1. 理解数据降维过程
2. 熟练使用Python或其他工具实现PCA算法

## Lab Equipment

1. A computer
2. Python

## Lab Procedure

### Explain the PCA dimensionality reduction process



### Data set

可以使用sklearn库中的iris数据集，其中每个样本有4个特征参数，分别为花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性。（数据集不限）。

数据集选取为：

加载数据集代码：

```python

```

### 数据标准化

数据标准化（去均值）代码：

```python

```

标准化后的数据为：

### 求协方差矩阵

求协方差矩阵代码：

协方差矩阵结果为：

### 求特征值和特征向量

求特征值和特征向量代码：

特征值及特征向量结果为：

### 将数据降到k维

将数据降到k维（k的值可以依据原数据集选取，如果使用iris数据集，k可以取值为2），按特征值大小排序，选取前k个特征值对应的特征向量，计算降维后的数据为：

## Lab Result



## Improvement and innovation

 