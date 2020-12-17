@[TOC](paddlepaddle2.0— 开启新梦想之Tensor 2)
大家好这里是一如既往小白的三岁，给大家带来PaddleTensor的第二话。
### AiStudio 地址
[https://aistudio.baidu.com/aistudio/projectdetail/1279000](https://aistudio.baidu.com/aistudio/projectdetail/1279000)

### 三岁白Paddle2.0系列历史地址
三岁白话Paddle2.0系列第一话：起航新征程：[https://aistudio.baidu.com/aistudio/projectdetail/1270135](https://aistudio.baidu.com/aistudio/projectdetail/1270135)

三岁白话paddle2.0系列第二话：开启入门之旅—（Tensor的大家庭）:[https://aistudio.baidu.com/aistudio/projectdetail/1275635](https://aistudio.baidu.com/aistudio/projectdetail/1275635)

CSDN系列文章合集：[https://blog.csdn.net/weixin_45623093/category_10616602.html](https://blog.csdn.net/weixin_45623093/category_10616602.html)

### 参考资料
PaddlePaddle官方文档：

[https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html)

百度：[https://www.baidu.com/s?wd=Tensor](https://www.baidu.com/s?wd=Tensor)


python官方文档：[https://docs.python.org/3/tutorial/introduction.html#strings](https://docs.python.org/3/tutorial/introduction.html#strings)


```python
# 导入环境
import paddle
import numpy as np
```

## Tensor name
上集我们说到了Tensor的place我们来说说我们Tensor独一无二的东西——名字！

这个名字和我们的名字不一样，类似于我们的身份证号码，我们的名字可以重复，但是身份证号码不会，Tensor的name是其唯一的标识符，为python 字符串类型，默认地，在每个Tensor创建时，Paddle会自定义一个独一无二的name

### 获取Paddle Tensor的name
查看一个Tensor的name可以通过Tensor.name属性


```python
print("Tensor name:", paddle.to_tensor(1).name)
```

    Tensor name: generated_tensor_101


# Tensor 的运算与操作
说完了Tensor的属性和定义我们来看看怎么处理Tensor

### Tensor 的 切片与索引
我们可以通过索引或切片方便地访问或修改 Tensor

Paddle 使用标准的 Python 索引规则与 Numpy 索引规则，与 [Indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings)【官方文档中对于索引的使用】类似。

![python官网解析](https://img-blog.csdnimg.cn/2020113016511340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTYyMzA5Mw==,size_16,color_FFFFFF,t_70)

![](https://img-blog.csdnimg.cn/img_convert/6a08b378a35582130b42821023a4e3d7.png)

![](https://img-blog.csdnimg.cn/img_convert/483e8925e79be79b51046b2813f99980.png)



```python
# paddle也一样的操作方式
#建立一个一维Tensor并输出查看结果
rank_1_tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", rank_1_tensor.numpy()) #使用numpy()可以查看结果
```

    Origin Tensor: [0 1 2 3 4 5 6 7 8]


![](https://img-blog.csdnimg.cn/img_convert/1d295b4dcfaab630201b75cf4474fbed.png)



```python
print("First element:", rank_1_tensor[0].numpy())  # 第一个值：0
print("Last element:", rank_1_tensor[-1].numpy())  # 最后一个值：8
print("All element:", rank_1_tensor[:].numpy())  # 全部值0-8
print("Before 3:", rank_1_tensor[:3].numpy())  # 前3个值0-2
print("From 6 to the end:", rank_1_tensor[6:].numpy())  # 6以后的值6-8
print("From 3 to 6:", rank_1_tensor[3:6].numpy())  # 第4到6位 3-5
print("Interval of 3:", rank_1_tensor[::3].numpy())  # 按照3为步长进行输出
print("Reverse:", rank_1_tensor[::-1].numpy())  # 倒叙输出
```

    First element: [0]
    Last element: [8]
    All element: [0 1 2 3 4 5 6 7 8]
    Before 3: [0 1 2]
    From 6 to the end: [6 7 8]
    From 3 to 6: [3 4 5]
    Interval of 3: [0 3 6]
    Reverse: [8 7 6 5 4 3 2 1 0]


### 多维Tensor切片与索引
不同维度之间采用，逗号进行分割

顺序可以参考“剥洋葱”：先高维度再低维度

注：如果没有进行索引则默认为全部


```python
# 建立一个二维Tensor
rank_2_tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print(rank_2_tensor.numpy())
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]



```python
# 获取第二维度中的第一序列值
print("First row:", rank_2_tensor[0].numpy())
# 获取第二维度中的第一序列中所有的值
print("First row:", rank_2_tensor[0, :].numpy())
# 获取第二维度所有序列（第一维度）中的第一个序列值
print("First column:", rank_2_tensor[:, 0].numpy())
# 获取第二维度所有序列（第一维度）中的最后第一个序列值
print("Last column:", rank_2_tensor[:, -1].numpy())
# 获取所有维度中的所有值
print("All element:", rank_2_tensor[:].numpy())
# 获取第二维度中的第一个序列的第二个值
print("First row and second column:", rank_2_tensor[0, 1].numpy())
```

    First row: [0 1 2 3]
    First row: [0 1 2 3]
    First column: [0 4 8]
    Last column: [ 3  7 11]
    All element: [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    First row and second column: [1]


## Tensor 的修改
这是一个需要谨慎的的操作，paddle Tensor修改以后不会保存原有数值，而是原地修改该 Tensor 的数值

如果涉及一些梯度运算等有可能导致后续结果的巨大变化，所以谨慎操作

**谨慎操作**


```python
# 创建一个[2,3]的数值为1的Tensor
x = paddle.to_tensor(np.ones((2, 3)).astype(np.float32))
print(x.numpy(), id(x))
```

    [[1. 1. 1.]
     [1. 1. 1.]] 140156419596848



```python
# 修改二维的数据
x[0] = 0
print(x.numpy(), id(x))
```

    [[0. 0. 0.]
     [1. 1. 1.]] 140156419596848


通过观察可以看到id没有发生改变


```python
x[...] = 3
print(x.numpy(), id(x))
```

    [[3. 3. 3.]
     [3. 3. 3.]] 140156419596848



```python
x[0:1] = np.array([1,2,3]) 
print(x.numpy(), id(x))
```

    [[1. 2. 3.]
     [1. 2. 3.]] 140156419596848


把第二维度的[0:1]修改为 np.array([1,2,3]) ，修改以后id保持不变


```python
x[1] = paddle.ones([3]) 
print(x.numpy(), id(x))
```

    [[1. 2. 3.]
     [1. 1. 1.]] 140156419596848


### Tensor 的其他操作
paddle Tensor 里面封装了许多函数可以供我们参考使用

在这里对比较经典的进行举例



```python
x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]], dtype="float64")  # 新建x，y两个Tensor
y = paddle.to_tensor([[5.5, 6.6], [7.7, 8.8]], dtype="float64")

print(paddle.add(x, y), "\n")  # 使用Paddle API进行操作
print(x.add(y))  # 使用Tensor 类成员函数进行操作
```

    Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=True,
           [[6.60000000, 8.80000000],
            [        11., 13.20000000]]) 
    
    Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=True,
           [[6.60000000, 8.80000000],
            [        11., 13.20000000]])


### 使用Tensor 类成员函数方式对常用函数进行展示


```python
x.abs()                       #逐元素取绝对值
x.ceil()                      #逐元素向上取整
x.floor()                     #逐元素向下取整
x.round()                     #逐元素四舍五入
x.exp()                       #逐元素计算自然常数为底的指数
x.log()                       #逐元素计算x的自然对数
x.reciprocal()                #逐元素求倒数
x.square()                    #逐元素计算平方
x.sqrt()                      #逐元素计算平方根
x.sin()                       #逐元素计算正弦
x.cos()                       #逐元素计算余弦
x.add(y)                      #逐元素相加
x.subtract(y)                 #逐元素相减
x.multiply(y)                 #逐元素相乘
x.divide(y)                   #逐元素相除
x.mod(y)                      #逐元素相除并取余
x.pow(y)                      #逐元素幂运算
x.max()                       #指定维度上元素最大值，默认为全部维度
x.min()                       #指定维度上元素最小值，默认为全部维度
x.prod()                      #指定维度上元素累乘，默认为全部维度
x.sum()                       #指定维度上元素的和，默认为全部维度
```


```python
# 举例
print('sum:', x.sum())  # 得到该Tensor所有的值的和
print('multiply:', x.multiply(y))  # 得到x,y 逐元素相乘的结果
print('abs:', x.abs())  # 逐元素取绝对值
```

    sum: Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
           [11.])
    multiply: Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=True,
           [[ 6.05000000, 14.52000000],
            [25.41000000, 38.72000000]])
    abs: Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=True,
           [[1.10000000, 2.20000000],
            [3.30000000, 4.40000000]])


#### Paddle对python数学运算相关的魔法函数进行了重写，以下操作与上述结果相同。


```python
x + y  -> x.add(y)            #逐元素相加
x - y  -> x.add(-y)           #逐元素相减
x * y  -> x.multiply(y)       #逐元素相乘
x / y  -> x.divide(y)         #逐元素相除
x // y -> x.floor_divide(y)   #逐元素相除并取整
x % y  -> x.remainder(y)      #逐元素相除并取余
x ** y -> x.pow(y)            #逐元素幂运算
```


```python
# 举例
print('add:', x + y)
print('floor_divide:', x % y)
```

    add: Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=True,
           [[6.60000000, 8.80000000],
            [        11., 13.20000000]])
    floor_divide: Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=True,
           [[1.10000000, 2.20000000],
            [3.30000000, 4.40000000]])


### 逻辑运算符


```python
x.isfinite()                  #判断tensor中元素是否是有限的数字，即不包括inf与nan
x.equal_all(y)                #判断两个tensor的全部元素是否相等，并返回shape为[1]的bool Tensor
x.equal(y)                    #判断两个tensor的每个元素是否相等，并返回shape相同的bool Tensor
x.not_equal(y)                #判断两个tensor的每个元素是否不相等
x.less_than(y)                #判断tensor x的元素是否小于tensor y的对应元素
x.less_equal(y)               #判断tensor x的元素是否小于或等于tensor y的对应元素
x.greater_than(y)             #判断tensor x的元素是否大于tensor y的对应元素
x.greater_equal(y)            #判断tensor x的元素是否大于或等于tensor y的对应元素
x.allclose(y)                 #判断tensor x的全部元素是否与tensor y的全部元素接近，并返回shape为[1]的bool Tensor
```

Paddle对python逻辑比较相关的魔法函数进行了重写，以下操作与上述结果相同。


```python
x == y  -> x.equal(y)         #判断两个tensor的每个元素是否相等
x != y  -> x.not_equal(y)     #判断两个tensor的每个元素是否不相等
x < y   -> x.less_than(y)     #判断tensor x的元素是否小于tensor y的对应元素
x <= y  -> x.less_equal(y)    #判断tensor x的元素是否小于或等于tensor y的对应元素
x > y   -> x.greater_than(y)  #判断tensor x的元素是否大于tensor y的对应元素
x >= y  -> x.greater_equal(y) #判断tensor x的元素是否大于或等于tensor y的对应元素
```

以下操作仅针对bool型Tensor


```python
x.logical_and(y)              #对两个bool型tensor逐元素进行逻辑与操作
x.logical_or(y)               #对两个bool型tensor逐元素进行逻辑或操作
x.logical_xor(y)              #对两个bool型tensor逐元素进行逻辑亦或操作
x.logical_not(y)              #对两个bool型tensor逐元素进行逻辑非操作
```


```python
# 有关的线性代数
x.cholesky()                  #矩阵的cholesky分解
x.t()                         #矩阵转置
x.transpose([1, 0])           #交换axis 0 与axis 1的顺序
x.norm('pro')                 #矩阵的Frobenius 范数
x.dist(y, p=2)                #矩阵（x-y）的2范数
x.norm('pro')                 #矩阵的Frobenius 范数
x.dist(y, p=2)                #矩阵（x-y）的2范数
x.matmul(y)                   #矩阵乘法
```

### Tensor的内容到这里就落下帷幕了
Tensor已经暂时落下帷幕。

三岁白话paddle系列还会继续，希望大家多多支持！

