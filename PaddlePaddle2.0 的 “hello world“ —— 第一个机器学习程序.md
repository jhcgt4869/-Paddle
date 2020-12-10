@[TOC](Paddle 界的Hello world)
大家好这里是三岁，给大家带来——
三岁白话paddle系列第五话！
欢迎大家批评指正！！！
### 参考文档
Paddle官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/tutorial/quick_start/hello_paddle/hello_paddle.html#id10](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/tutorial/quick_start/hello_paddle/hello_paddle.html#id10)
AIStudio位置地址：[https://aistudio.baidu.com/aistudio/projectdetail/1308649](https://aistudio.baidu.com/aistudio/projectdetail/1308649)
### CSDN地址
三岁白话系列CSDN：[https://blog.csdn.net/weixin_45623093/category_10616602.html](https://blog.csdn.net/weixin_45623093/category_10616602.html)

paddlepaddle社区号：[https://blog.csdn.net/PaddlePaddle](https://blog.csdn.net/PaddlePaddle)

# 出租车的故事
我们乘坐出租车的时候，会有一个10元的起步价，只要上车就需要收取。出租车每行驶1公里，需要再支付每公里2元的行驶费用。当一个乘客坐完出租车之后，车上的计价器需要算出来该乘客需要支付的乘车费用。

## 普通程序
通过距离得到对应的票价

分别获得1/3/5/9/10/20km的票价


```python
# 方法一：函数法
def calculate_fee(distance_travelled):
    return 10 + 2 * distance_travelled

for x in [1.0, 3.0, 5.0, 9.0, 10.0, 20.0]:
    print(calculate_fee(x))

```

    12.0
    16.0
    20.0
    28.0
    30.0
    50.0



```python
# 方法二：菜鸟法
Kilometers = [1.0, 3.0, 5.0, 9.0, 10.0, 20.0]
for i in Kilometers:
    money = 10 + 2*i
    print(money)
```

    12.0
    16.0
    20.0
    28.0
    30.0
    50.0


### 白话时间
该程序的写的方法方式很多，办法也很多，很多大神们甚至可以写出花来。

但是如果我们只有公里数和应付金额呢？？？

### 迷茫时间
突如其来的难题，有点困惑到了

开始思考怎么样的算法可以写出来！

办法终归有，但是更复杂的程序呢？？？

## 深度学习来帮忙
通过深度学习得到一个近视值，来解答这个困惑说不定很好让我们来一起看看


```python
# 导入paddle和判断版本
import paddle
print("paddle " + paddle.__version__)
```

    paddle 2.0.0-rc0


## 编辑数据


```python
x_data = paddle.to_tensor([[1.], [3.0], [5.0], [9.0], [10.0], [20.0]])
y_data = paddle.to_tensor([[12.], [16.0], [20.0], [28.0], [30.0], [50.0]])
print("x_data = ",x_data)
print("y_data = ",y_data)
```

    x_data =  Tensor(shape=[6, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[1.],
            [3.],
            [5.],
            [9.],
            [10.],
            [20.]])
    y_data =  Tensor(shape=[6, 1], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[12.],
            [16.],
            [20.],
            [28.],
            [30.],
            [50.]])


## 定义模型
根据对问题的解读该类型属于线性函数，类似于一元一次函数

`y_predict = w * x + b`


# paddle.nn.Linear

**线性变换层** [查看API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/nn/layer/common/Linear_cn.html)

对于每个输入Tensor X ，计算公式为：

`Out=XW+b`

其中， W 和 b 分别为权重和偏置。

注：Linear层只接受一个Tensor作为输入，形状为 [batch_size,∗,in_features] ，其中 ∗ 表示可以为任意个额外的维度。


```python
linear = paddle.nn.Linear(in_features=1, out_features=1)  # 定义初始化神经网络
```

### 查看初始化策略
**w** 的值会先进行随机生成

**b** 的值会先以0进行代替


```python
w_before_opt = linear.weight.numpy().item()  # 获取w的值
b_before_opt = linear.bias.numpy().item()  # 获取b的值

print("w before optimize: {}".format(w_before_opt))
print("b before optimize: {}".format(b_before_opt))

```

    w before optimize: 0.7223087549209595
    b before optimize: 0.0


#### 白话时间
多次点击会发现w的值确实每次都是不一样的！！！

这里面的策略可以直接DIY，具体的参考文档

## 优化神经网络

现在的神经网络类似于我们的**无情答卷人**疯狂做题但是有不知道对错，怎么办？？？

现在需要一个损失函数：相当于**改卷人**

需要一个优化策略：相当于**看到错误进行反思然后进行修改的好孩子**

此处使用的：

损失函数：paddle.nn.loss.MSELoss(reduction='mean')   [参考地址](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/nn/layer/loss/MSELoss_cn.html)

优化算法：class paddle.optimizer.SGD(learning_rate=0.001, parameters=None, weight_decay=None, grad_clip=None, name=None) [参考地址](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/optimizer/sgd/SGD_cn.html)


```python
mse_loss = paddle.nn.MSELoss()  
sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters = linear.parameters())
```

# 优化算法


```python
total_epoch = 7000  # 运行轮数
for i in range(total_epoch):
    y_predict = linear(x_data)
    loss = mse_loss(y_predict, y_data)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_grad()

    if i%1000 == 0:  # 每1000轮输出一次
        print("epoch {} loss {}".format(i, loss.numpy()))

print("finished training， loss {}".format(loss.numpy()))

```

    epoch 0 loss [1.5586754e-07]
    epoch 1000 loss [1.5586754e-07]
    epoch 2000 loss [1.5586754e-07]
    epoch 3000 loss [1.5586754e-07]
    epoch 4000 loss [1.5586754e-07]
    epoch 5000 loss [1.5586754e-07]
    epoch 6000 loss [1.5586754e-07]
    finished training， loss [1.5586754e-07]


## 白话时间
根据测试该程序在7000轮的时候结果就bao报错不变了

所以在7000轮时效果最好

### 查看结果



```python
w_after_opt = linear.weight.numpy().item()
b_after_opt = linear.bias.numpy().item()

print("w after optimize: {}".format(w_after_opt))
print("b after optimize: {}".format(b_after_opt))

```

    w after optimize: 2.0000507831573486
    b after optimize: 9.999356269836426


### 白话
经过查看这个结果已经和我们要的结果所差无几

所以这个效果还是比较好的

# Hello PalldPalld 2.0 !


```python
print("Hello PalldPalld 2.0 !")
```

    Hello PalldPalld 2.0 !


今天的内容就到这里啦，希望喜欢的朋友关注点赞留言，多多关注我们的paddlepaddle，生态需要你的努力，加油！
