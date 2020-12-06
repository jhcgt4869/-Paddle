@[TOC]( Paddle 白话第四篇—— 广播（broadcasting）

大家好这里是三岁，我们一起来看看paddle白话第四篇吧！！！

PaddlePaddle和其他框架一样，提供的一些API支持广播(broadcasting)机制，允许在一些运算时使用不同形状的张量。

## 广播（broadcasting）
解释：当两个数组的形状并不相同的时候，我们可以通过扩展数组的方法来实现相加、相减、相乘等操作
Paddle在广播中与Numpy的广播机制类似，但是又有革新，让我们一起来看看吧！

#### 参考资料
Paddle官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/guides/01_paddle2.0_introduction/basic_concept/broadcasting_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/guides/01_paddle2.0_introduction/basic_concept/broadcasting_cn.html)

Numpy官网-广播机制：[https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html#module-numpy.doc.broadcasting)

#### CSDN地址
三岁白话系列CSDN地址：
[https://blog.csdn.net/weixin_45623093/category_10616602.html](https://blog.csdn.net/weixin_45623093/category_10616602.html)

paddlepaddleCSDN系列文章：[https://blog.csdn.net/PaddlePaddle](https://blog.csdn.net/PaddlePaddle)

Ai Studio地址：[https://aistudio.baidu.com/aistudio/projectdetail/1305898](https://aistudio.baidu.com/aistudio/projectdetail/1305898)
```python
# 导入第三方库
import paddle
import numpy as np
```

### 同维度的操作


```python
# 同维度的操作
x = paddle.to_tensor(np.ones((2, 3, 4), np.float32))
y = paddle.to_tensor(np.ones((2, 3, 4), np.float32))
print("x=", x)
print("y=", y)
print('x+y=', x+y)  # 逐元素相加
print('x add y=', paddle.add(x, y))  # add API
```

    x= Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]],
    
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
    y= Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]],
    
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
    x+y= Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[[2., 2., 2., 2.],
             [2., 2., 2., 2.],
             [2., 2., 2., 2.]],
    
            [[2., 2., 2., 2.],
             [2., 2., 2., 2.],
             [2., 2., 2., 2.]]])
    x add y= Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[[2., 2., 2., 2.],
             [2., 2., 2., 2.],
             [2., 2., 2., 2.]],
    
            [[2., 2., 2., 2.],
             [2., 2., 2., 2.],
             [2., 2., 2., 2.]]])


## 广播规则
**为了进行广播，操作中两个阵列的尾轴尺寸必须相同，或者其中之一必须相同。**

解析：

如果两个数组的后缘维度（trailing dimension，即从末尾开始算起的维度）的轴长度相符，或其中的一方的长度为1，则认为它们是广播兼容的。广播会在缺失和（或）长度为1的维度上进行。

  这句话乃是理解广播的核心。广播主要发生在两种情况，一种是两个数组的维数不相等，但是它们的后缘维度的轴长相符，另外一种是有一方的长度为1。

## 类型1：数组维度不同，后缘维度的轴长相符

## 多维和一维广播操作
多维和一维的操作要准守广播规则


```python
x = paddle.to_tensor([1.0, 2.0, 3.0])
y = paddle.to_tensor(2.0)
print('x+y=', x+y)
```

    x+y= Tensor(shape=[3], dtype=float32, place=CPUPlace, stop_gradient=True,
           [3., 4., 5.])


#### 原理
![](https://img-blog.csdnimg.cn/img_convert/f409ea2351e4934f7cd875b197ded674.gif)



```python
x = paddle.to_tensor([[0, 0, 0],[1, 1, 1],[2, 2, 2], [3, 3, 3]])
y = paddle.to_tensor([1, 2, 3])
print('x+y', x+y)
```

    x+y Tensor(shape=[4, 3], dtype=int64, place=CPUPlace, stop_gradient=True,
           [[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]])


#### 原理：
![](https://img-blog.csdnimg.cn/img_convert/aae5cbc1c0e90db558206bdbdc21e76e.png)



### 错误示范：



```python
x = paddle.to_tensor([[ 0.0,  0.0,  0.0],
...            [10.0, 10.0, 10.0],
...            [20.0, 20.0, 20.0],
...            [30.0, 30.0, 30.0]]) # 大小：4*3
y = paddle.to_tensor([0, 1, 2, 3])  # 大小：4
print(x+y)
```

#### 问题解析
报错情况：

```
EnforceNotMet: 
----------------------
Error Message Summary:
----------------------
InvalidArgumentError: Broadcast dimension mismatch. Operands could not be broadcast together with the shape of X = [4, 3] and the shape of Y = [4]. Received [3] in X is not equal to [4] in Y at i:1.
  [Hint: Expected x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 || y_dims_array[i] <= 1 == true, but received x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 || y_dims_array[i] <= 1:0 != true:1.] (at /paddle/paddle/fluid/operators/elementwise/elementwise_op_function.h:160)
  [operator < elementwise_add > error]
```

### 问题
没有符合广播原则，最后的维度没有对齐
### 原理图:
![](https://img-blog.csdnimg.cn/img_convert/47b27eb46b4e4dc0d691a3a4d219d7fe.gif)


### 多维度和多维度之间的广播



```python
x = paddle.to_tensor([[[0, 1], [2, 3], [4, 5], [6,7]],
                    [[0, 1], [2, 3], [4, 5], [6,7]],
                    [[0, 1], [2, 3], [4, 5], [6,7]]])
print('x的形状', x.shape)
y = paddle.to_tensor([[0, 1], [2, 3], [4, 5], [6,7]])
print('y的形状', y.shape)
print('x+y', x+y)
```

    x的形状 [3, 4, 2]
    y的形状 [4, 2]
    x+y Tensor(shape=[3, 4, 2], dtype=int64, place=CPUPlace, stop_gradient=True,
           [[[0, 2],
             [4, 6],
             [ 8, 10],
             [12, 14]],
    
            [[0, 2],
             [4, 6],
             [ 8, 10],
             [12, 14]],
    
            [[0, 2],
             [4, 6],
             [ 8, 10],
             [12, 14]]])


### 原理
![](https://img-blog.csdnimg.cn/img_convert/572815d8fba575fe377838e075adec61.png)


## 类型二：数组维度相同，后缘维度的轴长不相同


```python
x = paddle.to_tensor(([[0.0], [10.0], [20.0], [30.0]]))
y = paddle.to_tensor([0.0, 1.0, 2.0])
print('x+y', x+y)
```

    x+y Tensor(shape=[4, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[0., 1., 2.],
            [10., 11., 12.],
            [20., 21., 22.],
            [30., 31., 32.]])


### 原理：
![](https://img-blog.csdnimg.cn/img_convert/12ee40859749ae7bdde7aa18d082b619.gif)


## 特殊情况


```python
x = paddle.to_tensor(np.ones((3, 5, 6), np.float32))
y = paddle.to_tensor(np.ones((1, 6), np.float32))
print(x+y)
```

    Tensor(shape=[3, 5, 6], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[[2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.]],
    
            [[2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.]],
    
            [[2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.],
             [2., 2., 2., 2., 2., 2.]]])


### 解析：
这个地方虽然数组维度不相同，后缘维度的轴长也不相同

但是这里后缘维度不同的地方是1，也可以先把1进行处理

让我们举例说明


```python
x = paddle.to_tensor([[[2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.]],

        [[2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.]],

        [[2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.],
         [2., 3., 4., 5., 6., 7.]]])

y = paddle.to_tensor([[5, 4, 3, 2, 1, 10]])

y = paddle.to_tensor([[5, 4, 3, 2, 1, 10]])
print(x+y)
```

    Tensor(shape=[3, 5, 6], dtype=float32, place=CPUPlace, stop_gradient=True,
           [[[ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.]],
    
            [[ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.]],
    
            [[ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.],
             [ 7.,  7.,  7.,  7.,  7., 17.]]])


看到这里都明白了什么吧！不懂也不解释了！！！

### 特殊说明
广播提供了一种对数组操作进行矢量化的方法，从而使循环在C而不是Python中发生。这样做无需复制不必要的数据，通常可以实现高效的算法实现。在某些情况下，广播不是一个好主意，因为广播会导致内存使用效率低下，从而减慢计算速度。

看情况量力而行 啦！
