# [三岁白话系列]PaddlePaddle2.0——手写数字识别
三岁白话系列第6话
### AI Studio地址：
[https://aistudio.baidu.com/aistudio/projectdetail/1324628](https://aistudio.baidu.com/aistudio/projectdetail/1324628)
### 参考文档：
Paddle官网：[https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/tutorial/quick_start/getting_started/getting_started.html#id3](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/tutorial/quick_start/getting_started/getting_started.html#id3)

paddle API查看地址：[https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/vision/datasets/mnist/MNIST_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/vision/datasets/mnist/MNIST_cn.html)

MNIST书写数字识别官网：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### CSDN地址
三岁白话系列CSDN：[https://blog.csdn.net/weixin_45623093/category_10616602.html](https://blog.csdn.net/weixin_45623093/category_10616602.html)

paddlepaddle社区号：[https://blog.csdn.net/PaddlePaddle](https://blog.csdn.net/PaddlePaddle)


```python
# 导入paddle，查看版本型号
import paddle
print(paddle.__version__)
```

    2.0.0-rc0


## 手写数字识别数据集
MNIST是手写识别数据集实例

其中训练集为60,000个示例，而测试集为10,000个示例



## 把训练集转换成图片
我们可以查看数据集里面的数据来进一步了解情况。

通过数据的下载和解压在对数据进行处理就得到了我们的`./mnist_train`文件夹

里面解压了0-9的文件可以进行查看

![](https://img-blog.csdnimg.cn/img_convert/683ca8de0eb2b0e2240b7c3a902be84f.png)

图像为3维，标签为1维。

图像大小为：28pix*28pix


```python
# 导入第三方库
import numpy as np
import struct
from PIL import Image
import os

# 分别对train-images.idx3-ubyte和train-labels.idx1-ubyte进行处理最后得到png的图片格式

data_file = './train-images.idx3-ubyte'
# It's 47040016B, but we should set to 47040000B
data_file_size = 47040016
data_file_size = str(data_file_size - 16) + 'B'
 
data_buf = open(data_file, 'rb').read()
 
magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)
 
label_file = './train-labels.idx1-ubyte'
 
# It's 60008B, but we should set to 60000B
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'
 
label_buf = open(label_file, 'rb').read()
 
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)
 
datas_root = 'mnist_train'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)
 
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
 
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + \
        'mnist_train_' + str(ii) + '.png'
    img.save(file_name)
```

## 查看手写图片示例!
![](https://img-blog.csdnimg.cn/img_convert/838fd12f9a79884554758e74e6969c03.png)
![](https://img-blog.csdnimg.cn/img_convert/ce0911fea5f1f8e36a3955d0101feb3d.png)
![](https://img-blog.csdnimg.cn/img_convert/083902bdaa9cdcc5be0a19177cbeddcc.png)
![](https://img-blog.csdnimg.cn/img_convert/9d656253221a9623e7a597346489f56b.png)
![](https://img-blog.csdnimg.cn/img_convert/477b6cb35f2413b6160d0770ab13bb9c.png)
![](https://img-blog.csdnimg.cn/img_convert/0886c22f7c954603e4d5eafa8738bd93.png)
![](https://img-blog.csdnimg.cn/img_convert/4af4b5ded7bc00eadf64ce972eff675f.png)
![](https://img-blog.csdnimg.cn/img_convert/cd71fc05737ea32e88bc9dd67915085f.png)
![](https://img-blog.csdnimg.cn/img_convert/725d83b03dcc439b09c1777e709539bc.png)
![](https://img-blog.csdnimg.cn/img_convert/499291ac9304d81eda670fe91a114941.png)

## PaddlePaddle2.0 手写数据识别API
`paddle.vision.datasets.MNIST`

通过特定的参数获得对应的数据

image_path (str) - 图像文件路径，如果 download 设置为 True ，此参数可以设置为None。默认值为None。

label_path (str) - 标签文件路径，如果 download 设置为 True ，此参数可以设置为None。默认值为None。

chw_format (bool) - 若为 True 输出形状为[1, 28, 28], 否则为 [1, 784]。默认值为 True 。

mode (str) - 'train' 或 'test' 模式，默认为 'train' 。

download (bool) - 是否自定下载数据集文件。默认为 True 。


```python
train_dataset = paddle.vision.datasets.MNIST(mode='train')  # 测试集数据
val_dataset =  paddle.vision.datasets.MNIST(mode='test')  # 训练集数据
```

    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz 
    Begin to download
    
    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz 
    Begin to download
    ........
    Download finished


## 模型搭建
通过`paddle.nn.Sequential`进行模型的搭建,把需要的数据都传进去。

##### paddle.nn.Sequential(*layers)
顺序容器。子Layer将按构造函数参数的顺序添加到此容器中。传递给构造函数的参数可以Layers或可迭代的name Layer元组。

layers (tuple) - Layers或可迭代的name Layer对。

##### paddle.nn.Linear(in_features, out_features, weight_attr=None, bias_attr=None, name=None)
线性变换层 。对于每个输入Tensor X ，计算公式为：`Out=XW+b`

其中， W 和 b 分别为权重和偏置。

in_features (int) – 线性变换层输入单元的数目。

out_features (int) – 线性变换层输出单元的数目。

weight_attr (ParamAttr, 可选) – 指定权重参数的属性。默认值为None，表示使用默认的权重参数属性，将权重参数初始化为0。具体用法请参见 ParamAttr 。

bias_attr (ParamAttr|bool, 可选) – 指定偏置参数的属性。 `bias_attr `为bool类型且设置为False时，表示不会为该层添加偏置。 `bias_attr` 如果设置为True或者None，则表示使用默认的偏置参数属性，将偏置参数初始化为0。具体用法请参见 [ParamAttr](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/fluid/param_attr/ParamAttr_cn.html#cn-api-fluid-paramattr) 。默认值为None。

name (str，可选) – 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api_guides/low_level/program.html#api-guide-name) ，一般无需设置，默认值为None。

##### class paddle.nn.ReLU(name=None)
ReLU激活层（Rectified Linear Unit）。计算公式如下：`ReLU(x)=max(0,x)`
	
其中，x 为输入的 Tensor

**形状:**

input: 任意形状的Tensor。

output: 和input具有相同形状的Tensor。

##### paddle.nn.Dropout(p=0.5, axis=None, mode="upscale_in_train”, name=None)
Dropout是一种正则化手段，该算子根据给定的丢弃概率 p ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。

p (float): 将输入节点置为0的概率， 即丢弃概率。默认: 0.5。

axis (int|list): 指定对输入 Tensor 进行Dropout操作的轴。默认: None。

mode (str): 丢弃单元的方式，有两种'upscale_in_train'和'downscale_in_infer'，默认: 'upscale_in_train'。


```python
mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),  # 输入线性变换层数目为784个，输出为512个
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),  # 丢弃概率为0.2
    paddle.nn.Linear(512, 10)  # 输入线性变换层数目为512个，输出为10个
)
```

## 启用fit接口来开启我们的模型训练

##### paddle.Model()
`Model` 对象是一个具备训练、测试、推理的神经网络。该对象同时支持静态图和动态图模式，通过 `paddle.disable_static()` 来切换。需要注意的是，该开关需要在实例化 `Model` 对象之前使用。输入需要使用 `paddle.static.InputSpec` 来定义。

##### paddle.optimizer.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=None, weight_decay=None, grad_clip=None, name=None, lazy_mode=False)

能够利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。

[具体参考](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/optimizer/adam/Adam_cn.html)

##### paddle.nn.loss.CrossEntropyLoss(weight=None, ignore_index=- 100, reduction='mean')
该OP计算输入input和标签label间的交叉熵损失 ，它结合了 LogSoftmax 和 NLLLoss 的OP计算，可用于训练一个 n 类分类器。

[具体参考](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc/api/paddle/nn/layer/loss/CrossEntropyLoss_cn.html)
##### paddle.metric.Accuracy
计算准确率(accuracy)

topk (int|tuple(int)) - 计算准确率的top个数，默认是1。

name (str, optional) - metric实例的名字，默认是'acc'。


```python
# 预计模型结构生成模型实例，便于进行后续的配置、训练和验证
model = paddle.Model(mnist)

# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(parameters=mnist.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 开始模型训练
model.fit(train_dataset,
          epochs=5,
          batch_size=32,
          verbose=1)
t,
          epochs=5,
          batch_size=32,
          verbose=1)

```

    Epoch 1/5
    step   80/1875 [>.............................] - loss: 6.3583 - acc: 0.7328 - ETA: 8s - 5ms/step

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and


    step  140/1875 [=>............................] - loss: 3.2578 - acc: 0.7797 - ETA: 6s - 4ms/stepstep 1875/1875 [==============================] - loss: 0.8694 - acc: 0.8769 - 3ms/step         
    Epoch 2/5
    step 1875/1875 [==============================] - loss: 0.2539 - acc: 0.9058 - 3ms/step         
    Epoch 3/5
    step 1875/1875 [==============================] - loss: 0.3372 - acc: 0.9103 - 3ms/step         
    Epoch 4/5
    step 1875/1875 [==============================] - loss: 0.1992 - acc: 0.9166 - 3ms/step         
    Epoch 5/5
    step 1875/1875 [==============================] - loss: 0.3574 - acc: 0.9181 - 3ms/step         


### 测试结果
通过loss和acc的结果来评断模型训练的质量和效率


```python
model.evaluate(val_dataset, verbose=0)
```




    {'loss': [3.576278e-07], 'acc': 0.9262}



今天的白话就到这里了，我们下次再见啦！！！

这里是三岁，小白三岁，传说中的Paddle最菜程序员[狗头]
感谢大家的关注，希望能够一键4连支持一下！！！
