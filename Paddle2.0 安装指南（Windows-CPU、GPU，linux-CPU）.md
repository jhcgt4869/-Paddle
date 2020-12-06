@[TOC](白话Paddle2.0-rc — 起航新征程)

## paddlepaddle一场新的飞跃！

自paddlepaddle 1.8后革新版本的2.0-rc已经发布了，新的版本新的飞跃，让我们一起脚踏飞桨的祥云，遨游代码的海洋，打开深度学习的大门，共同探索美好新未来



```python
# 查看当前安装的版本
import paddle
print("paddle " + paddle.__version__)
```

    paddle 2.0.0-rc0


## 让我们一起开启新世界的大门吧~~~
本文的AIstudio配套环境地址：[https://aistudio.baidu.com/aistudio/projectdetail/1270135](https://aistudio.baidu.com/aistudio/projectdetail/1270135)
### 等待，这里需要等待一下掉队的童鞋！
基于没有安装paddlepaddle的童鞋们我就给大家来排雷

#### 参考文档
百度paddlepaddle官方安装教程：[https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-windows-conda](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-windows-conda)

小鸭学院paddlepaddle安装错误处理教程：[https://blog.csdn.net/weixin_41450123/article/details/109701779](https://blog.csdn.net/weixin_41450123/article/details/109701779)

### 使用CPU版本的小伙伴朝这里看了
* 点开我们的官方文档，选择对应的版本。查看有关的代码进行安装，即可。

![](https://img-blog.csdnimg.cn/img_convert/b8f6ff6a9138ac971448e270d95d1af1.png)





```python
!python -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple  # 在cmd中运行时把!去掉即可
```

让我们看看实际的运行效果吧！

![](https://img-blog.csdnimg.cn/img_convert/724987bb064c3b44227ea8e9de56c6e2.png)

![](https://img-blog.csdnimg.cn/img_convert/bd964270fe414d554327a9a43b3cb989.png)


### 查看是否安装成功！
![](https://img-blog.csdnimg.cn/img_convert/55dbf3980edd1bf95e0f081d2bba0018.png)



```python
import paddle.fluid as fluid
fluid.install_check.run_check()
# 只要出现"Your Paddle Fluid is installed succesfully!"就是安装成功！
```

    Running Verify Fluid Program ... 
    Your Paddle Fluid works well on SINGLE GPU or CPU.
    Your Paddle Fluid works well on MUTIPLE GPU or CPU.
    Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now


![](https://img-blog.csdnimg.cn/img_convert/92af7e150b5a7cefb1582723b1f4e164.png)


### CPU版本安装注意事项
* 电脑有多个python环境，当前环境不是预期环境

解决：查看当前python环境，检查是否符合预期
* python低于python2.7.15，不支持paddlepaddle2.0.0-rc0（不建议python2安装paddlepaddle2.0-rc0）

解决：使用3.5.+及以上版本进行处理
* pip的版本太低导致paddle安装报错

解决：使用'python3 -m pip install --upgrade pip'等代码进行升级
* 安装网络异常，未成功安装

解决：重新进行安装

## GPU的小伙伴我们走啦走啦！
GPU所需要的环境和内容比较多，较为复杂但是不怕我们一步一步完全看啦！

老规矩让我们看看需要什么啦！
![](https://img-blog.csdnimg.cn/img_convert/251d6b297f959fa5e3ce3945e72b1676.png)

注意：此处需要根据实际情况对conda进行确认。

### 环境的准备


| 项目 | 要求 | 
| -------- | -------- |
| Windows    | 7/8/10 专业版/企业版 (64bit)     |
| CUDA   |9.0/9.1/9.2/10.0/10.1/10.2，且仅支持单卡|
| conda  | 4.8.3+ (64 bit)|

* conda的安装

地址：[Anaconda官网](https://www.anaconda.com/)

![](https://img-blog.csdnimg.cn/img_convert/7c44681549e18512abf000d9caef8183.png)

根据自己的需要和实际情况对anaconda进行下载或安装（建议安装最新版本）

### 虚拟环境的创建
大家对虚拟环境一定不陌生，都是老朋友了，那那那新手怎么办？

茫茫然？

小白带你白话创环境！（下面创建的是Anaconda虚拟环境如果不是conda环境请自行搜索）

* 1.pytho虚拟环境创建

paddlepaddle2.0目前暂时支持python2.7.15以上版本和3.5.+、3.6.+、3.7.+、3.8.+

（此处**不建议**使用2.7版本，paddlepaddle2.0将不再支持python2系列）

```
conda create -n <自定义名称> python=3.x (自己实际需要的版本)
```
**这里三岁自定义的名字是paddlepaddle2.0-re**

![](https://img-blog.csdnimg.cn/img_convert/85c0f15d15044bbe7d437960ce3dfa62.png)
* 2.查看自己的创建的虚拟环境及版本（先找到位置，这里三岁找了好久没有比较好的办法，还是看你们自定义安装的位置这里就不展示了）

`python --version`

![](https://img-blog.csdnimg.cn/img_convert/65b60e21b76e15cffe57af962622ac5b.png)
* 3.确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构

`python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"`

![](https://img-blog.csdnimg.cn/img_convert/25312f723c30f002c3d910a0f38e5f5d.png)


### 开始安装paddlepaddle2.0-rc GPU版本
* 1.进入指定环境

`activate xxx`xxx == 自己的环境名字

![](https://img-blog.csdnimg.cn/img_convert/cbb055d8c31fc418fb3f3553fe18cbaf.png)

** 在前面的括号里面出现了自己的环境名称就可以！**

* 2.开始安装
![](https://img-blog.csdnimg.cn/img_convert/456bd7d0ff047f56c992fcce40442230.png)

这里就需要大家根据自己的实际情况进行进行处理了不同的coda使用不同的版本。


### 安装结束后确认
这里的确认方式与CPU相同

![](https://img-blog.csdnimg.cn/img_convert/5769dbd8b442a3d910e0b0272f4d9f8c.png)

但是这里的问题可能比较多

emmm，看看解决方案吧！

### GPU安装容易出现的问题
* pip、python等版本不对

解决：升级pip、使用合适的版本进行安装
* conda 环境多导致版本选择错误

解决：查找conda中的虚拟环境，在正确的虚拟环境中进行安装

* CUDA不符合，显卡不匹配或不适用

解决：查看显卡是否是英伟达的然后CUDA版本是否正确如果不正确请，更换硬件或升级有关软件

* 其他花式报错：

解决方法：参考链接：[https://blog.csdn.net/weixin_41450123/article/details/109701779](https://blog.csdn.net/weixin_41450123/article/details/109701779)

## linux CPU的小伙伴我们启程啦
由于三岁只有阿里云，无法使用GPU版本，此处只做CPU的教程啦！

还是老规矩官网找到我们安装代码！

![](https://img-blog.csdnimg.cn/img_convert/2b8814f40e76789abd7c6296db6c73a3.png)



```python
python -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple
```

* 此处注意LINUX里面都是有python2作为底层内容的需要确保使用的合适的版本所以需要根据实际情况进行处理

![](https://img-blog.csdnimg.cn/img_convert/c3a9c2c3e2dc550cb920611f8fb7a226.png)

此处三岁把python改成了python3进行安装




### 爱情来得就是这么猝不及防
![](https://img-blog.csdnimg.cn/img_convert/4ced06a5864a081e139e26a7c32e5a82.png)

安装之间发现黄色的错误警报信息，顿时让一个linux慌了手脚。

一顿操作猛如虎，发现是path没有填写完整，进行改正发现已经问题不大

![](https://img-blog.csdnimg.cn/img_convert/401c1456d0346473a63ae60819b9eb39.png)



### 确认安装成功
![](https://img-blog.csdnimg.cn/img_convert/2e6756a286037726c560458c9b2e27fa.png)


### linux涉及的问题
* python安装的问题，linux涉及python3和Python2之类的共同问题

解决方案：合理安装python，升级pip对，path进行必要的更改
* 网络的问题：有些linux在虚拟机上或者在服务器端，远程操作，可能涉及网络不好导致无法安装

解决方案：处理好网络接口等各方面问题然后进行再次安装即可。
