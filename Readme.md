# ubuntu 20.04 下配置StegaStamp运行环境
本文主要介绍如何在ubuntu 20.04 系统下安装StegaStamp需要的运行环境，`以下操作均在sudo提权后进行`。

## 前置步骤
确保系统的软件列表得到更新，并且其余软件已更新。
```
# apt-get update -y
# apt-get upgrade -y
```
这里有个有趣的问题 `apt`和 `apt-get`有什么区别？感兴趣可以阅读：https://blog.csdn.net/liudsl/article/details/79200134
## 1. 安装`nvidia`显卡驱动<sup>[1]<sup>
### 1.1 检查ubuntu自带的驱动是否启动
```
# lsmod | grep nouveau
```
如果输入该指令后什么都没出现证明 `nouveau` 已经被禁用。否则需要手动将该驱动程序添加进入 `blacklist.conf` .
#### 1.1.1 修改 blacklist.conf 
```
# vi /etc/modprobe.d/blacklist-nouveau.conf
```
在打开的文件内按 `：` 进入 `insert模式` 并输入以下内容：
```
blacklist nouveau
options nouveau modeset=0
```
按 `ESC` 再按 `:wq` + Enter 进行保存。
#### 1.1.2 更新 initramfs<sup>[2]</sup>
```
# update-initramfs -u
```
#### 1.1.3 重启
```
# reboot
```
### 1.2 安装驱动
#### 1.2.1 查询显卡驱动版本号
查询显卡型号：
```
# lspci | grep VGA
```
再去官网查看显卡型号对应的驱动版本，官网链接：https://www.nvidia.com/Download/index.aspx?lang=en

#### 1.2.2 安装驱动程序
```
# apt -y install nvidia-driver-440
```
其中 `440` 是你查询到的版本号，你也可以从官网下载后安装。

#### 1.2.3 验证安装是否成功
```
# nvidia-smi
```

## 2. 安装 `cuda 10.1` <sup>[3]<sup>
ubuntu 20.04的官方库里自带有 `cuda 10.1` 的包。
```
# apt -y install nvidia-cuda-toolkit
```

## 3. 安装 `cuDNN 7`<sup>[4]<sup>
首先打开官网下载页并`登陆`。<br>
在列表中建议选择 
```
Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1
```
因为系统默认库中的`cuda 10.1`对应的是`cuDNN 7`<br>
根据需要选择下载：
```
cuDNN Runtime Library for Ubuntu18.04 (Deb)

cuDNN Developer Library for Ubuntu18.04 (Deb)

cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)
```
下载完打开安装即可。

## 4. 安装 `StegaStamp` 的依赖包
### 4.1 安装 `git` 工具
```
# apt install git
```

### 4.2 安装 `pip3` 工具
```
# apt-get install python3-pip
```

### 4.3 clone 项目到本地
```
# git clone --recurse-submodules https://github.com/tancik/StegaStamp.git
```

### 4.4 创建虚拟环境<sup>4<sup>
```
# python3 -m venv stegastamp-env
```

### 4.5 激活虚拟环境<sup>4<sup>
```
# source stegastamp-env/bin/activate
```

### 4.6 在虚拟环境下安装依赖包
`cd` 到项目目录下
```
(stegastamp-env) # cd Stegastamp
```
使用 `pip3` 指令安装依赖
```
(stegastamp-env) [Stegastamp] # pip3 install -r requirements.txt
```

## 5. 安装`tensorflow`
```
(stegastamp-env) [Stegastamp] # pip3 install tensorflow-gpu
```
## 6. 调试
做完以上步骤后可以尝试进行训练调试(`肯定会有报错，不要慌`)
```
(stegastamp-env) [Stegastamp] # python3 train.py mytrain
```
### 6.1 报错 1
```
Traceback (most recent call last):
  File "train.py", line 224, in <module>
    main()
  File "train.py", line 91, in main
    config = tf.ConfigProto()
AttributeError: module 'tensorflow' has no attribute 'ConfigProto'
```
原因：原项目使用的 tensorflow 版本为 `1.x` 现在python3.8 对应的版本已经升级为 `2.x`<br> 
解决方法: 按以下规则修改 `train.py`
```
使用 import tensorflow.compat.v1 as tf
替换 import tensorflow as tf
```

### 6.2 报错 2
```
Traceback (most recent call last):
  File "train.py", line 224, in <module>
    main()
  File "train.py", line 98, in main
    secret_pl = tf.placeholder(shape=[None,args.secret_size],dtype=tf.float32,name="input_prep")
  File "/home/chuan/python-envs/stegastamp-env/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py", line 3097, in placeholder
    raise RuntimeError("tf.placeholder() is not compatible with "
RuntimeError: tf.placeholder() is not compatible with eager execution.
```
解决方法：在 `train.py` 头部增加以下语句
```
tf.compat.v1.disable_eager_execution()
```

### 6.3 报错 3
```
Traceback (most recent call last):
  File "train.py", line 224, in <module>
    main()
  File "train.py", line 112, in main
    loss_op, secret_loss_op, D_loss_op, summary_op, image_summary_op, _ = models.build_model(
  File "/home/chuan/StegaStamp/models.py", line 198, in build_model
    input_warped = tf.contrib.image.transform(image_input, M[:,1,:], interpolation='BILINEAR')
AttributeError: module 'tensorflow' has no attribute 'contrib'
```
原因： `tensorflow 2.x`开始不带有 `contrib` 这个库。<br>
解决方法： 使用 `tensorflow_addons` (如果没有这个包 请 `pip3 install tensorflow_addons`)
```
import tensorflow_addons as tfa
使用 tfa.image.transform
替换 tf.contrib.image.transform
```

### 6.4 报错 4
```
Traceback (most recent call last):
  File "train.py", line 224, in <module>
    main()
  File "train.py", line 112, in main
    loss_op, secret_loss_op, D_loss_op, summary_op, image_summary_op, _ = models.build_model(
  File "/home/chuan/StegaStamp/models.py", line 239, in build_model
    transformed_image, transform_summaries = transform_net(encoded_image, args, global_step)
  File "/home/chuan/StegaStamp/models.py", line 141, in transform_net
    f = utils.random_blur_kernel(probs=[.25,.25], N_blur=7,
  File "/home/chuan/StegaStamp/utils.py", line 10, in random_blur_kernel
    coords = tf.to_float(tf.stack(tf.meshgrid(tf.range(N_blur), tf.range(N_blur), indexing='ij'), -1)) - (.5 * (N-1))
AttributeError: module 'tensorflow' has no attribute 'to_float'
```
原因：`Tensorflow 2.x` 不支持 `to_float` 函数。<br>
解决方法：
```
在 utils.py 文件下 
使用 import tensorflow.compat.v1 as tf
替换 import tensorflow as tf
```
### 6.4 报错 5
```
Traceback (most recent call last):
  File "train.py", line 224, in <module>
    main()
  File "train.py", line 112, in main
    loss_op, secret_loss_op, D_loss_op, summary_op, image_summary_op, _ = models.build_model(
  File "/home/chuan/StegaStamp/models.py", line 245, in build_model
    lpips_loss_op = tf.reduce_mean(lpips_tf.lpips(image_input, encoded_image))
  File "/home/chuan/StegaStamp/lpips/lpips_tf.py", line 60, in lpips
    default_graph = tf.get_default_graph()
AttributeError: module 'tensorflow' has no attribute 'get_default_graph'
```
原因：`Tensorflow 2.x` 不支持 `get_default_graph` 函数。<br>
解决方法：
```
在 lpips/lpips_tf.py 文件下 
使用 import tensorflow.compat.v1 as tf
替换 import tensorflow as tf
```

### 6.5 报错 6 （结束）
到这里解决完以上报错再运行`train.py`,会开始下载预训练的 `alexnet` 网络文件
```
>> Downloading net-lin_alex_v0.1.pb 100.1%
```
然后报错 :) 
```
Traceback (most recent call last):
  File "train.py", line 224, in <module>
    main()
  File "train.py", line 159, in main
    images, secrets = get_img_batch(files_list=files_list,
  File "train.py", line 28, in get_img_batch
    img_cover_path = random.choice(files_list)
  File "/usr/lib/python3.8/random.py", line 290, in choice
    raise IndexError('Cannot choose from an empty sequence') from None
IndexError: Cannot choose from an empty sequence
```
原因：训练文件夹内没有图片。<br>
解决方法：修改 train 的图片文件夹路径
## 7. 图像加密
### 7.1 下载预训练的模型并解压
```
(stegastamp-env) [Stegastamp] # wget http://people.eecs.berkeley.edu/~tancik/stegastamp/saved_models.tar.xz
(stegastamp-env) [Stegastamp] # tar -xJf saved_models.tar.xz
(stegastamp-env) [Stegastamp] # rm saved_models.tar.xz
```
### 7.2 使用 `encoder_image.py` 对图像进行加密
```
(stegastamp-env) [Stegastamp] # python3 encode_image.py \
  saved_models/stegastamp_pretrained \
  --image test_im.png  \
  --save_dir out/ \
  --secret Qin'sLab
```
如果看到该目录下多了一个out文件夹并且里面有加密的图像和其掩膜证明环境配置完成并且代码运行正常。
## 参考
1. 安装显卡 https://www.server-world.info/en/note?os=Ubuntu_20.04&p=nvidia
2. 什么是initramfs? https://www.cnblogs.com/tongyishu/p/12085153.html
3. 安装cuda10.1 https://www.server-world.info/en/note?os=Ubuntu_20.04&p=cuda&f=1
4. python虚拟环境配置 https://docs.python.org/zh-cn/3/tutorial/venv.html
5. 报错2参考 https://stackoverflow.com/questions/56561734/runtimeerror-tf-placeholder-is-not-compatible-with-eager-execution
6. 报错4参考 https://github.com/tensorflow/tensor2tensor/issues/1736
7. 官方代码库 https://github.com/tancik/StegaStamp

```
作者：Justzyzhang
github: https://github.com/slimhappy
```