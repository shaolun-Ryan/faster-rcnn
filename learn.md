learn
===
* 整个的Faster-rcnn模块其实就是一个`nn.Module`的Class， 实现了一次正向传播。
传播的模块经过了四个：
    * transform预处理
    * backbone特征提取
    * rpn预测框的学习
    * RoI bbox的回归和label的分类
    
其中transform虽然在forward的步骤中，但是没有任何需要训练的参数（`model.transform.parameters()`的个数为零）。
其余的三个模块是自带训练参数需要更新的。

* 安装其实就包括三部即可运行：
    1. 准备datasets
    2. 下载requirement lib
    3. 准备预训练模型