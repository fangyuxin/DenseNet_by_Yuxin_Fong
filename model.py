
import torch
import torch.nn as nn
from config import cfg



class _InitLayer(nn.Sequential):

    def __init__(self, num_input_features, num_init_features):
        super().__init__()

        self.init_layer = nn.Sequential()

        self.init_layer.add_module('Conv', nn.Conv2d(num_input_features, num_init_features, \
                                                     kernel_size=7, padding=3, stride=2))
        self.init_layer.add_module('Pool', nn.MaxPool2d(kernel_size=3, stride=2))

    def forward(self, x):
        return self.init_layer.forward(x)



class _EndLayer(nn.Sequential):

    def __init__(self, num_features, num_output_classes):
        super().__init__()

        self.num_features = num_features
        self.end_layer = nn.Sequential()

        self.end_layer.add_module('Avgpool', nn.AvgPool2d(kernel_size=6, stride=1))
        self.end_layer.add_module('FCN', nn.Linear(num_features, num_output_classes))

    def forward(self, x):
        return self.end_layer.FCN(self.end_layer.Avgpool(x).view(x.size(0), -1))




class _DenseLayer(nn.Sequential):

    def __init__(self, num_features, bn_size, growth_rate):
        super().__init__()

        self.dense_layer = nn.Sequential()

        self.dense_layer.add_module('BN_1', nn.BatchNorm2d(num_features))
        self.dense_layer.add_module('ReLU_1', nn.ReLU(inplace=True))
        self.dense_layer.add_module('Conv1_bn', nn.Conv2d(num_features, bn_size * growth_rate, \
                                           kernel_size=1, padding=0, stride=1, bias=False))
        self.dense_layer.add_module('BN_2', nn.BatchNorm2d(bn_size * growth_rate))
        self.dense_layer.add_module('ReLU_2', nn.ReLU(inplace=True))
        self.dense_layer.add_module('Conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, \
                                           kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, x):
        new_features = self.dense_layer.forward(x)
        return torch.cat([x, new_features], 1)




class _DenseBlock(nn.Sequential):

    def __init__(self, num_features, num_denselayer, bn_size, growth_rate):
        super().__init__()

        self.dense_block = nn.Sequential()

        for l in range(num_denselayer):
            self.dense_block.add_module('denselayer_{}'.format(l + 1), _DenseLayer(num_features,
                                                                                   bn_size,
                                                                                   growth_rate))
            num_features += growth_rate




class _TransLayer(nn.Sequential):

    def __init__(self, num_features, theta):
        super().__init__()

        self.trans_layer = nn.Sequential()

        self.trans_layer.add_module('BN', nn.BatchNorm2d(num_features))
        self.trans_layer.add_module('ReLU', nn.ReLU(inplace=True))
        self.trans_layer.add_module('Conv', nn.Conv2d(num_features, int(num_features * theta),
                                                      kernel_size=1, padding=0, stride=1, bias=False))
        self.trans_layer.add_module('Avgpool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.trans_layer.forward(x)




class DenseNet(nn.Module):

    '''
    num_input_features: 输入到DenseNet的输入通道数, 一般为1(灰度图)或者3(RGB图).
    num_init_features: 输入至第一个DenseBlock的特征映射的数量.
    num_output_class: DenseNet输出的通道数, 即需要分类的总数.
    block_cfg: 一般为一个长度为3或者4的数组, 数组的长度表示DenseBlock的个数,
    数组的每一个元素为每一个DenseBlock中DenseLayer的个数.
    bn_size: DenseLayer中bottleneck层输出特征映射的个数.
    growth_rate: DenseLayer输出的特征映射的数量, 下个输入增加的特征映射的数量.
    theta: 每个DenseBlack输出时，输出的特征映射相对输入的特征映射的压缩比例.
    '''

    def __init__(self,
                 num_input_features=3,
                 num_init_features=64,
                 num_output_classes=1000,
                 block_cfg=(6, 12, 24, 16),
                 bn_size=4,
                 growth_rate=32,
                 theta=0.5):
        super().__init__()

        # 初始化一个densenet.
        self.densenet = nn.Sequential()
        num_features = num_init_features

        # 为densenet加入最开始部分的的卷基层.
        self.densenet.add_module('init', _InitLayer(num_init_features,
                                                    num_input_features))

        # 为densenet加入denseblock.
        for i, num_denselayer in enumerate(block_cfg, start=1):
            self.densenet.add_module('denseblock_{}'.format(i), _DenseBlock(num_features,
                                                                            num_denselayer,
                                                                            bn_size,
                                                                            growth_rate
                                                                            ))
            num_features = num_features + block_cfg[i - 1] * growth_rate

            # 为densenet加入transition_layer.
            if i != len(block_cfg):
                self.densenet.add_module('translayer_{}'.format(i), _TransLayer(num_features, theta))
                num_features = int(num_features * theta)

        # 为densenet加入FCN, 做最终输出.
        self.densenet.add_module('end', _EndLayer(num_features,
                                                  num_output_classes))



    def forward(self, x):

        return self.densenet(x)



# 一个实例.

model = DenseNet(**cfg.param_dict)

print(model._modules)












