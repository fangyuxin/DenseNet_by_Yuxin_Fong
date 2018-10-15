

class DefaultConfig():

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

    param_dict = {
        'num_input_features': 3,
        'num_init_features': 64,
        'num_output_classes': 1000,
        'block_cfg': (6, 12, 24, 16),
        'bn_size': 4,
        'growth_rate': 2,
        'theta': 0.5
    }


cfg = DefaultConfig()
