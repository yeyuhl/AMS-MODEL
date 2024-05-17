from itertools import product as product
from math import ceil

import torch


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        #   读取最小尺寸
        self.min_sizes = cfg['min_sizes']
        #   读取步长
        self.steps = cfg['steps']
        #   读取裁剪阈值
        self.clip = cfg['clip']
        # ---------------------------#
        #   图片的尺寸
        # ---------------------------#
        self.image_size = image_size
        # ---------------------------#
        #   三个有效特征层高和宽
        # ---------------------------#
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    #   生成预测框
    def get_anchors(self):
        anchors = []
        #   遍历特征层
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # -----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            # -----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        #   将所有预测框转换为Tensor格式，然后将其形状调整为(-1, 4)
        #   -1表示自动推断行数，以便将所有锚框数据包含在Tensor中，4表示每行包含4个元素，分别代表锚框的中心点坐标和宽高。
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
